"""Run the full dashboard workflow: query -> RAG -> PPO -> detector -> PPO update."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".claude" / ".env")
# Normalize mixed-case key name to the standard form expected by the OpenAI client
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OpenAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OpenAI_API_KEY"]

from src.detection import create_detector  # noqa: E402
from src.rl import (  # noqa: E402
    AdaptiveDetectorSelector,
    DetectorPrior,
    PPODetectorSelector,
    PPOTrainingConfig,
    runtime_ppo_update,
)


BENCHMARK_PATH = ROOT / "data" / "processed" / "benchmark_results.json"
BASE_POLICY_PATH = ROOT / "data" / "processed" / "rl_policy.json"
RUNTIME_POLICY_PATH = ROOT / "data" / "processed" / "rl_policy_runtime.json"
EXPERIENCE_PATH = ROOT / "data" / "processed" / "rl_experience_buffer.json"
DOCUMENTS_PATH = ROOT / "models" / "financial_rag" / "documents.json"

VECTOR_DIM = 256
MAX_BUFFER_SIZE = 64
KNOWN_COMPANIES = ("Apple", "Microsoft", "Tesla", "Amazon", "Google", "Alphabet", "Meta", "NVIDIA")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "year",
}
TOKEN_ALIASES = {
    "revenue": {"sales", "sale"},
    "sales": {"revenue"},
    "income": {"earnings", "profit"},
    "operating": {"operations"},
    "segments": {"segment", "business"},
    "segment": {"segments", "business"},
    "risk": {"risks", "factors"},
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def signal_tokens(text: str) -> set[str]:
    tokens = {token for token in tokenize(text) if token not in STOPWORDS}
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(TOKEN_ALIASES.get(token, set()))
    return expanded


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [part.strip() for part in parts if part.strip()]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def trim_text(text: str, limit: int = 420) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def complexity_label(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def hash_embed(text: str, dim: int = VECTOR_DIM) -> np.ndarray:
    counts = Counter(tokenize(text))
    vector = np.zeros(dim, dtype=np.float32)

    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log1p(count))

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def load_saved_corpus() -> List[Dict]:
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    documents = payload.get("documents", [])
    metadata = payload.get("metadata", [])
    corpus = []
    for index, text in enumerate(documents):
        meta = metadata[index] if index < len(metadata) else {}
        corpus.append(
            {
                "text": text,
                "company": meta.get("company"),
                "filing_type": meta.get("filing_type"),
                "filing_date": meta.get("filing_date"),
                "chunk_id": meta.get("chunk_id"),
            }
        )
    return corpus


def build_hash_faiss_index(corpus: List[Dict]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    embeddings = np.vstack([hash_embed(item["text"]) for item in corpus]).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def normalize_similarity(score: float) -> float:
    return clamp((score + 1.0) / 2.0, 0.0, 1.0)


def detect_company_hint(question: str) -> str | None:
    lowered = question.lower()
    for company in KNOWN_COMPANIES:
        if company.lower() in lowered:
            return company
    return None


def retrieve_chunks(question: str, top_k: int) -> Tuple[List[Dict], Dict[str, float]]:
    corpus = load_saved_corpus()
    index, _ = build_hash_faiss_index(corpus)
    query_embedding = hash_embed(question).reshape(1, -1).astype("float32")
    candidate_count = len(corpus)
    scores, indices = index.search(query_embedding, candidate_count)

    query_tokens = signal_tokens(question)
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?", question))
    company_hint = detect_company_hint(question)
    focus_terms = {
        token
        for token in query_tokens
        if token not in {"fiscal", "year", "total", "company"}
        and token != (company_hint or "").lower()
        and not token.isdigit()
    }
    ranked_candidates = []

    for idx, raw_score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        item = corpus[int(idx)]
        doc_tokens = signal_tokens(item["text"])
        lexical_overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
        focus_hits = len(focus_terms & doc_tokens) / max(1, len(focus_terms))
        number_matches = len(query_numbers & set(re.findall(r"\d+(?:\.\d+)?", item["text"])))
        company_match = 1.0 if company_hint and item.get("company") == company_hint else 0.0
        combined_score = (
            0.35 * normalize_similarity(float(raw_score))
            + 0.3 * lexical_overlap
            + 0.25 * focus_hits
            + 0.2 * company_match
            + 0.08 * number_matches
        )
        if focus_terms and focus_hits == 0:
            combined_score -= 0.18
        combined_score = clamp(combined_score, 0.0, 0.99)
        ranked_candidates.append((combined_score, float(raw_score), item))

    ranked_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    best_candidates = ranked_candidates[:top_k]

    chunks: List[Dict] = []
    normalized_scores = [float(score) for score, _, _ in best_candidates]
    top_score = normalized_scores[0] if normalized_scores else 0.0
    second_score = normalized_scores[1] if len(normalized_scores) > 1 else 0.0
    score_gap = max(0.0, top_score - second_score)
    retrieval_confidence = clamp(
        0.7 * top_score + 0.3 * clamp(score_gap * 3.0, 0.0, 1.0),
        0.05,
        0.99,
    )

    for score, raw_score, item in best_candidates:
        chunks.append(
            {
                "text": item["text"],
                "score": float(score),
                "rawScore": float(raw_score),
                "company": item.get("company"),
                "filingType": item.get("filing_type"),
                "filingDate": item.get("filing_date"),
                "chunkId": item.get("chunk_id"),
            }
        )

    return chunks, {
        "retrieval_confidence": retrieval_confidence,
        "top_chunk_score": float(top_score),
        "score_gap": float(score_gap),
    }


def select_answer_sentences(question: str, chunks: List[Dict]) -> List[str]:
    question_tokens = signal_tokens(question)
    question_numbers = set(re.findall(r"\d+(?:\.\d+)?", question))
    company_hint = detect_company_hint(question)
    focus_terms = {
        token
        for token in question_tokens
        if token not in {"fiscal", "year", "total", "company"}
        and token != (company_hint or "").lower()
        and not token.isdigit()
    }
    scored_sentences: List[Tuple[float, str]] = []

    for rank, chunk in enumerate(chunks):
        base_score = 1.0 - 0.12 * rank + float(chunk.get("score", 0.0))
        for sentence in split_sentences(chunk.get("text", "")):
            sentence_tokens = signal_tokens(sentence)
            overlap = len(question_tokens & sentence_tokens) / max(1, len(question_tokens))
            focus_hits = len(focus_terms & sentence_tokens) / max(1, len(focus_terms))
            sentence_numbers = set(re.findall(r"\d+(?:\.\d+)?", sentence))
            number_bonus = 0.18 * len(question_numbers & sentence_numbers)
            company_bonus = 0.12 if (chunk.get("company") or "") in question else 0.0
            length_penalty = 0.08 if len(sentence.split()) > 45 else 0.0
            score = base_score + overlap + 0.45 * focus_hits + number_bonus + company_bonus - length_penalty
            if focus_terms and focus_hits == 0:
                score -= 0.2
            scored_sentences.append((score, sentence.strip()))

    scored_sentences.sort(key=lambda pair: pair[0], reverse=True)
    selected: List[str] = []
    seen = set()
    for _, sentence in scored_sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        selected.append(sentence)
        seen.add(normalized)
        if len(selected) == 2:
            break
    return selected


def generate_answer(question: str, chunks: List[Dict], retrieval_stats: Dict[str, float]) -> Dict:
    chosen_sentences = select_answer_sentences(question, chunks)
    if not chosen_sentences:
        answer = "The retrieved filing chunks did not contain enough detail to synthesize a grounded answer."
    elif len(chosen_sentences) == 1:
        answer = chosen_sentences[0]
    else:
        answer = f"{chosen_sentences[0]} {chosen_sentences[1]}"

    return {
        "answer": answer,
        "generator": "grounded_synthesis",
        "confidence": retrieval_stats["retrieval_confidence"],
        "sourceSentences": chosen_sentences,
    }


def load_detector(detector_name: str):
    if detector_name == "llm_judge":
        return create_detector(detector_name, backend="ollama", model="qwen2.5:7b")
    return create_detector(detector_name)


def summarize_result(detector_name: str, result_details: Dict) -> List[str]:
    details = result_details or {}

    if detector_name == "token_overlap":
        overlap = details.get("overlap_score")
        number_match = details.get("number_match_score")
        return [
            f"Overlap score: {overlap:.3f}" if isinstance(overlap, (int, float)) else "Overlap score unavailable",
            f"Number match score: {number_match:.3f}" if isinstance(number_match, (int, float)) else "Number match score unavailable",
            f"Threshold: {details.get('threshold')}",
        ]

    if detector_name == "semantic_similarity":
        similarity = details.get("aggregated_similarity")
        aggregation = details.get("aggregation_method")
        return [
            f"Aggregated similarity: {similarity:.3f}" if isinstance(similarity, (int, float)) else "Aggregated similarity unavailable",
            f"Aggregation: {aggregation}" if aggregation else "Aggregation unavailable",
            f"Threshold: {details.get('threshold')}",
        ]

    if detector_name == "bert_nli":
        scores = details.get("nli_scores") or []
        contradictions = [score.get("contradiction", 0.0) for score in scores if isinstance(score, dict)]
        entailments = [score.get("entailment", 0.0) for score in scores if isinstance(score, dict)]
        return [
            f"Max contradiction: {max(contradictions):.3f}" if contradictions else "Max contradiction unavailable",
            f"Max entailment: {max(entailments):.3f}" if entailments else "Max entailment unavailable",
            f"Chunks processed: {details.get('num_chunks_processed', 0)}",
        ]

    if detector_name == "llm_judge":
        issues = details.get("specific_issues") or []
        if issues:
            return [str(issue) for issue in issues[:3]]
        return [
            f"Backend: {details.get('backend', 'unknown')}",
            f"Model: {details.get('model', 'unknown')}",
            f"Threshold: {details.get('threshold')}",
        ]

    return ["Detector details unavailable"]


def load_benchmark_priors() -> Tuple[Dict[str, DetectorPrior], Dict]:
    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    priors: Dict[str, DetectorPrior] = {}
    for name, entry in raw.items():
        priors[name] = DetectorPrior(
            name=name,
            accuracy=float(entry["accuracy"]),
            avg_latency_ms=float(entry["avg_latency_ms"]),
            cost=float(entry["cost"]),
        )
    return priors, raw


def load_policy_payload() -> Tuple[Dict, Path]:
    if RUNTIME_POLICY_PATH.exists():
        with RUNTIME_POLICY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f), RUNTIME_POLICY_PATH
    with BASE_POLICY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f), BASE_POLICY_PATH


def detector_reasons(detector_name: str, state: Dict[str, float], benchmark_entry: Dict) -> List[str]:
    reasons = []
    if state.get("numeric_density", 0.0) > 0.05:
        reasons.append("Numeric cues are strong, so low-cost factual matching becomes attractive.")
    if state.get("has_comparison", 0.0) > 0:
        reasons.append("Comparison language raises reasoning complexity, which benefits stronger consistency checks.")
    if state.get("retrieval_confidence", 0.0) > 0.75:
        reasons.append("Retrieval confidence is high, so the policy can trust grounding evidence more heavily.")

    quality = benchmark_entry.get("f1", benchmark_entry.get("accuracy", 0.0))
    reasons.append(f"Benchmark quality for this detector is {quality * 100:.1f}% with cost ${benchmark_entry['cost']:.2f}.")
    return reasons[:3]


def select_detector_with_policy(
    question: str,
    answer: str,
    context: List[str],
    retrieval_stats: Dict[str, float],
    priors: Dict[str, DetectorPrior],
    benchmark: Dict,
    policy_payload: Dict,
) -> Tuple[Dict, Dict]:
    state = AdaptiveDetectorSelector.extract_state_features(
        question=question,
        context=context,
        answer=answer,
        retrieval_stats=retrieval_stats,
    )

    PPO_EPSILON = 0.15  # 15% random exploration to prevent mode collapse

    ppo_load_error: str | None = None
    if policy_payload.get("algorithm") == "ppo" and policy_payload.get("policy_model"):
        try:
            selector = PPODetectorSelector.from_policy_payload(priors, policy_payload)
            probabilities = selector.action_probabilities(state)
            explore = random.random() < PPO_EPSILON
            if explore:
                selected_detector = random.choice(list(probabilities.keys()))
            else:
                selected_detector = max(probabilities.items(), key=lambda item: item[1])[0]
            selected_probability = float(probabilities[selected_detector])
            state_vector = AdaptiveDetectorSelector.state_to_vector(state)
            normalized_vector = selector.normalize_vector(state_vector).tolist()
            state_value = selector.state_value(state)

            state_tensor = selector._state_tensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = selector.actor(state_tensor)
                distribution = Categorical(logits=logits)
                action_index = selector.detector_names.index(selected_detector)
                old_log_prob = float(distribution.log_prob(torch.tensor(action_index)).item())

            policy_info = {
                "algorithm": "ppo",
                "stateDim": len(state_vector),
                "stateFeatureNames": selector.feature_names,
                "stateVector": [float(value) for value in state_vector],
                "normalizedStateVector": [float(value) for value in normalized_vector],
                "selectedDetector": selected_detector,
                "selectedProbability": selected_probability,
                "probabilities": probabilities,
                "explorationMode": explore,
                "actorHiddenSize": selector.hidden_size,
                "criticHiddenSize": selector.hidden_size,
                "stateValue": state_value,
                "reasons": detector_reasons(selected_detector, state, benchmark[selected_detector]),
                "policyFile": str(RUNTIME_POLICY_PATH if RUNTIME_POLICY_PATH.exists() else BASE_POLICY_PATH),
            }
            runtime_context = {
                "selector": selector,
                "state": state,
                "state_vector": normalized_vector,
                "action_index": action_index,
                "old_log_prob": old_log_prob,
            }
            return policy_info, runtime_context
        except Exception as error:
            ppo_load_error = str(error)

    selector = AdaptiveDetectorSelector(detector_priors=priors)
    selected_detector = selector.select_detector(state)
    probability = 1.0 / max(1, len(priors))
    reasons = detector_reasons(selected_detector, state, benchmark[selected_detector])
    if ppo_load_error:
        reasons.insert(0, f"PPO policy snapshot was incompatible with the current state schema, so the router used fallback selection: {ppo_load_error}")
    policy_info = {
        "algorithm": "fallback",
        "stateDim": len(AdaptiveDetectorSelector.feature_names()),
        "stateFeatureNames": selector.feature_names(),
        "stateVector": AdaptiveDetectorSelector.state_to_vector(state),
        "normalizedStateVector": AdaptiveDetectorSelector.state_to_vector(state),
        "selectedDetector": selected_detector,
        "selectedProbability": probability,
        "probabilities": {name: probability for name in priors.keys()},
        "explorationMode": False,
        "actorHiddenSize": 0,
        "criticHiddenSize": 0,
        "stateValue": 0.0,
        "reasons": reasons,
        "policyFile": str(BASE_POLICY_PATH),
    }
    runtime_context = {
        "selector": None,
        "state": state,
        "state_vector": AdaptiveDetectorSelector.state_to_vector(state),
        "action_index": list(priors.keys()).index(selected_detector),
        "old_log_prob": math.log(max(probability, 1e-8)),
    }
    return policy_info, runtime_context


def load_experience_state() -> Dict:
    if not EXPERIENCE_PATH.exists():
        return {"experiences": [], "action_counts": {}, "updates": 0}
    with EXPERIENCE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_experience_state(payload: Dict) -> None:
    EXPERIENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIENCE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_runtime_policy(base_payload: Dict, selector: PPODetectorSelector) -> None:
    runtime_payload = dict(base_payload)
    runtime_payload["policy_model"] = selector.export_policy()
    runtime_payload["feature_schema"] = {"names": selector.feature_names}
    runtime_payload["runtime_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with RUNTIME_POLICY_PATH.open("w", encoding="utf-8") as f:
        json.dump(runtime_payload, f, indent=2)


def format_policy_hint(question: str, detector_name: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ["revenue", "operating", "income", "sales", "percent"]):
        return f"Numeric financial query -> {detector_name}"
    if any(token in lowered for token in ["risk", "compare", "change", "growth"]):
        return f"Higher-complexity reasoning -> {detector_name}"
    return f"Current query pattern -> {detector_name}"


def update_policy_runtime(
    question: str,
    reward: float,
    policy_payload: Dict,
    policy_info: Dict,
    runtime_context: Dict,
) -> Dict:
    buffer_state = load_experience_state()
    experiences = list(buffer_state.get("experiences", []))
    action_counts = dict(buffer_state.get("action_counts", {}))
    action_counts[policy_info["selectedDetector"]] = action_counts.get(policy_info["selectedDetector"], 0) + 1

    experiences.append(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": question,
            "state_vector": runtime_context["state_vector"],
            "action_index": runtime_context["action_index"],
            "action_name": policy_info["selectedDetector"],
            "reward": reward,
            "old_log_prob": runtime_context["old_log_prob"],
            "next_state": runtime_context["state_vector"],
        }
    )
    experiences = experiences[-MAX_BUFFER_SIZE:]

    updated = False
    avg_reward = reward
    policy_path = policy_info["policyFile"]

    if policy_payload.get("algorithm") == "ppo" and runtime_context.get("selector") is not None:
        config_data = policy_payload.get("config", {})
        training_config = PPOTrainingConfig(
            rollout_size=min(MAX_BUFFER_SIZE, int(config_data.get("rollout_size", 32))),
            update_epochs=max(2, int(config_data.get("update_epochs", 4))),
            minibatch_size=min(len(experiences), max(1, int(config_data.get("minibatch_size", 16)))),
            learning_rate=float(config_data.get("learning_rate", 3e-4)),
            hidden_size=int(config_data.get("hidden_size", policy_info["actorHiddenSize"] or 32)),
            clip_epsilon=float(config_data.get("clip_epsilon", 0.2)),
            value_coef=float(config_data.get("value_coef", 0.5)),
            entropy_coef=float(config_data.get("entropy_coef", 0.01)),
            max_grad_norm=float(config_data.get("max_grad_norm", 0.5)),
        )
        update_result = runtime_ppo_update(runtime_context["selector"], experiences, training_config)
        updated = bool(update_result.get("updated"))
        avg_reward = float(update_result.get("avg_reward", reward))
        save_runtime_policy(policy_payload, runtime_context["selector"])
        policy_path = str(RUNTIME_POLICY_PATH)

    buffer_state = {
        "experiences": experiences,
        "action_counts": action_counts,
        "updates": int(buffer_state.get("updates", 0)) + (1 if updated else 0),
        "last_reward": reward,
    }
    save_experience_state(buffer_state)

    return {
        "bufferSize": len(experiences),
        "reward": reward,
        "updated": updated,
        "formula": "reward = 0.8 * quality_signal - 0.5 * detector_cost - 0.1 * latency_penalty",
        "message": format_policy_hint(question, policy_info["selectedDetector"]),
        "recentActionCounts": action_counts,
        "averageBufferReward": avg_reward,
        "policyFile": policy_path,
        "replayPreview": experiences[-5:],
    }


def count_named_entities(question: str) -> int:
    entities = {
        match.strip()
        for match in re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", question)
    }
    lowered = question.lower()
    for company in KNOWN_COMPANIES:
        if company.lower() in lowered:
            entities.add(company)
    return len(entities)


def main() -> None:
    payload = json.load(sys.stdin)
    question = (payload.get("question") or "").strip()
    top_k = int(payload.get("top_k") or 3)

    if not question:
        print(json.dumps({"ok": False, "error": "question is required"}))
        return

    started = time.time()
    priors, benchmark = load_benchmark_priors()
    policy_payload, _ = load_policy_payload()

    retrieval_start = time.time()
    chunks, retrieval_stats = retrieve_chunks(question, top_k=top_k)
    retrieval_latency_ms = (time.time() - retrieval_start) * 1000
    generated = generate_answer(question, chunks, retrieval_stats)
    answer = generated["answer"]
    context = [chunk["text"] for chunk in chunks]

    policy_info, runtime_context = select_detector_with_policy(
        question=question,
        answer=answer,
        context=context,
        retrieval_stats=retrieval_stats,
        priors=priors,
        benchmark=benchmark,
        policy_payload=policy_payload,
    )

    selected_detector = policy_info["selectedDetector"]
    actual_detector = selected_detector
    warning_messages: List[str] = []

    try:
        detector = load_detector(selected_detector)
        result = detector.detect(question, answer, context)
    except Exception as detector_error:
        warning_messages.append(
            f"{selected_detector} could not run locally and fell back to token_overlap: {detector_error}"
        )
        actual_detector = "token_overlap"
        detector = create_detector("token_overlap")
        result = detector.detect(question, answer, context)

    llm_baseline_cost = priors["llm_judge"].cost if "llm_judge" in priors else max(prior.cost for prior in priors.values())
    actual_prior = priors[actual_detector]
    reward = AdaptiveDetectorSelector.compute_reward(
        correct=0 if result.is_hallucinated else 1,
        confidence=result.confidence,
        cost=actual_prior.cost,
        latency_ms=result.latency_ms,
    )

    training_info = update_policy_runtime(
        question=question,
        reward=reward,
        policy_payload=policy_payload,
        policy_info=policy_info,
        runtime_context=runtime_context,
    )

    state_lookup = {
        name: float(value)
        for name, value in zip(policy_info["stateFeatureNames"], policy_info["stateVector"])
    }
    complexity = state_lookup.get("complexity_score", 0.0)

    response = {
        "ok": True,
        "workflow": {
            "question": question,
            "queryProcessor": {
                "wordCount": int(state_lookup.get("question_len", 0.0)),
                "entityCount": count_named_entities(question),
                "complexity": round(complexity, 3),
                "complexityLabel": complexity_label(complexity),
                "features": state_lookup,
            },
            "rag": {
                "retrieval": {
                    "method": "FAISS Vector Search",
                    "topK": top_k,
                    "confidence": retrieval_stats["retrieval_confidence"],
                    "topChunkScore": retrieval_stats["top_chunk_score"],
                    "scoreGap": retrieval_stats["score_gap"],
                    "latencyMs": retrieval_latency_ms,
                    "chunks": [
                        {
                            **chunk,
                            "text": trim_text(chunk["text"]),
                        }
                        for chunk in chunks
                    ],
                },
                "generation": {
                    "method": "LLM-style grounded synthesis",
                    "engine": generated["generator"],
                    "confidence": generated["confidence"],
                    "answer": answer,
                    "sourceSentences": generated["sourceSentences"],
                },
            },
            "policy": policy_info,
            "detectorExecution": {
                "requestedDetector": selected_detector,
                "actualDetector": actual_detector,
                "verdict": "hallucinated" if result.is_hallucinated else "grounded",
                "verdictLabel": "Potential hallucination" if result.is_hallucinated else "Verified",
                "confidence": result.confidence,
                "hallucinationScore": result.hallucination_score,
                "explanation": result.explanation,
                "detailLines": summarize_result(actual_detector, result.details or {}),
                "latencyMs": result.latency_ms,
                "cost": actual_prior.cost,
                "warning": " ".join(warning_messages) if warning_messages else None,
            },
            "resultAssembler": {
                "answer": answer,
                "status": "Verified" if not result.is_hallucinated else "Needs review",
                "detector": actual_detector,
                "cost": actual_prior.cost,
                "latencyMs": result.latency_ms,
                "savingsVsBaseline": max(llm_baseline_cost - actual_prior.cost, 0.0),
            },
            "trainingUpdate": training_info,
            "totalLatencyMs": (time.time() - started) * 1000,
        },
    }
    print(json.dumps(response))


if __name__ == "__main__":
    main()
