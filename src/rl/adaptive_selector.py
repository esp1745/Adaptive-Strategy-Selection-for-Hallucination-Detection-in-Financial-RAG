"""Adaptive detector selection with a lightweight contextual bandit."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import math
import random
import re

FEATURE_NAMES = (
    "question_len",
    "entity_count",
    "numeric_density",
    "has_percentage",
    "has_comparison",
    "complexity_score",
    "context_chunks",
    "retrieval_confidence",
    "top_chunk_score",
    "score_gap",
    "answer_len",
    "answer_numeric_density",
)

KNOWN_COMPANIES = (
    "apple",
    "microsoft",
    "tesla",
    "amazon",
    "google",
    "alphabet",
    "meta",
    "nvidia",
)


@dataclass
class DetectorPrior:
    name: str
    accuracy: float
    avg_latency_ms: float
    cost: float


@dataclass
class DetectorStats:
    pulls: int = 0
    reward_sum: float = 0.0
    correct_sum: int = 0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls > 0 else 0.0

    @property
    def empirical_accuracy(self) -> float:
        return self.correct_sum / self.pulls if self.pulls > 0 else 0.0


class AdaptiveDetectorSelector:
    """Epsilon-greedy contextual bandit for detector routing."""

    def __init__(
        self,
        detector_priors: Dict[str, DetectorPrior],
        epsilon: float = 0.12,
        prior_strength: int = 25,
        exploration_bonus: float = 0.15,
    ):
        if not detector_priors:
            raise ValueError("detector_priors cannot be empty")

        self.detector_priors = detector_priors
        self.epsilon = epsilon
        self.prior_strength = prior_strength
        self.exploration_bonus = exploration_bonus

        self.stats: Dict[str, DetectorStats] = {}
        for name, prior in detector_priors.items():
            pulls = max(1, prior_strength)
            reward_sum = pulls * self._prior_reward(prior)
            correct_sum = int(round(prior.accuracy * pulls))
            self.stats[name] = DetectorStats(
                pulls=pulls,
                reward_sum=reward_sum,
                correct_sum=correct_sum,
            )

        self.total_steps = sum(s.pulls for s in self.stats.values())

    def _prior_reward(self, prior: DetectorPrior) -> float:
        return self.compute_reward(
            correct=1,
            confidence=prior.accuracy,
            cost=prior.cost,
            latency_ms=prior.avg_latency_ms,
        ) * prior.accuracy

    @staticmethod
    def extract_state_features(
        question: str,
        context: Optional[List[str]] = None,
        answer: str = "",
        retrieval_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        context = context or []
        retrieval_stats = retrieval_stats or {}
        normalized_question = question or ""
        query_tokens = re.findall(r"\b\w+\b", normalized_question.lower())
        num_matches = re.findall(r"\d+(?:\.\d+)?", normalized_question)
        entity_matches = {
            match.strip()
            for match in re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", normalized_question)
        }

        lowered_question = normalized_question.lower()
        for company in KNOWN_COMPANIES:
            if company in lowered_question:
                entity_matches.add(company.title())

        answer_tokens = re.findall(r"\b\w+\b", answer.lower())
        answer_num_matches = re.findall(r"\d+(?:\.\d+)?", answer)

        question_len = len(query_tokens)
        numeric_density = len(num_matches) / max(1, question_len)
        has_percentage = 1.0 if "%" in normalized_question else 0.0
        has_comparison = 1.0 if any(
            k in lowered_question
            for k in ["compare", "increase", "decrease", "growth", "change", "versus", "difference"]
        ) else 0.0
        complexity_score = min(
            1.0,
            0.35 * min(question_len / 20.0, 1.0)
            + 0.2 * min(len(entity_matches) / 3.0, 1.0)
            + 0.2 * numeric_density
            + 0.15 * has_percentage
            + 0.1 * has_comparison,
        )
        context_chunks = float(len(context))
        answer_len = float(len(answer_tokens))
        answer_numeric_density = len(answer_num_matches) / max(1.0, answer_len)

        top_chunk_score = float(retrieval_stats.get("top_chunk_score", retrieval_stats.get("top_score", 0.0)))
        retrieval_confidence = float(
            retrieval_stats.get(
                "retrieval_confidence",
                min(0.95, 0.35 + 0.15 * context_chunks + 0.1 * max(top_chunk_score, 0.0)),
            )
        )
        score_gap = float(retrieval_stats.get("score_gap", 0.0))

        return {
            "question_len": float(question_len),
            "entity_count": float(len(entity_matches)),
            "numeric_density": float(numeric_density),
            "has_percentage": has_percentage,
            "has_comparison": has_comparison,
            "complexity_score": float(complexity_score),
            "context_chunks": context_chunks,
            "retrieval_confidence": retrieval_confidence,
            "top_chunk_score": top_chunk_score,
            "score_gap": score_gap,
            "answer_len": answer_len,
            "answer_numeric_density": float(answer_numeric_density),
        }

    @staticmethod
    def feature_names() -> List[str]:
        return list(FEATURE_NAMES)

    @staticmethod
    def state_to_vector(state: Dict[str, float]) -> List[float]:
        return [float(state.get(name, 0.0)) for name in FEATURE_NAMES]

    @staticmethod
    def compute_reward(
        correct: int,
        confidence: float,
        cost: float,
        latency_ms: float,
        quality_weight: float = 0.8,
        cost_weight: float = 0.5,
        latency_weight: float = 0.1,
    ) -> float:
        latency_penalty = min(latency_ms / 10000.0, 1.0)
        quality_signal = float(confidence) if correct else -float(confidence)
        reward = quality_weight * quality_signal - cost_weight * float(cost) - latency_weight * latency_penalty
        return reward

    @staticmethod
    def context_bonus_for_detector(detector_name: str, state: Dict[str, float]) -> float:
        numeric_density = state.get("numeric_density", 0.0)
        question_len = state.get("question_len", 0.0)
        has_comparison = state.get("has_comparison", 0.0)
        complexity_score = state.get("complexity_score", 0.0)
        retrieval_confidence = state.get("retrieval_confidence", 0.0)
        score_gap = state.get("score_gap", 0.0)
        answer_len = state.get("answer_len", 0.0)

        bonus = 0.0
        if detector_name == "token_overlap":
            bonus += 0.10 * numeric_density
            bonus += 0.05 * state.get("has_percentage", 0.0)
            bonus += 0.08 * retrieval_confidence
            bonus -= 0.05 * complexity_score
        elif detector_name == "semantic_similarity":
            bonus += 0.06 * max(0.0, 1.0 - numeric_density)
            bonus += 0.04 * retrieval_confidence
            bonus += 0.03 * min(answer_len / 25.0, 1.0)
        elif detector_name == "bert_nli":
            bonus += 0.08 * has_comparison
            bonus += 0.03 * min(question_len / 30.0, 1.0)
            bonus += 0.06 * complexity_score
            bonus += 0.04 * score_gap
        elif detector_name == "llm_judge":
            bonus += 0.10 * has_comparison
            bonus += 0.05 * min(question_len / 40.0, 1.0)
            bonus += 0.06 * numeric_density
            bonus += 0.08 * complexity_score
            bonus += 0.05 * retrieval_confidence

        return bonus

    def _context_bonus(self, detector_name: str, state: Dict[str, float]) -> float:
        return self.context_bonus_for_detector(detector_name, state)

    def select_detector(self, state: Dict[str, float]) -> str:
        detector_names = list(self.detector_priors.keys())

        if random.random() < self.epsilon:
            return random.choice(detector_names)

        best_name = detector_names[0]
        best_score = float("-inf")

        for name in detector_names:
            stats = self.stats[name]
            exploit = stats.mean_reward
            explore = self.exploration_bonus * math.sqrt(math.log(self.total_steps + 1) / max(1, stats.pulls))
            context = self._context_bonus(name, state)
            score = exploit + explore + context

            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def update(
        self,
        detector_name: str,
        reward: float,
        correct: int,
    ) -> None:
        stats = self.stats[detector_name]
        stats.pulls += 1
        stats.reward_sum += reward
        stats.correct_sum += int(correct)
        self.total_steps += 1

    def policy_snapshot(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for name, stats in self.stats.items():
            result[name] = {
                "stats": asdict(stats),
                "mean_reward": round(stats.mean_reward, 4),
                "empirical_accuracy": round(stats.empirical_accuracy, 4),
                "prior": asdict(self.detector_priors[name]),
            }
        return result
