# Adaptive Strategy Selection for Hallucination Detection in Financial RAG

**Research Question:** Can a reinforcement learning agent learn when to use expensive versus cheap hallucination detection methods, achieving higher accuracy at lower average cost than any static strategy?

A PPO-trained neural policy observes 12-dimensional query features and adaptively routes each question to the most cost-efficient hallucination detector. The policy learns from every query and improves over time.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Hallucination Detectors](#hallucination-detectors)
4. [RL Policy](#rl-policy)
5. [Answer Synthesis](#answer-synthesis)
6. [Benchmarking](#benchmarking)
7. [Dashboard](#dashboard)
8. [Installation & Usage](#installation--usage)
9. [Key Files Reference](#key-files-reference)

---

## Architecture

```
Financial Query
      │
      ▼
Feature Extraction (12-dim state vector)
      │
      ▼
PPO Neural Policy ◄─── reward signal (updates after every query)
      │
      ▼
Selected Detector (Token Overlap / Semantic Similarity / BERT NLI / LLM Judge)
      │
      ▼
Hallucination Verdict + Confidence
```

**Pipeline steps:**
1. Query is parsed into a 12-dimensional state vector (word count, entity count, numeric density, complexity score, retrieval confidence, etc.)
2. PPO actor network outputs a probability distribution over 4 detectors
3. The selected detector checks the RAG-generated answer against retrieved SEC filing chunks
4. A reward is computed (`0.8 × quality_signal − 0.5 × detector_cost − 0.1 × latency_penalty`)
5. The experience is buffered and the PPO policy is updated in-place

**Key design decisions:**
- **Adaptive, not fixed** — the policy learns which detector works best per query type rather than applying a static rule
- **Cost-efficient** — avoids expensive LLM calls when a lightweight checker is sufficient
- **Continuously improving** — every query updates the policy weights via on-policy PPO

---

## Project Structure

```
Adaptive-Strategy-Selection/
├── dashboard/                      # Next.js web dashboard (primary UI)
│   ├── src/app/page.tsx            # Main dashboard UI (~1100 lines)
│   ├── src/app/api/verify/route.ts # API route — spawns verify_query.py subprocess
│   ├── src/lib/rlSelection.ts      # TypeScript re-implementation of bandit/PPO
│   └── scripts/verify_query.py    # Full pipeline: query → RAG → PPO → detector → update
├── src/
│   ├── detection/                  # Hallucination detectors
│   │   ├── base.py                 # Abstract base class + DetectionResult dataclass
│   │   ├── token_overlap.py        # Lexical overlap (BLEU/ROUGE-style)
│   │   ├── semantic_similarity.py  # Cosine similarity via sentence-transformers
│   │   ├── bert_nli.py             # NLI via facebook/bart-large-mnli
│   │   └── llm_judge.py            # LLM-as-judge (Ollama/OpenAI/Anthropic/mock)
│   └── rl/
│       ├── adaptive_selector.py    # Epsilon-greedy contextual bandit
│       ├── ppo_selector.py         # PPO neural policy (actor-critic)
│       └── __init__.py
├── data/
│   ├── raw/sec_filings/            # Downloaded SEC 10-K filings
│   └── processed/
│       ├── benchmark_results.json  # Detector benchmark scores
│       ├── rl_policy.json          # Trained PPO policy (base)
│       └── rl_policy_runtime.json  # Runtime-updated policy (auto-generated)
├── models/
│   └── financial_rag/
│       └── documents.json          # 975 SEC filing chunks + metadata
├── build_rag.py                    # Build and save RAG document index
├── benchmark_detectors.py          # Evaluate all detectors on test dataset
├── train_rl_selector.py            # Offline PPO training script
├── sec_downloader_v2.py            # SEC EDGAR API downloader
└── requirements.txt
```

---

## Hallucination Detectors

All detectors implement `BaseDetector` and return a `DetectionResult`:

```python
@dataclass
class DetectionResult:
    is_hallucinated: bool
    confidence: float          # 0.0–1.0
    hallucination_score: float
    method_name: str
    latency_ms: float
    explanation: Optional[str]
    details: Optional[Dict]
```

### Detector Comparison

| Detector | Cost | Typical Latency | Best For |
|---|---|---|---|
| Token Overlap | $0.05 | ~3–5ms | Number-heavy factual questions |
| Semantic Similarity | $0.10 | ~40ms | Paraphrased or rephrased facts |
| BERT NLI | $0.30 | ~50ms | Comparisons, trends, reasoning |
| LLM Judge | $1.00 | ~500ms–18s | Thorough review of complex claims |

### Token Overlap
Weighted n-gram precision (BLEU-style, 1–4 grams) plus number matching. Fast and reliable for questions with specific figures.

### Semantic Similarity
Cosine similarity between response and context embeddings using `all-MiniLM-L6-v2`. Aggregated across chunks via max/mean/weighted strategies.

### BERT NLI
Zero-shot classification via `facebook/bart-large-mnli`. Labels: entailment (grounded) / contradiction (hallucinated) / neutral.

### LLM Judge
Prompts a large language model to fact-check the response against source documents. Supported backends: `ollama` (Qwen2.5:7b), `openai`, `anthropic`, `mock`.

### Using Detectors Directly

```python
from src.detection import create_detector

detector = create_detector("token_overlap")
detector = create_detector("semantic_similarity")
detector = create_detector("bert_nli")
detector = create_detector("llm_judge", backend="ollama", model="qwen2.5:7b")

result = detector.detect(question, answer, context_chunks)
print(result.is_hallucinated, result.confidence, result.explanation)
```

---

## RL Policy

### State Features (12-dimensional)
`question_len`, `entity_count`, `numeric_density`, `has_percentage`, `has_comparison`, `complexity_score`, `retrieval_confidence`, `top_chunk_score`, `score_gap`, `answer_len`, `answer_numeric_density`, `answer_entity_count`

### PPO Architecture
- **Actor:** 2-layer MLP (12 → 32 → 32 → 4 logits) → softmax over detectors
- **Critic:** same architecture → scalar state value
- **Exploration:** 15% ε-greedy to prevent mode collapse
- **Update:** on-policy PPO with clipping (ε=0.2), entropy bonus, gradient clipping

### Reward Function
```
reward = 0.8 × quality_signal − 0.5 × detector_cost − 0.1 × latency_penalty
```
Where `quality_signal = correct × confidence` (1 if grounded, 0 if hallucinated).

### Policy Files
- `data/processed/rl_policy.json` — base trained policy
- `data/processed/rl_policy_runtime.json` — runtime-updated policy (auto-created after first query, takes precedence)

### Training
```bash
# Offline training
python train_rl_selector.py --algorithm ppo

# The dashboard updates the policy live on every query
```

---

## Answer Synthesis

The RAG answer generator (`verify_query.py: select_answer_sentences`) extracts up to 2 sentences from retrieved chunks using a scoring formula:

```
score = rank_priority + token_overlap + 0.45 × focus_hits + number_bonus + company_bonus
```

**Key properties:**
- Sentences with zero token overlap and zero focus term hits are filtered out (relevance gate)
- The second sentence is constrained to the same chunk as the first to ensure coherence
- Rank priority does not include the chunk's raw retrieval score, so sentence-level relevance signals are not swamped

**Retrieval** uses hash-based FAISS embeddings (no ML model required) combined with lexical overlap, focus term matching, number matching, and company filtering.

---

## Benchmarking

```bash
python benchmark_detectors.py
python benchmark_detectors.py --detector token_overlap
python benchmark_detectors.py --verbose
```

Results are saved to `data/processed/benchmark_results.json` and used by the PPO policy as prior cost/accuracy estimates when initializing selection probabilities.

---

## Dashboard

The Next.js dashboard is the primary interface. It runs the full pipeline end-to-end and shows:

- **Answer** — generated response with grounded/hallucinated verdict
- **Policy Decision** — which detector was selected, why, and the full probability distribution over all 4 detectors
- **Pipeline trace** — query features, retrieval stats, detector execution details, reward, and policy update status

```bash
cd dashboard
npm install
npm run dev
# Opens http://localhost:3000
```

The dashboard calls `/api/verify` which spawns `dashboard/scripts/verify_query.py` as a subprocess (60s timeout).

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- Node.js 18+ (for dashboard)
- Ollama with `qwen2.5:7b` pulled (for live LLM Judge; mock mode works without it)

### Python Setup

```bash
git clone https://github.com/esp1745/Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG.git
cd Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Download SEC filings (if not already present)
python sec_downloader_v2.py

# 2. Build RAG document index
python build_rag.py

# 3. Run detector benchmarks (generates benchmark_results.json)
python benchmark_detectors.py

# 4. Train RL policy offline
python train_rl_selector.py --algorithm ppo

# 5. Launch dashboard
cd dashboard && npm run dev
```

### Environment Variables

```bash
# Required for live LLM Judge via OpenAI or Anthropic (mock mode needs neither)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Key Files Reference

| File | Purpose |
|---|---|
| `dashboard/scripts/verify_query.py` | Full pipeline: query → RAG → PPO → detector → PPO update |
| `dashboard/src/app/page.tsx` | Next.js dashboard UI |
| `dashboard/src/app/api/verify/route.ts` | API route, spawns verify_query.py |
| `dashboard/src/lib/rlSelection.ts` | TypeScript bandit/PPO for browser-side display |
| `src/rl/ppo_selector.py` | PPO actor-critic implementation |
| `src/rl/adaptive_selector.py` | Epsilon-greedy bandit + feature extraction |
| `src/detection/token_overlap.py` | Lexical overlap detector |
| `src/detection/semantic_similarity.py` | Embedding similarity detector |
| `src/detection/bert_nli.py` | NLI-based detector |
| `src/detection/llm_judge.py` | LLM-as-judge detector |
| `train_rl_selector.py` | Offline PPO training |
| `benchmark_detectors.py` | Detector evaluation framework |
| `build_rag.py` | RAG index builder |
| `sec_downloader_v2.py` | SEC EDGAR API downloader |
| `data/processed/rl_policy.json` | Trained PPO policy weights |
| `models/financial_rag/documents.json` | 975 SEC filing chunks |

---

*Yeshiva University Independent Study Project*  
*GitHub: [esp1745/Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG](https://github.com/esp1745/Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG)*
