# Project Documentation: Cost-Adaptive Hallucination Detection in Financial RAG Systems

## Overview

This project implements a system for detecting hallucinations in financial RAG (Retrieval-Augmented Generation) responses. The goal is to develop a reinforcement learning agent that adaptively selects the optimal hallucination detection strategy based on query characteristics, balancing accuracy against computational cost.

**Research Question:** Can a reinforcement learning agent learn when to use expensive versus cheap hallucination detection methods, achieving higher accuracy at lower average computational cost than any static strategy?

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Data Collection](#data-collection)
3. [RAG System Implementation](#rag-system-implementation)
4. [Hallucination Detection Methods](#hallucination-detection-methods)
5. [Test Dataset Creation](#test-dataset-creation)
6. [Benchmarking](#benchmarking)
7. [Demo Applications](#demo-applications)
8. [Installation & Usage](#installation--usage)
9. [Key Files Reference](#key-files-reference)

---

## 1. Project Structure

```
financial-rag-hallucination/
├── app.py                      # Streamlit web demo
├── benchmark_detectors.py      # Evaluation framework
├── build_rag.py               # RAG system implementation
├── create_test_dataset.py     # Test dataset generator
├── demo_interactive.py        # CLI demo for presentations
├── demo_rag.py                # RAG demonstration script
├── evaluate_retrieval.py      # Retrieval accuracy testing
├── presentation_demo.py       # Formatted presentation demos
├── sec_downloader.py          # Original SEC downloader (deprecated)
├── sec_downloader_v2.py       # Fixed SEC EDGAR API downloader
├── data/
│   ├── raw/sec_filings/       # Downloaded SEC 10-K filings
│   └── processed/             # Processed test datasets
├── models/
│   └── financial_rag/         # Saved RAG index
│       ├── documents.json     # Document chunks
│       └── embeddings.npy     # FAISS embeddings
├── src/
│   ├── detection/             # Hallucination detectors
│   │   ├── __init__.py        # Factory and registry
│   │   ├── base.py            # Abstract base class
│   │   ├── token_overlap.py   # Lexical overlap detector
│   │   ├── semantic_similarity.py  # Embedding similarity
│   │   ├── bert_nli.py        # NLI-based detector
│   │   └── llm_judge.py       # LLM-as-judge detector
│   ├── rl/                    # RL environment (Week 2)
│   └── utils/
│       └── logger.py          # Logging utilities
└── tests/                     # Unit tests
```

---

## 2. Data Collection

### Problem Encountered
The original `sec_downloader.py` was downloading XBRL viewer pages instead of actual SEC filing content. The downloaded files contained minimal data (9-15KB) with SEC navigation HTML rather than the actual 10-K content.

### Solution: sec_downloader_v2.py

**File:** [sec_downloader_v2.py](sec_downloader_v2.py)

This fixed version properly downloads SEC 10-K filings using the SEC EDGAR API.

#### How It Works:

1. **Fetch Filing Index** - Uses SEC EDGAR submissions API to get list of filings:
   ```python
   url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
   ```

2. **Download Filing Content** - Tries multiple methods:
   - **Method 1:** Primary document (usually HTML file)
   - **Method 2:** Full submission `.txt` file with `<TYPE>10-K` extraction

3. **Clean HTML Content** - Parses HTML and extracts text:
   ```python
   soup = BeautifulSoup(content, 'html.parser')
   for tag in soup(['script', 'style', 'meta', 'link']):
       tag.decompose()
   content = soup.get_text(separator=' ')
   ```

4. **Text Cleanup** - Removes excess whitespace and normalizes text:
   ```python
   text = re.sub(r'\s+', ' ', text)
   text = re.sub(r'\n\s*\n', '\n\n', text)
   ```

#### Companies Downloaded:
| Company | CIK | Filings |
|---------|-----|---------|
| Apple | 0000320193 | 10-K 2025, 2024 |
| Microsoft | 0000789019 | 10-K 2025, 2024 |
| Tesla | 0001318605 | 10-K 2025, 2024 |
| Amazon | 0001018724 | 10-K 2025, 2024 |
| Google/Alphabet | 0001652044 | 10-K 2025, 2024 |

#### Output:
Files saved to `data/raw/sec_filings/{company}_filings.json` with structure:
```json
{
  "company_name": "Apple",
  "cik": "0000320193",
  "filings": [
    {
      "type": "10-K",
      "date": "2025-11-01",
      "content": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION..."
    }
  ]
}
```

#### Usage:
```bash
python3 sec_downloader_v2.py
```

---

## 3. RAG System Implementation

### File: build_rag.py

**Class:** `FinancialRAG`

The RAG system uses sentence-transformers for embeddings and FAISS for vector search.

#### Components:

1. **Embedding Model:** `all-MiniLM-L6-v2` (384-dimensional vectors)
   ```python
   self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
   ```

2. **Vector Index:** FAISS IndexFlatL2 (exact nearest neighbor search)
   ```python
   self.index = faiss.IndexFlatL2(dimension)  # dimension=384
   ```

3. **Document Chunking:** 300 words per chunk with 50-word overlap
   ```python
   def chunk_document(self, text, chunk_size=300, overlap=50):
       words = text.split()
       chunks = []
       start = 0
       while start < len(words):
           end = start + chunk_size
           chunk = ' '.join(words[start:end])
           if len(chunk.strip()) > 100:
               chunks.append(chunk)
           start = end - overlap
       return chunks
   ```

#### Key Methods:

| Method | Description |
|--------|-------------|
| `load_sec_filings(dir)` | Load and index SEC filings from JSON |
| `chunk_document(text)` | Split document into overlapping chunks |
| `build_index()` | Create FAISS index from all chunks |
| `save(path)` | Save index to disk |
| `load(path)` | Load index from disk |
| `retrieve(query, k)` | Retrieve top-k relevant chunks |

#### Statistics:
- **Total Documents:** 975 chunks indexed
- **Companies:** 5 (Apple, Microsoft, Tesla, Amazon, Google)
- **Filings per Company:** 2 (10-K 2025, 2024)
- **Retrieval Accuracy:** 100% (tested on 40 queries)

#### Usage:
```python
from build_rag import FinancialRAG

# Build and save
rag = FinancialRAG()
rag.load_sec_filings('data/raw/sec_filings')
rag.build_index()
rag.save('models/financial_rag')

# Load and query
rag = FinancialRAG()
rag.load('models/financial_rag')
results = rag.retrieve("What was Apple's revenue?", k=3)
```

---

## 4. Hallucination Detection Methods

Four detection methods were implemented with varying cost/accuracy tradeoffs.

### Base Interface

**File:** [src/detection/base.py](src/detection/base.py)

All detectors implement the `BaseDetector` interface:

```python
@dataclass
class DetectionResult:
    is_hallucinated: bool          # Binary classification
    confidence: float              # 0.0 to 1.0
    hallucination_score: float     # Raw score
    method_name: str
    latency_ms: float
    cost_estimate: float
    explanation: Optional[str]
    details: Optional[Dict]

class BaseDetector(ABC):
    name: str
    cost: float
    typical_latency_ms: float
    
    @abstractmethod
    def detect(self, question, response, context) -> DetectionResult:
        pass
```

---

### Method 1: Token Overlap Detector

**File:** [src/detection/token_overlap.py](src/detection/token_overlap.py)

**Cost:** 0.05 | **Latency:** ~3-5ms

Uses lexical overlap metrics (BLEU and ROUGE-like) to detect hallucinations. Low overlap between response and context indicates potential hallucination.

#### How It Works:

1. **Tokenization:** Lowercase and split on non-alphanumeric:
   ```python
   tokens = re.findall(r'\b\w+\b', text.lower())
   ```

2. **N-gram Precision (BLEU-style):**
   ```python
   def _ngram_precision(self, response_tokens, context_tokens, n):
       response_ngrams = self._get_ngrams(response_tokens, n)
       context_ngrams = self._get_ngrams(context_tokens, n)
       # Count matches
       matches = sum(min(response_counts[ng], context_counts[ng]) 
                     for ng in response_ngrams)
       return matches / len(response_ngrams)
   ```

3. **Weighted Combined Score:**
   ```python
   ngram_weights = (0.4, 0.3, 0.2, 0.1)  # 1-gram to 4-gram
   bleu = sum(w * precision_n for w, precision_n in zip(weights, precisions))
   ```

4. **Number Matching:** Extra weight for numerical accuracy:
   ```python
   def _extract_numbers(self, text):
       patterns = [
           r'\$[\d,]+\.?\d*\s*(?:billion|million|B|M)?',
           r'[\d,]+\.?\d*%',
           r'[\d,]+\.?\d*\s*(?:billion|million|B|M)'
       ]
       return numbers
   ```

#### Parameters:
- `threshold`: 0.3 (below = hallucination)
- `ngram_weights`: (0.4, 0.3, 0.2, 0.1)
- `extract_numbers`: True

---

### Method 2: Semantic Similarity Detector

**File:** [src/detection/semantic_similarity.py](src/detection/semantic_similarity.py)

**Cost:** 0.1 | **Latency:** ~40ms

Uses cosine similarity between response embeddings and context embeddings.

#### How It Works:

1. **Encode Response and Context:**
   ```python
   self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
   response_embedding = self.encoder.encode([response])
   context_embeddings = self.encoder.encode(context)
   ```

2. **Compute Cosine Similarity:**
   ```python
   def cosine_similarity(a, b):
       return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
   ```

3. **Aggregate Across Context Chunks:**
   - `max`: Take highest similarity (default)
   - `mean`: Average all similarities
   - `weighted`: Weight by context relevance

4. **Classification:**
   ```python
   if max_similarity < threshold:  # threshold=0.5
       is_hallucinated = True
   ```

---

### Method 3: BERT NLI Detector

**File:** [src/detection/bert_nli.py](src/detection/bert_nli.py)

**Cost:** 0.3 | **Latency:** ~50ms

Uses Natural Language Inference to classify relationships.

#### How It Works:

1. **Load NLI Model:**
   ```python
   from transformers import pipeline
   self.classifier = pipeline(
       "zero-shot-classification",
       model='facebook/bart-large-mnli',
       device=-1  # CPU
   )
   ```

2. **Classification Labels:**
   - **ENTAILMENT:** Response supported by context → NOT hallucinated
   - **CONTRADICTION:** Response contradicts context → Hallucinated
   - **NEUTRAL:** Uncertain

3. **Combine Context with Response:**
   ```python
   # Classify response against combined context
   result = self.classifier(
       response,
       candidate_labels=['entailment', 'contradiction', 'neutral'],
       hypothesis_template="This statement is {} by the provided context."
   )
   ```

4. **Alternative approach using entailment checking:**
   ```python
   # Check if context entails response
   labels = ['true', 'false']
   premise = f"Context: {context}"
   hypothesis = f"Based on the context: {response}"
   ```

---

### Method 4: LLM-as-Judge Detector

**File:** [src/detection/llm_judge.py](src/detection/llm_judge.py)

**Cost:** 1.0 | **Latency:** ~500ms

Uses a large language model to evaluate groundedness.

#### Backends Supported:
| Backend | Model | API Key Required |
|---------|-------|------------------|
| `openai` | gpt-4o-mini | OPENAI_API_KEY |
| `anthropic` | claude-3-sonnet | ANTHROPIC_API_KEY |
| `mock` | None | No |

#### System Prompt:
```
You are an expert fact-checker specialized in financial documents.
Your task is to determine if a response contains hallucinations.

A response is HALLUCINATED if it:
1. Contains numbers, percentages, or dates not found in sources
2. Makes claims about events not mentioned in sources
3. Misrepresents trends or comparisons
4. Attributes information to wrong company/time period

Respond with JSON:
{
    "is_hallucinated": true/false,
    "confidence": 0.0-1.0,
    "explanation": "...",
    "specific_issues": [...]
}
```

#### Mock Mode (for testing without API):
```python
class MockLLMJudge:
    def detect(self, question, response, context):
        # Heuristic-based detection
        has_numbers = bool(re.search(r'\$[\d,]+', response))
        numbers_in_context = self._check_numbers_in_context(response, context)
        
        if has_numbers and not numbers_in_context:
            return DetectionResult(is_hallucinated=True, confidence=0.7)
```

---

### Detection Factory

**File:** [src/detection/__init__.py](src/detection/__init__.py)

```python
from src.detection import create_detector, list_detectors

# Create detector by name
detector = create_detector('token_overlap')
detector = create_detector('semantic_similarity')
detector = create_detector('bert_nli')
detector = create_detector('llm_judge', backend='mock')

# List all available detectors
detectors = list_detectors()
# ['token_overlap', 'semantic_similarity', 'bert_nli', 'llm_judge']
```

---

## 5. Test Dataset

### Primary Dataset: PHANTOM (NeurIPS 2024)

**Paper:** "PHANTOM: A Benchmark for Hallucination Detection in Financial Long-Context QA"  
**Published:** NeurIPS 2024 (November 2024)  
**Link:** [https://openreview.net/forum?id=5YQAo0S3Hm](https://openreview.net/forum?id=5YQAo0S3Hm)

PHANTOM is specifically designed for hallucination detection in financial document QA - a perfect match for this project's domain.

#### Why PHANTOM?
| Feature | PHANTOM | General Datasets (TruthfulQA, HaluEval) |
|---------|---------|----------------------------------------|
| Domain | Financial documents | General knowledge |
| Context | Long-form SEC filings | Short passages |
| Hallucination types | Financial-specific | Generic |
| Numerical accuracy | Tested | Not emphasized |

#### Loading PHANTOM:
```python
from download_phantom import load_phantom_dataset, get_phantom_examples

# Load dataset
dataset = load_phantom_dataset()

# Get evaluation examples
examples = get_phantom_examples(dataset)
```

---

### Secondary Dataset: Custom SEC Filing Dataset

**File:** `create_test_dataset.py`  
**Output:** `data/processed/hallucination_test_dataset.json`

This custom dataset supplements PHANTOM with examples from our specific RAG system.

#### Dataset Structure:
```json
{
  "metadata": {
    "description": "Hallucination detection test dataset for Financial RAG",
    "companies": ["Apple", "Microsoft", "Tesla", "Amazon", "Google"],
    "source": "SEC 10-K filings (2025-2026)",
    "num_examples": 30,
    "label_scheme": {
      "grounded": "Response is factually consistent with source documents",
      "hallucinated": "Response contains fabricated or incorrect information"
    }
  },
  "examples": [
    {
      "id": 1,
      "question": "What was Apple's total revenue in fiscal year 2025?",
      "company": "Apple",
      "domain": "revenue",
      "grounded_response": "Apple's total net sales for fiscal year 2025 were $416.2 billion...",
      "hallucinated_response": "Apple's total revenue for fiscal year 2025 was $485.3 billion...",
      "hallucination_type": "numerical_fabrication",
      "difficulty": "easy"
    }
  ]
}
```

#### Hallucination Types:
| Type | Description | Example |
|------|-------------|---------|
| `numerical_fabrication` | Incorrect numbers | "$485B" instead of "$416B" |
| `entity_fabrication` | Invented products/services | "Apple AI+ subscription" |
| `temporal_error` | Wrong dates/periods | "Q3 2024" instead of "FY 2025" |
| `attribution_error` | Wrong company | "Microsoft's iPhone" |
| `relationship_distortion` | Wrong trends | "22% increase" instead of "4% decrease" |

#### Statistics:
- **Total Examples:** 30
- **Per Company:** 6 examples each
- **Difficulty Distribution:** Easy (12), Medium (12), Hard (6)
- **Balanced Labels:** 30 grounded + 30 hallucinated = 60 test cases

---

## 6. Benchmarking

### File: benchmark_detectors.py

Evaluates all detection methods on the test dataset.

#### Metrics Calculated:
- **Accuracy:** (TP + TN) / Total
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1 Score:** 2 * (P * R) / (P + R)
- **Average Latency**
- **Cost**

#### Benchmark Results (Sample):

| Detector | Accuracy | Precision | Recall | F1 | Latency | Cost |
|----------|----------|-----------|--------|-----|---------|------|
| Token Overlap | 75.2% | 80.0% | 82.5% | 81.2% | 3.2ms | 0.05 |
| Semantic Sim | 65.0% | 57.1% | 20.0% | 32.6% | 41.3ms | 0.10 |
| BERT NLI | 58.3% | 52.4% | 95.0% | 67.5% | 52.1ms | 0.30 |
| LLM Judge | 70.0% | 66.7% | 63.3% | 65.0% | 5.2ms* | 1.00 |

*Mock mode - real LLM would be ~500ms

#### Usage:
```bash
python3 benchmark_detectors.py
python3 benchmark_detectors.py --detector token_overlap  # Single detector
python3 benchmark_detectors.py --verbose  # Detailed output
```

---

## 7. Demo Applications

### CLI Demo: demo_interactive.py

Interactive command-line demo for presentations.

```bash
python3 demo_interactive.py
```

**Menu Options:**
1. RAG Retrieval Demo - Search SEC filings
2. Hallucination Detection Demo - Test responses
3. Cost vs Accuracy Tradeoffs - Compare methods
4. RL Approach Explanation - Project motivation
5. Interactive Mode - Custom queries
0. Exit

---

### Web Demo: app.py (Streamlit)

Interactive web-based demo.

```bash
streamlit run app.py
# Opens http://localhost:8501
```

**Tabs:**
1. **RAG Query** - Search SEC filings interactively
2. **Hallucination Detection** - Test response groundedness
3. **Method Comparison** - Run all 4 detectors side-by-side
4. **Dataset Info** - View test dataset statistics

#### Features:
- Cached model loading (`@st.cache_resource`)
- Real-time latency display
- Side-by-side method comparison
- Pre-loaded example queries

---

## 8. Installation & Usage

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/esp1745/Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG.git
cd Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
faiss-cpu>=1.7.0
numpy>=1.24.0
beautifulsoup4>=4.12.0
requests>=2.28.0
streamlit>=1.28.0
```

### Quick Start

```bash
# 1. Download SEC filings (if not already done)
python3 sec_downloader_v2.py

# 2. Build RAG index
python3 build_rag.py

# 3. Create test dataset
python3 create_test_dataset.py

# 4. Run benchmarks
python3 benchmark_detectors.py

# 5. Launch demo
streamlit run app.py
# OR
python3 demo_interactive.py
```

### Environment Variables (Optional)
```bash
# For real LLM-as-Judge (not required for mock mode)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 9. Key Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `sec_downloader_v2.py` | Download SEC filings | `download_all_filings()` |
| `build_rag.py` | RAG system | `FinancialRAG.retrieve()` |
| `create_test_dataset.py` | Generate test data | `create_test_dataset()` |
| `benchmark_detectors.py` | Evaluate detectors | `benchmark_detector()` |
| `src/detection/base.py` | Base interface | `BaseDetector`, `DetectionResult` |
| `src/detection/token_overlap.py` | Lexical overlap | `TokenOverlapDetector.detect()` |
| `src/detection/semantic_similarity.py` | Embedding similarity | `SemanticSimilarityDetector.detect()` |
| `src/detection/bert_nli.py` | NLI classification | `BERTNLIDetector.detect()` |
| `src/detection/llm_judge.py` | LLM evaluation | `LLMJudgeDetector.detect()` |
| `app.py` | Web demo | Streamlit app |
| `demo_interactive.py` | CLI demo | Menu-driven interface |

---

## Next Steps (Week 2)

1. **RL Environment** - Gymnasium environment for detector selection
2. **Feature Extraction** - 12-dimensional state vector from query characteristics
3. **PPO Training** - Train agent using Stable-Baselines3
4. **Evaluation** - Compare RL policy vs static strategies

---

## Repository

**GitHub:** [esp1745/Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG](https://github.com/esp1745/Adaptive-Strategy-Selection-for-Hallucination-Detection-in-Financial-RAG)

---

*Documentation generated: February 2026*
*Yeshiva University Independent Study Project*
