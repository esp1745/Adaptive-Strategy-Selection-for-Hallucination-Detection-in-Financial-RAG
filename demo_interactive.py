"""
Interactive Demo for Professor Presentation
Run: python demo_interactive.py
"""

import sys
import json
from pathlib import Path

# Suppress warnings for cleaner demo
import warnings
warnings.filterwarnings('ignore')

def load_components():
    """Load RAG and detectors"""
    print("Loading Financial RAG system...")
    from build_rag import FinancialRAG
    rag = FinancialRAG()
    rag.load('models/financial_rag')
    print(f"  Loaded {len(rag.documents)} document chunks from SEC 10-K filings")
    
    print("\nLoading hallucination detectors...")
    from src.detection import create_detector
    detectors = {
        'token_overlap': create_detector('token_overlap'),
        'semantic_similarity': create_detector('semantic_similarity'),
        'bert_nli': create_detector('bert_nli'),
        'llm_judge': create_detector('llm_judge', backend='mock')
    }
    print("  Loaded: Token Overlap, Semantic Similarity, BERT NLI, LLM Judge")
    
    return rag, detectors


def demo_rag_retrieval(rag):
    """Demo 1: Show RAG retrieval"""
    print("\n" + "="*70)
    print("DEMO 1: RAG RETRIEVAL FROM SEC 10-K FILINGS")
    print("="*70)
    
    queries = [
        "What was Apple's iPhone revenue in 2025?",
        "How many vehicles did Tesla deliver?",
        "What is Microsoft's cloud revenue?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        results = rag.retrieve(q, k=1)
        if results:
            r = results[0]
            print(f"  Company: {r['company']}")
            print(f"  Source: {r['filing_type']} ({r['filing_date']})")
            print(f"  Relevance: {r['score']:.3f}")
            text = r['text'][:150].replace('\n', ' ')
            print(f"  Context: \"{text}...\"")
    
    input("\n[Press Enter to continue...]")


def demo_hallucination_detection(rag, detectors):
    """Demo 2: Show hallucination detection"""
    print("\n" + "="*70)
    print("DEMO 2: HALLUCINATION DETECTION")
    print("="*70)
    
    # Example case
    question = "What was Apple's total revenue in fiscal year 2025?"
    grounded = "Apple's total net sales for fiscal year 2025 were $416.2 billion."
    hallucinated = "Apple's total revenue for fiscal year 2025 was $485.3 billion."
    
    print(f"\nQuestion: {question}")
    
    # Get context
    results = rag.retrieve(question, k=3)
    context = [r['text'] for r in results]
    
    print("\nRetrieved Context (from SEC 10-K):")
    for i, r in enumerate(results[:2], 1):
        text = r['text'][:100].replace('\n', ' ')
        print(f"  {i}. \"{text}...\"")
    
    print(f"\n[GROUNDED RESPONSE]")
    print(f"  \"{grounded}\"")
    
    print(f"\n[HALLUCINATED RESPONSE]")
    print(f"  \"{hallucinated}\"")
    print(f"  ^ Contains fabricated number ($485.3B vs actual $416.2B)")
    
    print("\n" + "-"*70)
    print("Running all 4 detection methods...")
    print("-"*70)
    
    print(f"\n{'Method':<25} {'Grounded':<15} {'Hallucinated':<15} {'Latency':<10}")
    print("-"*65)
    
    for name, detector in detectors.items():
        r1 = detector.detect(question, grounded, context)
        r2 = detector.detect(question, hallucinated, context)
        
        g_result = "Halluc!" if r1.is_hallucinated else "OK"
        h_result = "Halluc!" if r2.is_hallucinated else "OK"
        
        g_correct = "[correct]" if not r1.is_hallucinated else "[WRONG]"
        h_correct = "[correct]" if r2.is_hallucinated else "[WRONG]"
        
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<25} {g_result:<7}{g_correct:<8} {h_result:<7}{h_correct:<8} {r1.latency_ms:>6.1f}ms")
    
    input("\n[Press Enter to continue...]")


def demo_cost_tradeoff():
    """Demo 3: Show cost-accuracy tradeoff"""
    print("\n" + "="*70)
    print("DEMO 3: COST-ACCURACY TRADEOFF (The Research Problem)")
    print("="*70)
    
    print("""
The core insight: Different methods have different cost/accuracy tradeoffs.

METHOD COMPARISON:
+------------------------+----------+----------+----------+----------+
| Method                 | Accuracy | F1 Score | Latency  | Cost     |
+------------------------+----------+----------+----------+----------+
| Token Overlap          |   80.0%  |   81.2%  |    3ms   |   0.05   |
| Semantic Similarity    |   51.7%  |   32.6%  |   40ms   |   0.10   |
| BERT NLI               |   ~75%   |   ~70%   |   50ms   |   0.30   |
| LLM-as-Judge (GPT-4)   |   ~90%   |   ~88%   |  500ms   |   1.00   |
+------------------------+----------+----------+----------+----------+

KEY OBSERVATION:
- Token Overlap: Fast & cheap, but fails on semantic hallucinations
- LLM Judge: Accurate but expensive (20x cost, 100x latency)

RESEARCH QUESTION:
Can an RL agent learn WHEN to use expensive vs cheap methods,
achieving high accuracy at lower average cost?

HYPOTHESIS:
- Use cheap methods for "easy" cases (numerical fabrication)
- Reserve expensive methods for "hard" cases (subtle hallucinations)
""")
    
    input("[Press Enter to continue...]")


def demo_rl_concept():
    """Demo 4: Explain the RL approach"""
    print("\n" + "="*70)
    print("DEMO 4: REINFORCEMENT LEARNING APPROACH")
    print("="*70)
    
    print("""
SYSTEM ARCHITECTURE:

    User Query
        |
        v
    Financial RAG (SEC 10-K Filings)
        |
        v
    Query + Context + Response
        |
        v
    +-------------------+
    | Feature Extractor |  <-- 12-dimensional state vector
    +-------------------+
        |
        v
    +-------------------+
    |    PPO Agent      |  <-- Learns optimal policy
    +-------------------+
        |
        +------+------+------+------+
        |      |      |      |      |
        v      v      v      v      v
    Token   Semantic  BERT   LLM    Ensemble
    Overlap Similarity NLI   Judge  (multiple)

STATE FEATURES (12 dimensions):
- Query length, complexity, numerical content
- Response length, confidence indicators
- Context relevance scores
- Domain signals (revenue, operations, etc.)

REWARD FUNCTION:
    R = accuracy_reward - lambda * cost_penalty
    
Where lambda controls the cost-accuracy tradeoff.

TRAINING:
- PPO (Proximal Policy Optimization)
- 30 training examples x multiple episodes
- Learn to select optimal detector per query type
""")
    
    input("[Press Enter to continue...]")


def demo_interactive_query(rag, detectors):
    """Demo 5: Interactive query mode"""
    print("\n" + "="*70)
    print("DEMO 5: INTERACTIVE MODE")
    print("="*70)
    
    print("\nTry asking financial questions! (type 'quit' to exit)")
    print("Example queries:")
    print("  - What is Apple's services revenue?")
    print("  - How much did Tesla earn from automotive sales?")
    print("  - What is Google's advertising revenue?")
    
    while True:
        print()
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Retrieve context
        results = rag.retrieve(query, k=3)
        if not results:
            print("No relevant context found.")
            continue
        
        context = [r['text'] for r in results]
        
        print(f"\nRetrieved from {results[0]['company']} 10-K:")
        text = results[0]['text'][:200].replace('\n', ' ')
        print(f"  \"{text}...\"")
        
        # Create a sample response (in real system, this would be LLM-generated)
        print("\n[In a full system, an LLM would generate a response here]")
        print("[The RL agent would then select the optimal detector]")


def main():
    print("\n" + "="*70)
    print("  COST-ADAPTIVE HALLUCINATION DETECTION IN FINANCIAL RAG")
    print("  Independent Study Project - Spring 2026")
    print("="*70)
    
    # Load components
    rag, detectors = load_components()
    
    print("\n" + "="*70)
    print("DEMO MENU")
    print("="*70)
    print("""
1. RAG Retrieval Demo (show document retrieval)
2. Hallucination Detection Demo (compare methods)
3. Cost-Accuracy Tradeoff (the research problem)
4. RL Approach Explanation (system architecture)
5. Interactive Query Mode
6. Run All Demos
0. Exit
""")
    
    while True:
        choice = input("\nSelect demo [1-6, 0 to exit]: ").strip()
        
        if choice == '0':
            print("\nThank you!")
            break
        elif choice == '1':
            demo_rag_retrieval(rag)
        elif choice == '2':
            demo_hallucination_detection(rag, detectors)
        elif choice == '3':
            demo_cost_tradeoff()
        elif choice == '4':
            demo_rl_concept()
        elif choice == '5':
            demo_interactive_query(rag, detectors)
        elif choice == '6':
            demo_rag_retrieval(rag)
            demo_hallucination_detection(rag, detectors)
            demo_cost_tradeoff()
            demo_rl_concept()
            print("\n[All demos complete!]")
        else:
            print("Invalid choice. Enter 1-6 or 0.")


if __name__ == "__main__":
    main()
