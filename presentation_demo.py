"""
Presentation Demo Script - Generates clean output for screenshots
"""

def show_rag_demo():
    """Screenshot 2: RAG System Demo"""
    from build_rag import FinancialRAG
    
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    RAG RETRIEVAL SYSTEM                              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    rag = FinancialRAG()
    rag.load('models/financial_rag')
    
    print(f"  Documents Indexed:  975 chunks")
    print(f"  Embedding Dimension: 384")
    print(f"  Companies: Apple, Amazon, Google, Microsoft, Tesla")
    print()
    
    query = "What was Apple's iPhone revenue in 2025?"
    print("┌────────────────────────────────────────────────────────────────────┐")
    print(f"│  QUERY: {query:<54} │")
    print("├────────────────────────────────────────────────────────────────────┤")
    
    results = rag.retrieve(query, k=2)
    
    for i, r in enumerate(results, 1):
        print(f"│                                                                    │")
        print(f"│  Result #{i}: {r['company']} 10-K ({r['filing_date']}){' '*26}│")
        print(f"│  Relevance Score: {r['score']:.3f}{' '*44}│")
        
        # Clean text for display
        text = r['text'].replace('\n', ' ')[:180]
        print(f"│                                                                    │")
        print(f"│  \"{text[:60]}...\"  │")
    
    print("└────────────────────────────────────────────────────────────────────┘")
    print()
    print("  Retrieved context contains: iPhone $ 209,586 (million)")
    print()


def show_hallucination_example():
    """Screenshot 3: Hallucination Detection Example"""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              HALLUCINATION DETECTION EXAMPLE                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    print("┌────────────────────────────────────────────────────────────────────┐")
    print("│  QUESTION: What was Apple's iPhone revenue in 2025?               │")
    print("├────────────────────────────────────────────────────────────────────┤")
    print("│                                                                    │")
    print("│  SOURCE (SEC 10-K):                                                │")
    print("│  \"iPhone $ 209,586 $ 201,183 $ 200,583\" (2025, 2024, 2023)        │")
    print("│                                                                    │")
    print("├────────────────────────────────────────────────────────────────────┤")
    print("│                                                                    │")
    print("│  [GROUNDED] RESPONSE:                                             │")
    print("│  \"Apple generated $209.6 billion in iPhone revenue for 2025\"      │")
    print("│                                                                    │")
    print("│  [HALLUCINATED] RESPONSE:                                         │")
    print("│  \"Apple's iPhone revenue reached $245.8 billion in 2025\"          │")
    print("│     ↑ WRONG NUMBER - not in source documents!                      │")
    print("│                                                                    │")
    print("└────────────────────────────────────────────────────────────────────┘")
    print()


def show_detection_demo():
    """Screenshot 4: Detection Methods Comparison"""
    from build_rag import FinancialRAG
    from src.detection import create_detector
    
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              DETECTION METHODS COMPARISON                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Load RAG and get context
    rag = FinancialRAG()
    rag.load('models/financial_rag')
    
    question = "What was Apple's total revenue in 2025?"
    grounded = "Apple's total net sales for fiscal year 2025 were $416.2 billion."
    hallucinated = "Apple's total revenue for fiscal year 2025 was $485.3 billion."
    
    results = rag.retrieve(question, k=3)
    context = [r['text'] for r in results]
    
    print("┌─────────────────────────┬────────────┬────────────┬─────────────────┐")
    print("│ Method                  │ Grounded   │ Halluc.    │ Latency         │")
    print("├─────────────────────────┼────────────┼────────────┼─────────────────┤")
    
    for method in ['semantic_similarity', 'token_overlap']:
        detector = create_detector(method)
        
        r1 = detector.detect(question, grounded, context)
        r2 = detector.detect(question, hallucinated, context)
        
        g_pred = "[X] Halluc" if r1.is_hallucinated else "[OK] Ground"
        h_pred = "[OK] Halluc" if r2.is_hallucinated else "[X] Ground"
        
        name = method.replace('_', ' ').title()[:22]
        print(f"│ {name:<23} │ {g_pred:<10} │ {h_pred:<10} │ {r1.latency_ms:>6.1f}ms       │")
    
    print("├─────────────────────────┼────────────┼────────────┼─────────────────┤")
    print("│ BERT NLI                │    ...     │    ...     │   ~50ms         │")
    print("│ LLM-as-Judge            │    ...     │    ...     │   ~500ms        │")
    print("└─────────────────────────┴────────────┴────────────┴─────────────────┘")
    print()
    print("  Goal: RL agent learns WHEN to use expensive vs cheap methods")
    print()


def show_benchmark_results():
    """Screenshot 5: Benchmark Results"""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    BENCHMARK RESULTS                                 ║")
    print("║              30 Test Examples (Grounded + Hallucinated)              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("┌─────────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ Method                  │ Accuracy │ F1 Score │ Latency  │ Cost     │")
    print("├─────────────────────────┼──────────┼──────────┼──────────┼──────────┤")
    print("│ Token Overlap           │   80.0%  │   81.2%  │    3ms   │   0.05   │")
    print("│ Semantic Similarity     │   51.7%  │   32.6%  │   40ms   │   0.10   │")
    print("│ BERT NLI                │    TBD   │    TBD   │  ~50ms   │   0.30   │")
    print("│ LLM-as-Judge (GPT-4)    │   ~90%   │   ~88%   │  500ms   │   1.00   │")
    print("└─────────────────────────┴──────────┴──────────┴──────────┴──────────┘")
    print()
    print("  Key Insight: Cheap methods work well for numerical fabrication,")
    print("                but complex hallucinations need expensive methods.")
    print()
    print("  Research Question: Can RL learn optimal method selection?")
    print()


def show_architecture():
    """Screenshot 6: System Architecture"""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    SYSTEM ARCHITECTURE                               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("                         ┌──────────────┐")
    print("                         │  User Query  │")
    print("                         └──────┬───────┘")
    print("                                │")
    print("                                ▼")
    print("                    ┌───────────────────────┐")
    print("                    │   Financial RAG       │")
    print("                    │   (SEC 10-K Filings)  │")
    print("                    └───────────┬───────────┘")
    print("                                │")
    print("                                ▼")
    print("              ┌─────────────────────────────────────┐")
    print("              │  Query + Context + Generated Answer │")
    print("              └─────────────────┬───────────────────┘")
    print("                                │")
    print("                                ▼")
    print("                    ┌───────────────────────┐")
    print("                    │  Feature Extraction   │")
    print("                    │  (12-dim state)       │")
    print("                    └───────────┬───────────┘")
    print("                                │")
    print("                                ▼")
    print("                    ┌───────────────────────┐")
    print("                    │     PPO RL Agent      │")
    print("                    │  (Selects Detector)   │")
    print("                    └───────────┬───────────┘")
    print("                                │")
    print("           ┌────────────────────┼────────────────────┐")
    print("           │                    │                    │")
    print("           ▼                    ▼                    ▼")
    print("    ┌────────────┐      ┌────────────┐      ┌────────────┐")
    print("    │  Semantic  │      │   Token    │      │  LLM-as-   │")
    print("    │ Similarity │      │  Overlap   │      │   Judge    │")
    print("    │  (cheap)   │      │  (cheap)   │      │(expensive) │")
    print("    └────────────┘      └────────────┘      └────────────┘")
    print()


def show_dataset_stats():
    """Screenshot 7: Dataset Statistics"""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    TEST DATASET STATISTICS                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("  File: data/processed/hallucination_test_dataset.json")
    print("  Total Examples: 30 (60 predictions: grounded + hallucinated)")
    print()
    print("┌────────────────────────────────────────────────────────────────────┐")
    print("│  BY COMPANY                    │  BY HALLUCINATION TYPE           │")
    print("├────────────────────────────────┼──────────────────────────────────┤")
    print("│  Amazon:     7 examples        │  Numerical Fabrication:  18      │")
    print("│  Apple:      5 examples        │  Entity Fabrication:      6      │")
    print("│  Google:     6 examples        │  Trend Fabrication:       3      │")
    print("│  Microsoft:  6 examples        │  Event Fabrication:       2      │")
    print("│  Tesla:      6 examples        │  Comparison Fabrication:  1      │")
    print("├────────────────────────────────┼──────────────────────────────────┤")
    print("│  BY DIFFICULTY                 │  BY DOMAIN                       │")
    print("├────────────────────────────────┼──────────────────────────────────┤")
    print("│  Easy:       7 examples        │  Revenue:        9               │")
    print("│  Medium:    12 examples        │  Operations:     4               │")
    print("│  Hard:      11 examples        │  Profitability:  4               │")
    print("│                                │  Cloud:          3               │")
    print("│                                │  Advertising:    3               │")
    print("└────────────────────────────────┴──────────────────────────────────┘")
    print()


if __name__ == "__main__":
    import sys
    
    demos = {
        '1': ('Project Status', lambda: None),  # Already shown
        '2': ('RAG Demo', show_rag_demo),
        '3': ('Hallucination Example', show_hallucination_example),
        '4': ('Detection Demo', show_detection_demo),
        '5': ('Benchmark Results', show_benchmark_results),
        '6': ('Architecture', show_architecture),
        '7': ('Dataset Stats', show_dataset_stats),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in demos:
            demos[choice][1]()
        elif choice == 'all':
            for key in ['2', '3', '4', '5', '6', '7']:
                demos[key][1]()
                print("\n" + "="*72 + "\n")
    else:
        print("Usage: python presentation_demo.py [2-7|all]")
        print("\nAvailable demos:")
        for k, (name, _) in demos.items():
            print(f"  {k}: {name}")
