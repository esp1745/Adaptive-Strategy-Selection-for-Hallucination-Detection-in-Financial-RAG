"""
RAG System Demo - Shows how the Financial RAG retrieval works
"""

from build_rag import FinancialRAG

def main():
    # Load RAG system
    print("="*70)
    print("FINANCIAL RAG SYSTEM DEMO")
    print("="*70)

    rag = FinancialRAG()
    rag.load('models/financial_rag')

    print(f"\nRAG System Stats:")
    print(f"  Total document chunks: {len(rag.documents)}")
    print(f"  Embedding dimension: {rag.embeddings.shape[1]}")
    companies = set(m['company'] for m in rag.metadata)
    print(f"  Companies indexed: {', '.join(sorted(companies))}")

    # Demo queries
    demo_queries = [
        "What was Apple's iPhone revenue in 2025?",
        "What are Tesla's risk factors?",
        "How much revenue did Microsoft Azure generate?",
        "What was Amazon's net income?",
        "What is Google's advertising business?",
    ]

    print("\n" + "="*70)
    print("RAG RETRIEVAL DEMO")
    print("="*70)

    for query in demo_queries:
        print(f"\n{'-'*70}")
        print(f"QUERY: {query}")
        print("-"*70)
        
        # Retrieve top 3 documents
        results = rag.retrieve(query, k=3)
        
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {r['company']} - {r['filing_type']} ({r['filing_date']})")
            print(f"      Score: {r['score']:.3f} (lower = more relevant)")
            # Show relevant snippet
            text = r['text'][:300].replace('\n', ' ')
            print(f"      Text: {text}...")

    print("\n" + "="*70)
    print("HOW THE RAG PIPELINE WORKS")
    print("="*70)
    print("""
1. DOCUMENT LOADING
   - SEC 10-K filings downloaded from EDGAR API
   - Each filing split into ~300 word chunks with 50 word overlap
   - 975 total chunks indexed across 5 companies

2. EMBEDDING CREATION
   - Each chunk encoded using sentence-transformers (all-MiniLM-L6-v2)
   - Creates 384-dimensional dense vectors
   - Captures semantic meaning of financial text

3. VECTOR INDEXING  
   - FAISS IndexFlatL2 for fast similarity search
   - All 975 embeddings stored in memory
   - Enables sub-millisecond retrieval

4. QUERY & RETRIEVAL
   - User query encoded to same 384-dim vector space
   - L2 distance search finds nearest document chunks
   - Returns top-k most similar chunks with company/filing metadata

5. CONTEXT FOR LLM
   - Retrieved chunks become the "context" for answer generation
   - LLM uses ONLY this context to generate responses
   - If answer isn't in context, LLM may hallucinate!
""")

    # Show a specific example with numbers
    print("="*70)
    print("EXAMPLE: Finding Apple's Revenue Numbers")
    print("="*70)
    
    query = "What was Apple's total revenue?"
    print(f"\nQuery: {query}\n")
    
    results = rag.retrieve(query, k=1)
    r = results[0]
    
    print(f"Retrieved from: {r['company']} {r['filing_type']} ({r['filing_date']})")
    print(f"\nFull context chunk:")
    print("-"*70)
    print(r['text'])
    print("-"*70)
    
    print("\nThis context contains the GROUND TRUTH data.")
    print("Any LLM response should match these numbers exactly.")
    print("If an LLM says different numbers, that's a HALLUCINATION.")


if __name__ == "__main__":
    main()
