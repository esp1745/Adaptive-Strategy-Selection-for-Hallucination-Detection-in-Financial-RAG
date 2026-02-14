# Create notebooks/02_test_rag.ipynb or test_rag.py

from build_rag import SimpleRAG

# Create sample financial Q&A pairs
financial_docs = [
    "Tesla's total revenue for Q3 2024 was $25.2 billion, up 8% year-over-year.",
    "The automotive segment generated $20.0 billion in revenue.",
    "Energy generation and storage revenue reached $2.4 billion.",
    "Tesla's operating income was $2.7 billion with a 10.8% operating margin.",
    "The company delivered 462,890 vehicles in Q3 2024.",
    "Free cash flow for the quarter was $2.7 billion.",
    "Total cash and investments stood at $33.6 billion at quarter end."
]

# Initialize RAG
print("=" * 60)
print("SIMPLE FINANCIAL RAG SYSTEM TEST")
print("=" * 60)

rag = SimpleRAG()
rag.add_documents(financial_docs)

# Test queries
queries = [
    "What was Tesla's revenue in Q3 2024?",
    "How many vehicles did Tesla deliver?",
    "What was the operating margin?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print("="*60)
    
    results = rag.retrieve(query, k=2)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. RETRIEVED (score={score:.3f}):")
        print(f"   {doc}")

print("\nRAG system working! You now have:")
print("  - Document retrieval")
print("  - Semantic search") 
print("  - Basic infrastructure")