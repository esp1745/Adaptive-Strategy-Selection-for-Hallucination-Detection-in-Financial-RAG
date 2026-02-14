"""
Evaluate retrieval accuracy
"""

from build_rag import FinancialRAG

rag = FinancialRAG()
rag.load('models/financial_rag')

# Questions with known correct companies
test_cases = [
    ("What was Apple's iPhone revenue?", "Apple"),
    ("What are Tesla's autopilot features?", "Tesla"),
    ("What is Microsoft's Azure business?", "Microsoft"),
    ("What is Amazon's AWS growth?", "Amazon"),
    ("What is Google's search advertising?", "Google"),
]

print("EVALUATING RETRIEVAL ACCURACY")
print("="*60)

correct = 0
total = len(test_cases)

for question, expected_company in test_cases:
    results = rag.retrieve(question, k=3)
    
    # Check if top result is correct company
    top_company = results[0]['company']
    is_correct = top_company == expected_company
    
    if is_correct:
        correct += 1
        status = "CORRECT"
    else:
        status = f"WRONG (got {top_company})"
    
    print(f"\n{status}: {question}")
    print(f"  Expected: {expected_company}")
    print(f"  Top result: {top_company} (score: {results[0]['score']:.3f})")

accuracy = correct / total * 100
print("\n" + "="*60)
print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
print("="*60)