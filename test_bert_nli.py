"""Quick test of BERT NLI detector"""

from src.detection import create_detector
from build_rag import FinancialRAG

print('Testing BERT NLI detector...')
print('This will download the model on first run (~500MB)')

rag = FinancialRAG()
rag.load('models/financial_rag')

question = 'What was Apple total revenue in 2025?'
context = [r['text'] for r in rag.retrieve(question, k=2)]

grounded = 'Apple total net sales for fiscal year 2025 were 416.2 billion dollars.'
hallucinated = 'Apple total revenue for fiscal year 2025 was 485.3 billion dollars.'

try:
    detector = create_detector('bert_nli')
    
    result = detector.detect(question, grounded, context)
    print(f'Grounded: hallucinated={result.is_hallucinated}, latency={result.latency_ms:.0f}ms')
    
    result = detector.detect(question, hallucinated, context)
    print(f'Hallucinated: hallucinated={result.is_hallucinated}, latency={result.latency_ms:.0f}ms')
    
    print('BERT NLI detector working!')
except Exception as e:
    print(f'BERT NLI error: {e}')
    import traceback
    traceback.print_exc()
