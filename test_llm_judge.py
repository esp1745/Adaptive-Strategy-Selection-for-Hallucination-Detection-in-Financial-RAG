"""Test LLM Judge on both datasets"""
import time
import json
from build_rag import FinancialRAG
from src.detection import create_detector

def calculate_metrics(predictions, labels):
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    
    accuracy = (tp + tn) / len(predictions) if predictions else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

def main():
    # Load RAG
    print("Loading RAG system...")
    rag = FinancialRAG()
    rag.load('models/financial_rag')
    
    # Create LLM Judge detector
    detector = create_detector('llm_judge')
    print(f"Using LLM Judge: {detector.backend}/{detector.model}")
    print()
    
    # ============================================================
    # Test on CUSTOM DATASET
    # ============================================================
    print("=" * 60)
    print("CUSTOM DATASET (5 examples = 10 test cases)")
    print("=" * 60)
    
    with open('data/processed/hallucination_test_dataset.json') as f:
        custom_data = json.load(f)
    
    predictions = []
    labels = []
    latencies = []
    
    for i, example in enumerate(custom_data['examples'][:5]):
        q = example['question']
        results = rag.retrieve(q, k=3)
        context = [r['text'] for r in results[:3]]
        
        for resp_type in ['grounded', 'hallucinated']:
            resp = example[f'{resp_type}_response']
            true_label = (resp_type == 'hallucinated')
            
            start = time.time()
            result = detector.detect(q, resp, context)
            latency = (time.time() - start) * 1000
            
            predictions.append(result.is_hallucinated)
            labels.append(true_label)
            latencies.append(latency)
            
            status = 'OK' if result.is_hallucinated == true_label else 'X'
            print(f"  [{status}] Q{i+1} ({resp_type}): pred={result.is_hallucinated}, "
                  f"conf={result.confidence:.2f}, latency={latency/1000:.1f}s")
    
    acc, prec, rec, f1 = calculate_metrics(predictions, labels)
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n  CUSTOM Results: Accuracy={acc:.1%}, F1={f1:.1%}, Avg Latency={avg_latency/1000:.1f}s")
    
    custom_results = {'accuracy': acc, 'f1': f1, 'avg_latency_s': avg_latency/1000}
    
    # ============================================================
    # Test on PHANTOM DATASET  
    # ============================================================
    print("\n" + "=" * 60)
    print("PHANTOM DATASET (10 examples)")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        phantom = load_dataset('seyled/Phantom_Hallucination_Detection',
                               data_files='PhantomDataset/Phantom_10k_seed.csv')
        df = phantom['train'].to_pandas().head(10)
        
        predictions = []
        labels = []
        latencies = []
        
        for idx, row in df.iterrows():
            q = row['query']
            resp = row['answer']
            context = [row['context']]
            true_label = row['ground_truth_label'] == 'hallucination'
            
            start = time.time()
            result = detector.detect(q, resp, context)
            latency = (time.time() - start) * 1000
            
            predictions.append(result.is_hallucinated)
            labels.append(true_label)
            latencies.append(latency)
            
            status = 'OK' if result.is_hallucinated == true_label else 'X'
            label_str = 'hallucinated' if true_label else 'grounded'
            print(f"  [{status}] phantom_{idx} ({label_str}): pred={result.is_hallucinated}, "
                  f"conf={result.confidence:.2f}, latency={latency/1000:.1f}s")
        
        acc, prec, rec, f1 = calculate_metrics(predictions, labels)
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n  PHANTOM Results: Accuracy={acc:.1%}, F1={f1:.1%}, Avg Latency={avg_latency/1000:.1f}s")
        
        phantom_results = {'accuracy': acc, 'f1': f1, 'avg_latency_s': avg_latency/1000}
        
    except Exception as e:
        print(f"  Error loading PHANTOM: {e}")
        phantom_results = None
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("LLM JUDGE (Qwen2.5-7B) SUMMARY")
    print("=" * 60)
    print(f"  Custom Dataset:  Accuracy={custom_results['accuracy']:.1%}, F1={custom_results['f1']:.1%}")
    if phantom_results:
        print(f"  PHANTOM Dataset: Accuracy={phantom_results['accuracy']:.1%}, F1={phantom_results['f1']:.1%}")

if __name__ == "__main__":
    main()
