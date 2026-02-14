"""
Benchmark all hallucination detection methods on the test dataset.

Measures:
- Accuracy, Precision, Recall, F1 score
- Average latency per method
- Cost comparison
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from build_rag import FinancialRAG
from src.detection import (
    create_detector,
    list_detectors,
    get_detector_info,
    DetectionResult
)


@dataclass
class BenchmarkMetrics:
    """Metrics for a single detector"""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    avg_latency_ms: float
    total_time_s: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    cost: float
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'total_time_s': round(self.total_time_s, 2),
            'cost': self.cost,
            'confusion_matrix': {
                'tp': self.true_positives,
                'fp': self.false_positives,
                'tn': self.true_negatives,
                'fn': self.false_negatives
            }
        }


def load_test_dataset(path: str = 'data/processed/hallucination_test_dataset.json') -> Dict:
    """Load the test dataset"""
    with open(path, 'r') as f:
        return json.load(f)


def calculate_metrics(
    predictions: List[bool],
    labels: List[bool]
) -> Tuple[float, float, float, float, int, int, int, int]:
    """Calculate classification metrics"""
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    
    accuracy = (tp + tn) / len(predictions) if predictions else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, tp, fp, tn, fn


def benchmark_detector(
    detector_name: str,
    dataset: Dict,
    rag: FinancialRAG,
    verbose: bool = True
) -> BenchmarkMetrics:
    """Benchmark a single detector on the dataset"""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {detector_name}")
        print('='*60)
    
    # Create detector
    try:
        detector = create_detector(detector_name)
    except Exception as e:
        print(f"ERROR creating detector: {e}")
        return None
    
    predictions = []
    labels = []
    latencies = []
    
    examples = dataset['examples']
    
    for i, example in enumerate(examples):
        question = example['question']
        company = example['company']
        
        # Get context from RAG
        results = rag.retrieve(question, k=3)
        context = [r['text'] for r in results if r['company'] == company]
        
        if not context:
            # Use any top results if company-specific context not found
            context = [r['text'] for r in results[:3]]
        
        # Test both grounded and hallucinated responses
        for response_type in ['grounded', 'hallucinated']:
            response = example[f'{response_type}_response']
            true_label = (response_type == 'hallucinated')
            
            try:
                start = time.time()
                result = detector.detect(question, response, context)
                latency = (time.time() - start) * 1000
                
                predictions.append(result.is_hallucinated)
                labels.append(true_label)
                latencies.append(latency)
                
                if verbose and i < 3:  # Show first few examples
                    status = "[OK]" if result.is_hallucinated == true_label else "[X]"
                    print(f"  {status} Q{example['id']} ({response_type}): "
                          f"pred={result.is_hallucinated}, "
                          f"conf={result.confidence:.2f}, "
                          f"latency={latency:.1f}ms")
                    
            except Exception as e:
                print(f"  ERROR on example {example['id']}: {e}")
                continue
    
    # Calculate metrics
    if not predictions:
        return None
    
    accuracy, precision, recall, f1, tp, fp, tn, fn = calculate_metrics(predictions, labels)
    avg_latency = sum(latencies) / len(latencies)
    total_time = sum(latencies) / 1000
    
    metrics = BenchmarkMetrics(
        name=detector_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        avg_latency_ms=avg_latency,
        total_time_s=total_time,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        cost=detector.cost
    )
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Accuracy:  {accuracy:.1%}")
        print(f"    Precision: {precision:.1%}")
        print(f"    Recall:    {recall:.1%}")
        print(f"    F1 Score:  {f1:.1%}")
        print(f"    Avg Latency: {avg_latency:.1f}ms")
        print(f"    Total Time: {total_time:.1f}s")
    
    return metrics


def run_full_benchmark(
    detectors: List[str] = None,
    verbose: bool = True
) -> Dict[str, BenchmarkMetrics]:
    """Run benchmark on all (or specified) detectors"""
    
    print("\n" + "="*70)
    print("HALLUCINATION DETECTION BENCHMARK")
    print("="*70)
    
    # Load RAG
    print("\nLoading RAG system...")
    rag = FinancialRAG()
    rag.load('models/financial_rag')
    print(f"  Loaded {len(rag.documents)} documents")
    
    # Load dataset
    print("\nLoading test dataset...")
    dataset = load_test_dataset()
    print(f"  Loaded {len(dataset['examples'])} examples")
    
    # Available detectors
    if detectors is None:
        detectors = list_detectors()
    
    print(f"\nDetectors to benchmark: {', '.join(detectors)}")
    
    # Get detector info
    print("\nDetector Information:")
    info = get_detector_info()
    for name in detectors:
        if name in info:
            d = info[name]
            print(f"  {name}: cost={d['cost']}, latency~{d['typical_latency_ms']}ms")
    
    # Run benchmarks
    results = {}
    for detector_name in detectors:
        try:
            metrics = benchmark_detector(detector_name, dataset, rag, verbose)
            if metrics:
                results[detector_name] = metrics
        except Exception as e:
            print(f"\nERROR benchmarking {detector_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\n{'Detector':<25} {'Accuracy':>10} {'F1':>10} {'Latency':>12} {'Cost':>8}")
    print("-" * 70)
    
    for name, m in sorted(results.items(), key=lambda x: x[1].f1, reverse=True):
        print(f"{name:<25} {m.accuracy:>10.1%} {m.f1:>10.1%} {m.avg_latency_ms:>10.1f}ms {m.cost:>8.2f}")
    
    # Save results
    output_path = Path('data/processed/benchmark_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(
            {name: m.to_dict() for name, m in results.items()},
            f,
            indent=2
        )
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def quick_test():
    """Quick test with a single example"""
    print("\n" + "="*60)
    print("QUICK TEST - Single Example")
    print("="*60)
    
    # Load RAG
    rag = FinancialRAG()
    rag.load('models/financial_rag')
    
    # Test example
    question = "What was Apple's total revenue in 2025?"
    grounded = "Apple's total net sales for fiscal year 2025 were $416.2 billion."
    hallucinated = "Apple's total revenue for fiscal year 2025 was $485.3 billion."
    
    # Get context
    results = rag.retrieve(question, k=3)
    context = [r['text'] for r in results]
    
    print(f"\nQuestion: {question}")
    print(f"\nContext (first 200 chars): {context[0][:200]}...")
    
    # Test each detector
    for detector_name in ['semantic_similarity', 'token_overlap']:
        print(f"\n--- {detector_name} ---")
        
        detector = create_detector(detector_name)
        
        # Test grounded response
        result = detector.detect(question, grounded, context)
        print(f"Grounded: hallucinated={result.is_hallucinated} "
              f"(confidence={result.confidence:.2f})")
        
        # Test hallucinated response
        result = detector.detect(question, hallucinated, context)
        print(f"Hallucinated: hallucinated={result.is_hallucinated} "
              f"(confidence={result.confidence:.2f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark hallucination detectors')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--detectors', nargs='+', help='Specific detectors to benchmark')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        # Run benchmark with fast detectors only by default (skip BERT which is slow to load)
        detectors = args.detectors or ['semantic_similarity', 'token_overlap', 'llm_judge']
        run_full_benchmark(detectors=detectors, verbose=not args.quiet)
