"""
BERT-based NLI Hallucination Detector

Uses a pre-trained Natural Language Inference model to classify
whether a response is entailed by, contradicts, or is neutral to the context.

Cost: MEDIUM (~50ms)
Best for: Mixed numerical and text, detecting logical contradictions
"""

from typing import List, Dict, Optional
import numpy as np

from .base import BaseDetector, DetectionResult


class BERTNLIDetector(BaseDetector):
    """
    Detects hallucinations using Natural Language Inference.
    
    Uses a pre-trained NLI model (BART-large-mnli or similar) to
    classify the relationship between context (premise) and 
    response (hypothesis).
    
    Classification:
    - ENTAILMENT: Response is supported by context → NOT hallucinated
    - CONTRADICTION: Response contradicts context → Hallucinated
    - NEUTRAL: Response is not supported but doesn't contradict → Uncertain
    """
    
    name = "bert_nli"
    cost = 0.3  # Medium cost
    typical_latency_ms = 50.0
    
    def __init__(
        self,
        model_name: str = 'facebook/bart-large-mnli',
        threshold: float = 0.5,
        max_context_length: int = 512,
        aggregation: str = 'max_contradiction'  # 'max_contradiction', 'mean', 'vote'
    ):
        """
        Initialize the BERT NLI detector.
        
        Args:
            model_name: HuggingFace model name for NLI
            threshold: Contradiction probability above this = hallucination
            max_context_length: Max tokens for context chunking
            aggregation: How to aggregate scores across context chunks
        """
        self.model_name = model_name
        self.threshold = threshold
        self.max_context_length = max_context_length
        self.aggregation = aggregation
        self.classifier = None  # Lazy loading
        self.tokenizer = None
    
    def _load_model(self):
        """Lazy load the NLI model"""
        if self.classifier is None:
            from transformers import pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1  # CPU, change to 0 for GPU
            )
    
    def _chunk_context(self, context: str, max_chars: int = 1000) -> List[str]:
        """Split context into chunks for processing"""
        words = context.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [context]
    
    def _nli_classify(
        self,
        premise: str,
        hypothesis: str
    ) -> Dict[str, float]:
        """
        Classify relationship between premise and hypothesis.
        Returns probabilities for entailment, contradiction, neutral.
        """
        # Use zero-shot classification as NLI proxy
        # Labels represent relationship to the premise
        result = self.classifier(
            hypothesis,
            candidate_labels=['true', 'false', 'uncertain'],
            hypothesis_template="Based on the context: {}. This statement is {{}}.",
            multi_label=False
        )
        
        # Map to NLI labels
        label_map = {
            'true': 'entailment',
            'false': 'contradiction',
            'uncertain': 'neutral'
        }
        
        scores = {}
        for label, score in zip(result['labels'], result['scores']):
            nli_label = label_map.get(label, label)
            scores[nli_label] = score
        
        return scores
    
    def _direct_nli(
        self,
        premise: str,
        hypothesis: str
    ) -> Dict[str, float]:
        """
        Direct NLI classification using the model's native capability.
        More accurate than zero-shot proxy.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Load model directly for NLI
            if not hasattr(self, '_nli_model'):
                self._nli_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Tokenize premise + hypothesis pair
            inputs = self._nli_tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self._nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            # MNLI labels: contradiction, neutral, entailment (indices 0, 1, 2)
            return {
                'contradiction': float(probs[0]),
                'neutral': float(probs[1]),
                'entailment': float(probs[2])
            }
        except Exception as e:
            # Fallback to zero-shot if direct NLI fails
            return self._nli_classify(premise, hypothesis)
    
    def detect(
        self,
        question: str,
        response: str,
        context: List[str],
        threshold: Optional[float] = None,
        **kwargs
    ) -> DetectionResult:
        """
        Detect hallucination using NLI classification.
        """
        import time
        start = time.time()
        
        self._load_model()
        threshold = threshold or self.threshold
        
        if not context:
            return DetectionResult(
                is_hallucinated=True,
                confidence=0.5,
                hallucination_score=1.0,
                method_name=self.name,
                latency_ms=0,
                cost_estimate=self.cost,
                explanation="No context provided for verification"
            )
        
        # Process each context chunk
        all_nli_scores = []
        
        for ctx in context:
            # Chunk long contexts
            chunks = self._chunk_context(ctx)
            
            for chunk in chunks:
                try:
                    scores = self._direct_nli(chunk, response)
                    all_nli_scores.append(scores)
                except Exception:
                    # Skip problematic chunks
                    continue
        
        if not all_nli_scores:
            return DetectionResult(
                is_hallucinated=True,
                confidence=0.3,
                hallucination_score=0.7,
                method_name=self.name,
                latency_ms=(time.time() - start) * 1000,
                cost_estimate=self.cost,
                explanation="Could not process context for NLI"
            )
        
        # Aggregate scores across chunks
        if self.aggregation == 'max_contradiction':
            # Most damning evidence approach
            contradiction_scores = [s['contradiction'] for s in all_nli_scores]
            entailment_scores = [s['entailment'] for s in all_nli_scores]
            
            max_contradiction = max(contradiction_scores)
            max_entailment = max(entailment_scores)
            
            # If any chunk strongly contradicts, likely hallucination
            # If any chunk strongly entails, likely grounded
            if max_contradiction > max_entailment:
                hallucination_score = max_contradiction
            else:
                hallucination_score = 1.0 - max_entailment
                
        elif self.aggregation == 'mean':
            mean_contradiction = np.mean([s['contradiction'] for s in all_nli_scores])
            mean_entailment = np.mean([s['entailment'] for s in all_nli_scores])
            hallucination_score = mean_contradiction / (mean_contradiction + mean_entailment + 1e-8)
            
        elif self.aggregation == 'vote':
            # Majority voting on classification
            votes = {'contradiction': 0, 'neutral': 0, 'entailment': 0}
            for scores in all_nli_scores:
                winner = max(scores, key=scores.get)
                votes[winner] += 1
            
            total_votes = sum(votes.values())
            hallucination_score = votes['contradiction'] / total_votes
        else:
            # Default to max_contradiction
            hallucination_score = max(s['contradiction'] for s in all_nli_scores)
        
        # Classification
        is_hallucinated = hallucination_score > threshold
        
        # Confidence
        distance_from_threshold = abs(hallucination_score - threshold)
        confidence = min(0.5 + distance_from_threshold, 1.0)
        
        latency = (time.time() - start) * 1000
        
        # Determine explanation
        if is_hallucinated:
            explanation = f"NLI indicates contradiction (score: {hallucination_score:.3f})"
        else:
            explanation = f"NLI indicates entailment/support (score: {hallucination_score:.3f})"
        
        return DetectionResult(
            is_hallucinated=is_hallucinated,
            confidence=confidence,
            hallucination_score=hallucination_score,
            method_name=self.name,
            latency_ms=latency,
            cost_estimate=self.cost,
            explanation=explanation,
            details={
                'nli_scores': all_nli_scores,
                'num_chunks_processed': len(all_nli_scores),
                'aggregation_method': self.aggregation,
                'threshold': threshold
            }
        )
    
    def get_features(self) -> Dict:
        """Get detector features for RL state"""
        return {
            'name': self.name,
            'cost': self.cost,
            'typical_latency_ms': self.typical_latency_ms,
            'model_name': self.model_name,
            'threshold': self.threshold,
            'aggregation': self.aggregation
        }


def create_detector(**kwargs) -> BERTNLIDetector:
    """Factory function to create detector instance"""
    return BERTNLIDetector(**kwargs)
