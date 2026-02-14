"""
Semantic Similarity Hallucination Detector

Uses cosine similarity between response embeddings and context embeddings
to detect hallucinations. Low similarity indicates potential hallucination.

Cost: LOW (~10ms)
Best for: Qualitative questions, topic drift detection
"""

import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from .base import BaseDetector, DetectionResult


class SemanticSimilarityDetector(BaseDetector):
    """
    Detects hallucinations by measuring semantic similarity between
    the response and the retrieved context documents.
    
    Hypothesis: A hallucinated response will have lower semantic
    similarity to the source documents than a grounded response.
    """
    
    name = "semantic_similarity"
    cost = 0.1  # Low cost
    typical_latency_ms = 10.0
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        threshold: float = 0.5,
        aggregation: str = 'max'  # 'max', 'mean', 'weighted'
    ):
        """
        Initialize the semantic similarity detector.
        
        Args:
            model_name: Sentence transformer model to use
            threshold: Similarity below this = hallucination (0-1)
            aggregation: How to aggregate similarities across context docs
        """
        self.model_name = model_name
        self.threshold = threshold
        self.aggregation = aggregation
        self.encoder = None  # Lazy loading
    
    def _load_encoder(self):
        """Lazy load the encoder model"""
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_name)
    
    def detect(
        self,
        question: str,
        response: str,
        context: List[str],
        threshold: Optional[float] = None,
        **kwargs
    ) -> DetectionResult:
        """
        Detect hallucination using semantic similarity.
        
        Returns high hallucination score if response is semantically
        distant from all context documents.
        """
        import time
        start = time.time()
        
        self._load_encoder()
        threshold = threshold or self.threshold
        
        if not context:
            # No context = can't verify = assume hallucinated
            return DetectionResult(
                is_hallucinated=True,
                confidence=0.5,
                hallucination_score=1.0,
                method_name=self.name,
                latency_ms=0,
                cost_estimate=self.cost,
                explanation="No context provided for verification"
            )
        
        # Encode response and context
        response_embedding = self.encoder.encode([response], convert_to_numpy=True)[0]
        context_embeddings = self.encoder.encode(context, convert_to_numpy=True)
        
        # Calculate cosine similarities
        similarities = []
        for ctx_emb in context_embeddings:
            sim = np.dot(response_embedding, ctx_emb) / (
                np.linalg.norm(response_embedding) * np.linalg.norm(ctx_emb) + 1e-8
            )
            similarities.append(float(sim))
        
        # Aggregate similarities
        if self.aggregation == 'max':
            agg_similarity = max(similarities)
        elif self.aggregation == 'mean':
            agg_similarity = np.mean(similarities)
        elif self.aggregation == 'weighted':
            # Weight by position (earlier docs often more relevant)
            weights = [1.0 / (i + 1) for i in range(len(similarities))]
            agg_similarity = np.average(similarities, weights=weights)
        else:
            agg_similarity = max(similarities)
        
        # Convert similarity to hallucination score (inverse relationship)
        # High similarity = low hallucination score
        hallucination_score = 1.0 - agg_similarity
        
        # Determine classification
        is_hallucinated = agg_similarity < threshold
        
        # Confidence based on distance from threshold
        distance_from_threshold = abs(agg_similarity - threshold)
        confidence = min(0.5 + distance_from_threshold, 1.0)
        
        latency = (time.time() - start) * 1000
        
        return DetectionResult(
            is_hallucinated=is_hallucinated,
            confidence=confidence,
            hallucination_score=hallucination_score,
            method_name=self.name,
            latency_ms=latency,
            cost_estimate=self.cost,
            explanation=f"Max similarity to context: {agg_similarity:.3f} (threshold: {threshold})",
            details={
                'similarities': similarities,
                'aggregated_similarity': agg_similarity,
                'threshold': threshold,
                'aggregation_method': self.aggregation
            }
        )
    
    def get_features(self) -> Dict:
        """Get detector features for RL state"""
        return {
            'name': self.name,
            'cost': self.cost,
            'typical_latency_ms': self.typical_latency_ms,
            'threshold': self.threshold,
            'model_name': self.model_name,
            'aggregation': self.aggregation
        }


def create_detector(**kwargs) -> SemanticSimilarityDetector:
    """Factory function to create detector instance"""
    return SemanticSimilarityDetector(**kwargs)
