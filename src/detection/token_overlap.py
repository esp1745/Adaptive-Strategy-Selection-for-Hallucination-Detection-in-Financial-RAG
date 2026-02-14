"""
Token Overlap Hallucination Detector

Uses lexical overlap metrics (BLEU, ROUGE) to detect hallucinations.
Low overlap between response and context indicates potential hallucination.

Cost: LOW (~5ms)
Best for: Factual grounding checks, verbatim information
"""

import re
from collections import Counter
from typing import List, Dict, Optional, Tuple

from .base import BaseDetector, DetectionResult


class TokenOverlapDetector(BaseDetector):
    """
    Detects hallucinations by measuring lexical overlap between
    the response and the retrieved context documents.
    
    Uses BLEU-like n-gram precision and ROUGE-like recall metrics.
    
    Hypothesis: A hallucinated response will have lower token overlap
    with source documents, especially for specific terms and numbers.
    """
    
    name = "token_overlap"
    cost = 0.05  # Very low cost (pure computation)
    typical_latency_ms = 5.0
    
    def __init__(
        self,
        threshold: float = 0.3,
        ngram_weights: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1),
        use_rouge: bool = True,
        extract_numbers: bool = True
    ):
        """
        Initialize the token overlap detector.
        
        Args:
            threshold: Overlap below this = hallucination (0-1)
            ngram_weights: Weights for 1-gram, 2-gram, 3-gram, 4-gram
            use_rouge: Whether to include ROUGE-L (longest common subsequence)
            extract_numbers: Whether to give extra weight to numerical matches
        """
        self.threshold = threshold
        self.ngram_weights = ngram_weights
        self.use_rouge = use_rouge
        self.extract_numbers = extract_numbers
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _ngram_precision(
        self,
        response_tokens: List[str],
        context_tokens: List[str],
        n: int
    ) -> float:
        """Calculate n-gram precision (BLEU-style)"""
        response_ngrams = self._get_ngrams(response_tokens, n)
        context_ngrams = self._get_ngrams(context_tokens, n)
        
        if not response_ngrams:
            return 0.0
        
        context_ngram_counts = Counter(context_ngrams)
        response_ngram_counts = Counter(response_ngrams)
        
        matches = 0
        for ngram, count in response_ngram_counts.items():
            matches += min(count, context_ngram_counts.get(ngram, 0))
        
        return matches / len(response_ngrams)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0
        
        # Use dynamic programming with space optimization
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        return prev[n]
    
    def _rouge_l(
        self,
        response_tokens: List[str],
        context_tokens: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE-L scores (LCS-based)"""
        lcs_len = self._lcs_length(response_tokens, context_tokens)
        
        precision = lcs_len / len(response_tokens) if response_tokens else 0.0
        recall = lcs_len / len(context_tokens) if context_tokens else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numerical values from text"""
        # Match numbers including decimals, percentages, currency
        patterns = [
            r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand))?',  # Currency
            r'[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand))?',    # Plain numbers
            r'[\d,]+(?:\.\d+)?%',                                        # Percentages
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            numbers.extend(matches)
        
        # Normalize numbers
        normalized = []
        for num in numbers:
            # Remove formatting
            clean = re.sub(r'[,$%]', '', num)
            clean = re.sub(r'\s*(billion|million|thousand)', '', clean)
            if clean:
                normalized.append(clean)
        
        return normalized
    
    def _number_match_score(
        self,
        response: str,
        context: str
    ) -> float:
        """Score based on numerical value matches"""
        response_nums = set(self._extract_numbers(response))
        context_nums = set(self._extract_numbers(context))
        
        if not response_nums:
            return 1.0  # No numbers to verify
        
        matches = response_nums & context_nums
        return len(matches) / len(response_nums)
    
    def detect(
        self,
        question: str,
        response: str,
        context: List[str],
        threshold: Optional[float] = None,
        **kwargs
    ) -> DetectionResult:
        """
        Detect hallucination using token overlap metrics.
        """
        import time
        start = time.time()
        
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
        
        # Combine all context into one
        combined_context = ' '.join(context)
        
        # Tokenize
        response_tokens = self._tokenize(response)
        context_tokens = self._tokenize(combined_context)
        
        # Calculate n-gram precisions
        ngram_scores = []
        for n in range(1, len(self.ngram_weights) + 1):
            score = self._ngram_precision(response_tokens, context_tokens, n)
            ngram_scores.append(score)
        
        # Weighted average of n-gram scores
        bleu_score = sum(
            w * s for w, s in zip(self.ngram_weights, ngram_scores)
        )
        
        # ROUGE-L scores
        rouge_scores = None
        if self.use_rouge:
            rouge_scores = self._rouge_l(response_tokens, context_tokens)
        
        # Number matching
        number_score = 1.0
        if self.extract_numbers:
            number_score = self._number_match_score(response, combined_context)
        
        # Aggregate scores
        # Weight: 40% BLEU, 30% ROUGE-L F1, 30% number matching
        if self.use_rouge and rouge_scores:
            overlap_score = (
                0.4 * bleu_score +
                0.3 * rouge_scores['f1'] +
                0.3 * number_score
            )
        else:
            overlap_score = 0.5 * bleu_score + 0.5 * number_score
        
        # Convert to hallucination score (inverse)
        hallucination_score = 1.0 - overlap_score
        
        # Classification
        is_hallucinated = overlap_score < threshold
        
        # Confidence
        distance_from_threshold = abs(overlap_score - threshold)
        confidence = min(0.5 + distance_from_threshold, 1.0)
        
        latency = (time.time() - start) * 1000
        
        return DetectionResult(
            is_hallucinated=is_hallucinated,
            confidence=confidence,
            hallucination_score=hallucination_score,
            method_name=self.name,
            latency_ms=latency,
            cost_estimate=self.cost,
            explanation=f"Token overlap score: {overlap_score:.3f} (threshold: {threshold})",
            details={
                'bleu_score': bleu_score,
                'ngram_scores': dict(zip(
                    [f'{i+1}-gram' for i in range(len(ngram_scores))],
                    ngram_scores
                )),
                'rouge_l': rouge_scores,
                'number_match_score': number_score,
                'overlap_score': overlap_score,
                'threshold': threshold
            }
        )
    
    def get_features(self) -> Dict:
        """Get detector features for RL state"""
        return {
            'name': self.name,
            'cost': self.cost,
            'typical_latency_ms': self.typical_latency_ms,
            'threshold': self.threshold,
            'use_rouge': self.use_rouge,
            'extract_numbers': self.extract_numbers
        }


def create_detector(**kwargs) -> TokenOverlapDetector:
    """Factory function to create detector instance"""
    return TokenOverlapDetector(**kwargs)
