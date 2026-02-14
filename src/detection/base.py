"""
Base interface for hallucination detection methods
All detectors implement the same interface for easy comparison
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
import time


@dataclass
class DetectionResult:
    """Result from a hallucination detection method"""
    
    # Core outputs
    is_hallucinated: bool          # Binary classification
    confidence: float               # 0.0 to 1.0 confidence score
    hallucination_score: float      # Raw score (higher = more likely hallucinated)
    
    # Metadata
    method_name: str               # Name of detection method
    latency_ms: float              # Time taken in milliseconds
    cost_estimate: float           # Relative cost (0.0 to 1.0)
    
    # Optional details
    explanation: Optional[str] = None  # Why this classification was made
    details: Optional[Dict] = None     # Method-specific details
    
    def to_dict(self) -> Dict:
        return {
            'is_hallucinated': self.is_hallucinated,
            'confidence': self.confidence,
            'hallucination_score': self.hallucination_score,
            'method_name': self.method_name,
            'latency_ms': self.latency_ms,
            'cost_estimate': self.cost_estimate,
            'explanation': self.explanation,
            'details': self.details
        }


class BaseDetector(ABC):
    """Abstract base class for hallucination detectors"""
    
    # Class-level attributes for RL state features
    name: str = "base"
    cost: float = 0.0           # Relative cost: 0.0 (free) to 1.0 (expensive)
    typical_latency_ms: float = 0.0
    
    @abstractmethod
    def detect(
        self,
        question: str,
        response: str,
        context: List[str],
        **kwargs
    ) -> DetectionResult:
        """
        Detect if a response is hallucinated given the question and context.
        
        Args:
            question: The original question asked
            response: The generated response to evaluate
            context: List of retrieved document chunks used to generate response
            **kwargs: Method-specific parameters
            
        Returns:
            DetectionResult with classification and metadata
        """
        pass
    
    def _timed_detect(
        self,
        question: str,
        response: str,
        context: List[str],
        **kwargs
    ) -> DetectionResult:
        """Wrapper that adds timing to detect calls"""
        start = time.time()
        result = self.detect(question, response, context, **kwargs)
        result.latency_ms = (time.time() - start) * 1000
        return result
    
    @abstractmethod
    def get_features(self) -> Dict:
        """
        Get features about this detector for RL state.
        Used by the RL agent to understand detector characteristics.
        """
        pass


class DetectorRegistry:
    """Registry of available detection methods"""
    
    _detectors: Dict[str, BaseDetector] = {}
    
    @classmethod
    def register(cls, detector: BaseDetector):
        """Register a detector instance"""
        cls._detectors[detector.name] = detector
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseDetector]:
        """Get a detector by name"""
        return cls._detectors.get(name)
    
    @classmethod
    def list_detectors(cls) -> List[str]:
        """List all registered detector names"""
        return list(cls._detectors.keys())
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseDetector]:
        """Get all registered detectors"""
        return cls._detectors.copy()
