"""
Hallucination Detection Methods

This module provides four detection strategies with varying cost/accuracy tradeoffs:

1. Semantic Similarity (semantic_similarity)
   - Cost: LOW (~10ms)
   - Uses: Sentence embeddings + cosine similarity
   - Best for: Qualitative questions, topic drift detection

2. Token Overlap (token_overlap)
   - Cost: LOW (~5ms)  
   - Uses: BLEU/ROUGE metrics, n-gram matching
   - Best for: Factual grounding, verbatim information

3. BERT NLI Classifier (bert_nli)
   - Cost: MEDIUM (~50ms)
   - Uses: Natural Language Inference model
   - Best for: Mixed content, logical contradictions

4. LLM-as-Judge (llm_judge)
   - Cost: HIGH (~500ms, API costs)
   - Uses: GPT-4 or similar LLM
   - Best for: High-stakes numerical queries, complex reasoning

Usage:
    from src.detection import create_detector, get_all_detectors
    
    # Create a specific detector
    detector = create_detector('semantic_similarity')
    result = detector.detect(question, response, context)
    
    # Get all detectors
    detectors = get_all_detectors()
"""

from typing import Dict, Optional, List

from .base import (
    BaseDetector,
    DetectionResult,
    DetectorRegistry
)

from .semantic_similarity import SemanticSimilarityDetector
from .token_overlap import TokenOverlapDetector
from .bert_nli import BERTNLIDetector
from .llm_judge import LLMJudgeDetector


# Detector name to class mapping
DETECTOR_CLASSES = {
    'semantic_similarity': SemanticSimilarityDetector,
    'token_overlap': TokenOverlapDetector,
    'bert_nli': BERTNLIDetector,
    'llm_judge': LLMJudgeDetector,
}

# Default configurations for each detector
DEFAULT_CONFIGS = {
    'semantic_similarity': {
        'threshold': 0.5,
        'aggregation': 'max'
    },
    'token_overlap': {
        'threshold': 0.3,
        'use_rouge': True,
        'extract_numbers': True
    },
    'bert_nli': {
        'threshold': 0.5,
        'aggregation': 'max_contradiction'
    },
    'llm_judge': {
        'backend': 'ollama',  # Use local Ollama with Qwen2.5-7B
        'model': 'qwen2.5:7b',
        'threshold': 0.5
    }
}


def create_detector(
    name: str,
    **kwargs
) -> BaseDetector:
    """
    Create a detector instance by name.
    
    Args:
        name: Detector name ('semantic_similarity', 'token_overlap', 
              'bert_nli', 'llm_judge')
        **kwargs: Override default configuration
        
    Returns:
        Configured detector instance
    """
    if name not in DETECTOR_CLASSES:
        available = ', '.join(DETECTOR_CLASSES.keys())
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")
    
    # Merge default config with overrides
    config = DEFAULT_CONFIGS.get(name, {}).copy()
    config.update(kwargs)
    
    # Create and return detector
    detector_class = DETECTOR_CLASSES[name]
    return detector_class(**config)


def get_all_detectors(**global_kwargs) -> Dict[str, BaseDetector]:
    """
    Create instances of all available detectors.
    
    Args:
        **global_kwargs: Configuration applied to all detectors
        
    Returns:
        Dictionary mapping detector names to instances
    """
    detectors = {}
    for name in DETECTOR_CLASSES:
        try:
            detectors[name] = create_detector(name, **global_kwargs)
        except Exception as e:
            print(f"Warning: Could not create {name} detector: {e}")
    return detectors


def list_detectors() -> List[str]:
    """List all available detector names"""
    return list(DETECTOR_CLASSES.keys())


def get_detector_info() -> Dict[str, Dict]:
    """
    Get information about all detectors.
    
    Returns:
        Dictionary with detector metadata (cost, latency, description)
    """
    info = {}
    for name, cls in DETECTOR_CLASSES.items():
        info[name] = {
            'name': cls.name,
            'cost': cls.cost,
            'typical_latency_ms': cls.typical_latency_ms,
            'class': cls.__name__,
            'description': cls.__doc__.strip().split('\n')[0] if cls.__doc__ else ''
        }
    return info


# Export public API
__all__ = [
    # Base classes
    'BaseDetector',
    'DetectionResult',
    'DetectorRegistry',
    
    # Detector classes
    'SemanticSimilarityDetector',
    'TokenOverlapDetector', 
    'BERTNLIDetector',
    'LLMJudgeDetector',
    
    # Factory functions
    'create_detector',
    'get_all_detectors',
    'list_detectors',
    'get_detector_info',
    
    # Constants
    'DETECTOR_CLASSES',
    'DEFAULT_CONFIGS',
]
