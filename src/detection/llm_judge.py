"""
LLM-as-Judge Hallucination Detector

Uses a large language model to evaluate whether a response is
grounded in the provided context or contains hallucinations.

Cost: HIGH (~500ms, API costs)
Best for: High-stakes numerical queries, complex reasoning
"""

import json
import os
from typing import List, Dict, Optional

from .base import BaseDetector, DetectionResult


# System prompt for hallucination detection
JUDGE_SYSTEM_PROMPT = """You are an expert fact-checker specialized in financial documents.
Your task is to determine if a response contains hallucinations (fabricated or incorrect information)
when compared to the provided source documents.

A response is considered HALLUCINATED if it:
1. Contains numbers, percentages, or dates not found in the source documents
2. Makes claims about events, products, or services not mentioned in the sources
3. Misrepresents trends, comparisons, or relationships from the source
4. Attributes information to the wrong company or time period

A response is considered GROUNDED if:
1. All factual claims can be verified from the source documents
2. Numbers and dates match the source documents
3. The response accurately represents the information in context

Respond with a JSON object containing:
{
    "is_hallucinated": true/false,
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of your reasoning",
    "specific_issues": ["list of specific hallucinations found, if any"]
}"""


class LLMJudgeDetector(BaseDetector):
    """
    Uses an LLM to judge whether a response is hallucinated.
    
    Supports multiple backends:
    - OpenAI API (GPT-4, GPT-3.5)
    - Anthropic API (Claude)
    - Local models via Ollama
    - Mock mode for testing
    """
    
    name = "llm_judge"
    cost = 1.0  # Highest cost
    typical_latency_ms = 500.0
    
    def __init__(
        self,
        backend: str = 'openai',  # 'openai', 'anthropic', 'ollama', 'mock'
        model: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        threshold: float = 0.5,
        temperature: float = 0.0,
        max_context_chars: int = 8000
    ):
        """
        Initialize the LLM judge detector.
        
        Args:
            backend: Which LLM backend to use
            model: Model name for the backend
            api_key: API key (or set via environment variable)
            threshold: Confidence threshold for hallucination classification
            temperature: LLM temperature (0 for deterministic)
            max_context_chars: Maximum characters of context to include
        """
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.threshold = threshold
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self._client = None
    
    def _get_client(self):
        """Initialize the appropriate API client"""
        if self._client is not None:
            return self._client
        
        if self.backend == 'openai':
            try:
                from openai import OpenAI
                api_key = self.api_key or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
                
        elif self.backend == 'anthropic':
            try:
                import anthropic
                api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
                
        elif self.backend == 'ollama':
            try:
                import requests
                self._client = requests.Session()
            except ImportError:
                raise ImportError("requests package not installed. Run: pip install requests")
                
        elif self.backend == 'mock':
            self._client = 'mock'
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        return self._client
    
    def _build_prompt(
        self,
        question: str,
        response: str,
        context: List[str]
    ) -> str:
        """Build the user prompt for the LLM"""
        # Truncate context if too long
        combined_context = '\n\n---\n\n'.join(context)
        if len(combined_context) > self.max_context_chars:
            combined_context = combined_context[:self.max_context_chars] + "\n... [truncated]"
        
        prompt = f"""## Question Asked
{question}

## Source Documents (from SEC 10-K filings)
{combined_context}

## Response to Evaluate
{response}

## Your Task
Analyze the response and determine if it contains any hallucinations when compared to the source documents. 
Focus especially on:
- Numerical accuracy (revenue figures, percentages, dates)
- Factual claims about products, services, or events
- Correct attribution to companies and time periods

Respond with JSON only."""

        return prompt
    
    def _call_openai(self, prompt: str) -> Dict:
        """Call OpenAI API"""
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        return json.loads(result_text)
    
    def _call_anthropic(self, prompt: str) -> Dict:
        """Call Anthropic API"""
        client = self._get_client()
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt + "\n\nRespond with JSON only."}
            ]
        )
        
        result_text = response.content[0].text
        # Extract JSON from response
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            raise
    
    def _call_ollama(self, prompt: str) -> Dict:
        """Call local Ollama API"""
        import requests
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': self.model,
                'prompt': f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
                'stream': False,
                'format': 'json'
            }
        )
        
        result = response.json()
        return json.loads(result['response'])
    
    def _call_mock(self, prompt: str, response_text: str) -> Dict:
        """
        Mock LLM response for testing without API access.
        Uses heuristics to simulate LLM judgment.
        """
        import re
        
        # Simple heuristics for mock detection
        issues = []
        
        # Check for suspiciously round numbers
        numbers = re.findall(r'\$?(\d+(?:\.\d+)?)\s*(?:billion|million)?', response_text.lower())
        for num in numbers:
            # Very round numbers ending in .0 are often hallucinated
            if '.' not in num and len(num) > 2:
                if int(num) % 10 == 0 and int(num) % 100 == 0:
                    issues.append(f"Suspiciously round number: {num}")
        
        # Check for specific fabrication patterns
        fabrication_patterns = [
            r'first time|first ever|for the first time',
            r'announced plans to|will launch|launching',
            r'surpass(ed|ing)|overtook|exceeding',
            r'investigation|lawsuit|fine',
        ]
        
        for pattern in fabrication_patterns:
            if re.search(pattern, response_text.lower()):
                issues.append(f"Potential fabrication pattern: {pattern}")
        
        # Determine result based on issues found
        is_hallucinated = len(issues) > 0
        confidence = min(0.5 + len(issues) * 0.15, 0.95)
        
        return {
            'is_hallucinated': is_hallucinated,
            'confidence': confidence,
            'explanation': f"Mock analysis found {len(issues)} potential issues",
            'specific_issues': issues
        }
    
    def detect(
        self,
        question: str,
        response: str,
        context: List[str],
        threshold: Optional[float] = None,
        **kwargs
    ) -> DetectionResult:
        """
        Detect hallucination using LLM judgment.
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
        
        # Build prompt
        prompt = self._build_prompt(question, response, context)
        
        # Call appropriate backend
        try:
            if self.backend == 'openai':
                result = self._call_openai(prompt)
            elif self.backend == 'anthropic':
                result = self._call_anthropic(prompt)
            elif self.backend == 'ollama':
                result = self._call_ollama(prompt)
            elif self.backend == 'mock':
                result = self._call_mock(prompt, response)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            # Return uncertain result on API failure
            return DetectionResult(
                is_hallucinated=True,  # Fail safe: assume hallucination
                confidence=0.3,
                hallucination_score=0.7,
                method_name=self.name,
                latency_ms=(time.time() - start) * 1000,
                cost_estimate=self.cost,
                explanation=f"LLM API error: {str(e)}"
            )
        
        latency = (time.time() - start) * 1000
        
        # Extract results
        is_hallucinated = result.get('is_hallucinated', True)
        confidence = result.get('confidence', 0.5)
        explanation = result.get('explanation', 'No explanation provided')
        specific_issues = result.get('specific_issues', [])
        
        # Convert to hallucination score
        hallucination_score = confidence if is_hallucinated else (1.0 - confidence)
        
        # Apply threshold
        final_classification = hallucination_score > threshold
        
        return DetectionResult(
            is_hallucinated=final_classification,
            confidence=confidence,
            hallucination_score=hallucination_score,
            method_name=self.name,
            latency_ms=latency,
            cost_estimate=self.cost,
            explanation=explanation,
            details={
                'backend': self.backend,
                'model': self.model,
                'specific_issues': specific_issues,
                'raw_result': result,
                'threshold': threshold
            }
        )
    
    def get_features(self) -> Dict:
        """Get detector features for RL state"""
        return {
            'name': self.name,
            'cost': self.cost,
            'typical_latency_ms': self.typical_latency_ms,
            'backend': self.backend,
            'model': self.model,
            'threshold': self.threshold
        }


def create_detector(**kwargs) -> LLMJudgeDetector:
    """Factory function to create detector instance"""
    return LLMJudgeDetector(**kwargs)
