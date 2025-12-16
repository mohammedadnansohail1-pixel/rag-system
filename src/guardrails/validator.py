"""Guardrails validator for RAG responses."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.guardrails.config import GuardrailsConfig, PRODUCTION_CONFIG
from src.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of guardrails validation.
    
    Attributes:
        is_valid: Whether the retrieval passed validation
        filtered_results: Results that passed filtering
        rejection_reason: Why validation failed (if applicable)
        avg_score: Average score of filtered results
        confidence: Overall confidence level (low/medium/high)
    """
    is_valid: bool
    filtered_results: List[RetrievalResult]
    rejection_reason: Optional[str] = None
    avg_score: float = 0.0
    confidence: str = "low"
    
    @property
    def confidence_emoji(self) -> str:
        """Return emoji indicator for confidence."""
        return {
            "high": "🟢",
            "medium": "🟡",
            "low": "🔴"
        }.get(self.confidence, "⚪")


class GuardrailsValidator:
    """
    Validates retrieval results before sending to LLM.
    
    Implements multiple layers of defense:
    1. Score threshold filtering
    2. Minimum source requirements
    3. Average score validation
    4. Confidence assessment
    
    Usage:
        validator = GuardrailsValidator(config)
        result = validator.validate(retrieval_results)
        
        if result.is_valid:
            # Safe to generate response
            pass
        else:
            # Return uncertainty response
            pass
    """
    
    def __init__(self, config: Optional[GuardrailsConfig] = None):
        """
        Args:
            config: Guardrails configuration (uses production defaults if None)
        """
        self.config = config or PRODUCTION_CONFIG
        self.config.validate()
        
        logger.info(
            f"Initialized GuardrailsValidator: "
            f"threshold={self.config.score_threshold}, "
            f"min_sources={self.config.min_sources}"
        )
    
    def validate(self, results: List[RetrievalResult]) -> ValidationResult:
        """
        Validate retrieval results against guardrails.
        
        Args:
            results: Raw retrieval results
            
        Returns:
            ValidationResult with filtered results and metadata
        """
        if not results:
            return ValidationResult(
                is_valid=False,
                filtered_results=[],
                rejection_reason="No retrieval results",
                confidence="low"
            )
        
        # Step 1: Filter by score threshold
        filtered = self._filter_by_score(results)
        
        # Step 2: Check minimum sources
        if len(filtered) < self.config.min_sources:
            return ValidationResult(
                is_valid=False,
                filtered_results=filtered,
                rejection_reason=f"Insufficient quality sources: {len(filtered)} < {self.config.min_sources}",
                avg_score=self._calculate_avg_score(filtered),
                confidence="low"
            )
        
        # Step 3: Limit to max sources
        filtered = filtered[:self.config.max_sources]
        
        # Step 4: Check average score
        avg_score = self._calculate_avg_score(filtered)
        if avg_score < self.config.min_avg_score:
            return ValidationResult(
                is_valid=False,
                filtered_results=filtered,
                rejection_reason=f"Low average relevance: {avg_score:.2f} < {self.config.min_avg_score}",
                avg_score=avg_score,
                confidence="low"
            )
        
        # Step 5: Determine confidence level
        confidence = self._assess_confidence(filtered, avg_score)
        
        logger.info(
            f"Validation passed: {len(filtered)} sources, "
            f"avg_score={avg_score:.2f}, confidence={confidence}"
        )
        
        return ValidationResult(
            is_valid=True,
            filtered_results=filtered,
            avg_score=avg_score,
            confidence=confidence
        )
    
    def _filter_by_score(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Filter results below score threshold."""
        filtered = []
        rejected = []
        
        for r in results:
            if r.score >= self.config.score_threshold:
                filtered.append(r)
            else:
                rejected.append(r)
        
        if rejected and self.config.log_filtered_chunks:
            logger.debug(
                f"Filtered {len(rejected)} chunks below threshold "
                f"(scores: {[f'{r.score:.2f}' for r in rejected]})"
            )
        
        return filtered
    
    def _calculate_avg_score(self, results: List[RetrievalResult]) -> float:
        """Calculate average score of results."""
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
    
    def _assess_confidence(
        self, 
        results: List[RetrievalResult], 
        avg_score: float
    ) -> str:
        """
        Assess confidence level based on results.
        
        High: 3+ sources with avg score > 0.7
        Medium: 2+ sources with avg score > 0.5
        Low: Everything else
        """
        if len(results) >= 3 and avg_score >= 0.7:
            return "high"
        elif len(results) >= 2 and avg_score >= 0.5:
            return "medium"
        else:
            return "low"


def get_uncertainty_response(reason: str) -> str:
    """
    Generate appropriate uncertainty response.
    
    Args:
        reason: Why the query couldn't be answered
        
    Returns:
        User-friendly uncertainty message
    """
    base_message = (
        "I don't have enough reliable information in my knowledge base "
        "to answer this question accurately."
    )
    
    if "No retrieval" in reason:
        return f"{base_message} No relevant documents were found."
    elif "Insufficient" in reason:
        return f"{base_message} The available sources don't provide sufficient coverage of this topic."
    elif "Low average" in reason:
        return f"{base_message} The retrieved information doesn't appear directly relevant to your question."
    else:
        return base_message
