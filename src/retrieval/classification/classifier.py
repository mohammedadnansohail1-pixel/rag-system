"""Main query complexity classifier with adaptive routing."""
import logging
from typing import Optional

from .base import QueryComplexity, ClassificationResult
from .rules import RuleBasedClassifier
from .llm_classifier import LLMClassifier

logger = logging.getLogger(__name__)


class QueryComplexityClassifier:
    """
    Classify queries by complexity for adaptive reranking.
    
    Uses rule-based classification with optional LLM fallback.
    
    Usage:
        classifier = QueryComplexityClassifier()
        complexity = classifier.classify("What is Azure?")
        
        # Check if reranking needed
        if classifier.should_rerank("Compare AWS vs Azure"):
            # apply reranker
            pass
    """
    
    def __init__(
        self,
        llm_host: str = "http://localhost:11434",
        llm_model: str = "llama3.2",
        confidence_threshold: float = 0.7,
        use_llm_fallback: bool = True,
        llm_timeout: float = 5.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.use_llm_fallback = use_llm_fallback
        
        self._rule_classifier = RuleBasedClassifier()
        self._llm_classifier = LLMClassifier(
            host=llm_host,
            model=llm_model,
            timeout=llm_timeout,
        ) if use_llm_fallback else None
        
        logger.info(
            f"Initialized QueryComplexityClassifier: "
            f"llm_fallback={use_llm_fallback}, threshold={confidence_threshold}"
        )
    
    def classify(self, query: str) -> QueryComplexity:
        """Classify query complexity."""
        return self.classify_detailed(query).complexity
    
    def classify_detailed(self, query: str) -> ClassificationResult:
        """Classify query with full details."""
        complexity, confidence = self._rule_classifier.classify(query)
        
        logger.debug(
            f"Rule-based: '{query[:40]}...' -> "
            f"{complexity.value} (conf={confidence:.2f})"
        )
        
        # LLM fallback for low confidence
        if (
            confidence < self.confidence_threshold 
            and self._llm_classifier 
            and self._llm_classifier.is_available()
        ):
            llm_result = self._llm_classifier.classify(query)
            if llm_result is not None:
                logger.debug(f"LLM override: {complexity.value} -> {llm_result.value}")
                return ClassificationResult(
                    complexity=llm_result,
                    confidence=0.85,
                    method="llm",
                )
        
        return ClassificationResult(
            complexity=complexity,
            confidence=confidence,
            method="rules",
        )
    
    def should_rerank(self, query: str) -> bool:
        """Quick check if query should be reranked."""
        return self.classify(query) != QueryComplexity.SIMPLE
    
    def get_candidates_multiplier(self, query: str) -> int:
        """Get how many extra candidates to fetch."""
        return self.classify_detailed(query).candidates_multiplier
