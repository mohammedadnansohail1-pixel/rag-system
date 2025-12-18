"""Base types for query classification."""
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QueryComplexity(Enum):
    """Query complexity levels for routing to appropriate reranking strategy."""
    SIMPLE = "simple"      # No reranking - dense/hybrid sufficient
    MEDIUM = "medium"      # Cross-encoder reranking
    COMPLEX = "complex"    # LLM listwise reranking


@dataclass
class ClassificationResult:
    """Result of query classification."""
    complexity: QueryComplexity
    confidence: float
    method: str  # "rules" or "llm"
    
    @property
    def should_rerank(self) -> bool:
        """Whether reranking is recommended."""
        return self.complexity != QueryComplexity.SIMPLE
    
    @property
    def candidates_multiplier(self) -> int:
        """How many extra candidates to fetch for reranking."""
        multipliers = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MEDIUM: 3,
            QueryComplexity.COMPLEX: 5,
        }
        return multipliers[self.complexity]
