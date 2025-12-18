"""Guardrails configuration with sensible defaults."""

from dataclasses import dataclass


@dataclass
class GuardrailsConfig:
    """
    Configuration for RAG guardrails.
    
    Attributes:
        score_threshold: Minimum similarity score for chunks (0.0-1.0)
        min_sources: Minimum number of quality sources required
        max_sources: Maximum sources to pass to LLM
        min_avg_score: Minimum average score across sources
        require_explicit_uncertainty: Force LLM to state uncertainty
        log_filtered_chunks: Log chunks that were filtered out
    """
    # Retrieval filtering
    score_threshold: float = 0.35
    min_sources: int = 2
    max_sources: int = 5
    min_avg_score: float = 0.45
    
    # Response behavior
    require_explicit_uncertainty: bool = True
    
    # Observability
    log_filtered_chunks: bool = True
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        if self.min_sources < 1:
            raise ValueError("min_sources must be at least 1")
        if self.max_sources < self.min_sources:
            raise ValueError("max_sources must be >= min_sources")
        if not 0.0 <= self.min_avg_score <= 1.0:
            raise ValueError("min_avg_score must be between 0.0 and 1.0")


# Production defaults - balanced settings
PRODUCTION_CONFIG = GuardrailsConfig(
    score_threshold=0.35,
    min_sources=2,
    max_sources=5,
    min_avg_score=0.45,
    require_explicit_uncertainty=True,
    log_filtered_chunks=True,
)

# Permissive settings - for testing/development
PERMISSIVE_CONFIG = GuardrailsConfig(
    score_threshold=0.2,
    min_sources=1,
    max_sources=10,
    min_avg_score=0.3,
    require_explicit_uncertainty=False,
    log_filtered_chunks=False,
)
