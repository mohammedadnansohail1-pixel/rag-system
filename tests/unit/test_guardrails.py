"""Tests for guardrails module."""

import pytest
from src.guardrails.config import GuardrailsConfig, PRODUCTION_CONFIG
from src.guardrails.validator import GuardrailsValidator, get_uncertainty_response
from src.retrieval.base import RetrievalResult


class TestGuardrailsConfig:
    """Tests for GuardrailsConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = GuardrailsConfig()
        assert config.score_threshold == 0.35
        assert config.min_sources == 2

    def test_validate_valid_config(self):
        """Should pass validation with valid config."""
        config = GuardrailsConfig(score_threshold=0.5, min_sources=2)
        config.validate()  # Should not raise

    def test_validate_invalid_threshold(self):
        """Should reject invalid score threshold."""
        config = GuardrailsConfig(score_threshold=1.5)
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_min_sources(self):
        """Should reject invalid min_sources."""
        config = GuardrailsConfig(min_sources=0)
        with pytest.raises(ValueError):
            config.validate()


class TestGuardrailsValidator:
    """Tests for GuardrailsValidator."""

    def test_validate_empty_results(self):
        """Should reject empty results."""
        validator = GuardrailsValidator()
        result = validator.validate([])
        
        assert result.is_valid is False
        assert "No retrieval" in result.rejection_reason

    def test_validate_low_scores(self):
        """Should reject results below threshold."""
        validator = GuardrailsValidator(GuardrailsConfig(score_threshold=0.5))
        results = [
            RetrievalResult(content="test", metadata={}, score=0.3),
            RetrievalResult(content="test", metadata={}, score=0.2),
        ]
        
        result = validator.validate(results)
        assert result.is_valid is False

    def test_validate_insufficient_sources(self):
        """Should reject if too few quality sources."""
        validator = GuardrailsValidator(GuardrailsConfig(
            score_threshold=0.5,
            min_sources=3
        ))
        results = [
            RetrievalResult(content="test", metadata={}, score=0.6),
            RetrievalResult(content="test", metadata={}, score=0.7),
        ]
        
        result = validator.validate(results)
        assert result.is_valid is False
        assert "Insufficient" in result.rejection_reason

    def test_validate_success(self):
        """Should pass with good results."""
        validator = GuardrailsValidator(GuardrailsConfig(
            score_threshold=0.4,
            min_sources=2,
            min_avg_score=0.5
        ))
        results = [
            RetrievalResult(content="test1", metadata={}, score=0.8),
            RetrievalResult(content="test2", metadata={}, score=0.7),
            RetrievalResult(content="test3", metadata={}, score=0.6),
        ]
        
        result = validator.validate(results)
        assert result.is_valid is True
        assert len(result.filtered_results) == 3

    def test_confidence_high(self):
        """Should assess high confidence correctly."""
        validator = GuardrailsValidator()
        results = [
            RetrievalResult(content="test", metadata={}, score=0.8),
            RetrievalResult(content="test", metadata={}, score=0.75),
            RetrievalResult(content="test", metadata={}, score=0.7),
        ]
        
        result = validator.validate(results)
        assert result.confidence == "high"

    def test_confidence_medium(self):
        """Should assess medium confidence correctly."""
        validator = GuardrailsValidator()
        results = [
            RetrievalResult(content="test", metadata={}, score=0.6),
            RetrievalResult(content="test", metadata={}, score=0.55),
        ]
        
        result = validator.validate(results)
        assert result.confidence == "medium"


class TestUncertaintyResponse:
    """Tests for uncertainty response generation."""

    def test_no_retrieval_message(self):
        """Should generate appropriate message for no results."""
        response = get_uncertainty_response("No retrieval results")
        assert "No relevant documents" in response

    def test_insufficient_sources_message(self):
        """Should generate appropriate message for insufficient sources."""
        response = get_uncertainty_response("Insufficient quality sources")
        assert "sufficient coverage" in response

    def test_low_relevance_message(self):
        """Should generate appropriate message for low relevance."""
        response = get_uncertainty_response("Low average relevance")
        assert "directly relevant" in response
