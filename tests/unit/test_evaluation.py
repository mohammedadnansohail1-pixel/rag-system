"""Tests for evaluation module."""

import pytest
from unittest.mock import MagicMock

from src.evaluation.metrics import (
    EvaluationResult,
    BatchEvaluationResult,
    calculate_context_relevance,
    calculate_answer_relevance_simple,
    calculate_faithfulness_simple,
    calculate_overall_score,
)
from src.evaluation.evaluator import RAGEvaluator


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_context_relevance_high_scores(self):
        """Should return high relevance for high scores."""
        score = calculate_context_relevance(
            query="What is ML?",
            contexts=["Machine learning is...", "ML algorithms..."],
            scores=[0.9, 0.85]
        )
        assert score >= 0.8

    def test_context_relevance_low_scores(self):
        """Should return low relevance for low scores."""
        score = calculate_context_relevance(
            query="What is ML?",
            contexts=["Cooking recipes..."],
            scores=[0.2]
        )
        assert score <= 0.3

    def test_context_relevance_empty(self):
        """Should return 0 for empty scores."""
        score = calculate_context_relevance(
            query="Test",
            contexts=[],
            scores=[]
        )
        assert score == 0.0

    def test_answer_relevance_good_match(self):
        """Should return high score for relevant answer."""
        score = calculate_answer_relevance_simple(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI that enables systems to learn from data."
        )
        assert score >= 0.5

    def test_answer_relevance_poor_match(self):
        """Should return low score for irrelevant answer."""
        score = calculate_answer_relevance_simple(
            query="What is machine learning?",
            answer="The weather today is sunny and warm."
        )
        assert score <= 0.3

    def test_answer_relevance_empty(self):
        """Should return 0 for empty answer."""
        score = calculate_answer_relevance_simple(
            query="Test",
            answer=""
        )
        assert score == 0.0

    def test_faithfulness_grounded(self):
        """Should return high score for grounded answer."""
        score = calculate_faithfulness_simple(
            answer="According to the context, gradient descent is an optimization algorithm.",
            contexts=["Gradient descent is an optimization algorithm used to minimize loss."]
        )
        assert score >= 0.5

    def test_faithfulness_not_grounded(self):
        """Should return lower score for ungrounded answer."""
        score = calculate_faithfulness_simple(
            answer="Quantum computing uses qubits for parallel processing.",
            contexts=["Machine learning uses data to train models."]
        )
        assert score <= 0.5

    def test_faithfulness_empty(self):
        """Should return 0 for empty inputs."""
        score = calculate_faithfulness_simple(
            answer="",
            contexts=[]
        )
        assert score == 0.0

    def test_overall_score_calculation(self):
        """Should calculate weighted overall score."""
        score = calculate_overall_score(
            context_relevance=0.8,
            answer_relevance=0.7,
            faithfulness=0.9
        )
        # Default weights: context=0.3, answer=0.3, faith=0.4
        expected = (0.3 * 0.8 + 0.3 * 0.7 + 0.4 * 0.9)
        assert abs(score - expected) < 0.01

    def test_overall_score_custom_weights(self):
        """Should use custom weights."""
        score = calculate_overall_score(
            context_relevance=1.0,
            answer_relevance=0.0,
            faithfulness=0.0,
            weights={"context_relevance": 1.0, "answer_relevance": 0.0, "faithfulness": 0.0}
        )
        assert score == 1.0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Should create result with all fields."""
        result = EvaluationResult(
            query="Test query",
            answer="Test answer",
            context_relevance=0.8,
            answer_relevance=0.7,
            faithfulness=0.9,
            overall_score=0.8
        )
        
        assert result.query == "Test query"
        assert result.overall_score == 0.8

    def test_repr(self):
        """Should have readable repr."""
        result = EvaluationResult(
            query="Test",
            answer="Answer",
            context_relevance=0.8,
            answer_relevance=0.7,
            faithfulness=0.9,
            overall_score=0.8
        )
        
        repr_str = repr(result)
        assert "0.80" in repr_str


class TestBatchEvaluationResult:
    """Tests for BatchEvaluationResult dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = BatchEvaluationResult(
            results=[],
            avg_context_relevance=0.8,
            avg_answer_relevance=0.7,
            avg_faithfulness=0.9,
            avg_overall=0.8,
            num_samples=10
        )
        
        d = result.to_dict()
        assert d["num_samples"] == 10
        assert d["avg_overall"] == 0.8


class TestRAGEvaluator:
    """Tests for RAGEvaluator."""

    def test_evaluate_response(self):
        """Should evaluate a response."""
        mock_pipeline = MagicMock()
        evaluator = RAGEvaluator(mock_pipeline)
        
        mock_response = MagicMock()
        mock_response.answer = "Machine learning is a type of AI."
        mock_response.sources = [
            {"content": "ML is artificial intelligence", "score": 0.8}
        ]
        mock_response.avg_score = 0.8
        mock_response.confidence = "high"
        mock_response.validation_passed = True
        
        result = evaluator.evaluate_response("What is ML?", mock_response)
        
        assert isinstance(result, EvaluationResult)
        assert result.query == "What is ML?"
        assert 0 <= result.overall_score <= 1

    def test_evaluate_query(self):
        """Should query pipeline and evaluate."""
        mock_pipeline = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.sources = [{"content": "Test", "score": 0.7}]
        mock_response.avg_score = 0.7
        mock_response.confidence = "medium"
        mock_response.validation_passed = True
        mock_pipeline.query.return_value = mock_response
        
        evaluator = RAGEvaluator(mock_pipeline)
        result = evaluator.evaluate_query("Test query")
        
        mock_pipeline.query.assert_called_once()
        assert isinstance(result, EvaluationResult)

    def test_evaluate_batch(self):
        """Should evaluate multiple queries."""
        mock_pipeline = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "Answer"
        mock_response.sources = [{"content": "Context", "score": 0.7}]
        mock_response.avg_score = 0.7
        mock_response.confidence = "medium"
        mock_response.validation_passed = True
        mock_pipeline.query.return_value = mock_response
        
        evaluator = RAGEvaluator(mock_pipeline)
        result = evaluator.evaluate_batch([
            {"query": "Query 1"},
            {"query": "Query 2"},
        ])
        
        assert isinstance(result, BatchEvaluationResult)
        assert result.num_samples == 2
        assert len(result.results) == 2
