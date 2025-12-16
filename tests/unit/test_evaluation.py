"""Tests for evaluation module."""

import pytest
from unittest.mock import MagicMock

from src.evaluation.metrics import (
    RAGEvaluator,
    EvaluationResult,
    BatchEvaluationResult,
)


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = EvaluationResult(
            faithfulness=0.8,
            relevance=0.7,
            answer_relevance=0.9,
            context_precision=0.6,
            overall_score=0.75
        )
        
        d = result.to_dict()
        
        assert d["faithfulness"] == 0.8
        assert d["relevance"] == 0.7
        assert d["overall_score"] == 0.75


class TestRAGEvaluator:
    """Tests for RAGEvaluator."""

    def test_init(self):
        """Should initialize with LLM."""
        mock_llm = MagicMock()
        evaluator = RAGEvaluator(mock_llm)
        
        assert evaluator.llm == mock_llm
        assert "faithfulness" in evaluator.weights

    def test_evaluate(self):
        """Should evaluate RAG response."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "8"
        
        evaluator = RAGEvaluator(mock_llm)
        
        result = evaluator.evaluate(
            query="What is ML?",
            answer="ML is machine learning.",
            contexts=["Machine learning is a type of AI."]
        )
        
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.overall_score <= 1

    def test_evaluate_empty_contexts(self):
        """Should handle empty contexts."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "5"
        
        evaluator = RAGEvaluator(mock_llm)
        
        result = evaluator.evaluate(
            query="What is ML?",
            answer="I don't know.",
            contexts=[]
        )
        
        assert result.faithfulness == 0.0
        assert result.relevance == 0.0
        assert result.context_precision == 0.0

    def test_evaluate_batch(self):
        """Should evaluate multiple queries."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "7"
        
        evaluator = RAGEvaluator(mock_llm)
        
        result = evaluator.evaluate_batch(
            queries=["Q1", "Q2"],
            answers=["A1", "A2"],
            contexts_list=[["C1"], ["C2"]]
        )
        
        assert isinstance(result, BatchEvaluationResult)
        assert len(result.results) == 2
        assert 0 <= result.avg_overall_score <= 1

    def test_extract_score(self):
        """Should extract score from response."""
        mock_llm = MagicMock()
        evaluator = RAGEvaluator(mock_llm)
        
        assert evaluator._extract_score("8") == 8.0
        assert evaluator._extract_score("Score: 7.5") == 7.5
        assert evaluator._extract_score("I rate this 9 out of 10") == 9.0
        assert evaluator._extract_score("no number here") == 5.0

    def test_extract_score_clamps_values(self):
        """Should clamp scores to 0-10."""
        mock_llm = MagicMock()
        evaluator = RAGEvaluator(mock_llm)
        
        assert evaluator._extract_score("15") == 10.0
        assert evaluator._extract_score("-5") == 5.0  # No negative match

    def test_custom_weights(self):
        """Should accept custom weights."""
        mock_llm = MagicMock()
        custom_weights = {
            "faithfulness": 0.5,
            "relevance": 0.2,
            "answer_relevance": 0.2,
            "context_precision": 0.1,
        }
        
        evaluator = RAGEvaluator(mock_llm, weights=custom_weights)
        
        assert evaluator.weights["faithfulness"] == 0.5


class TestBatchEvaluationResult:
    """Tests for BatchEvaluationResult."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = BatchEvaluationResult(
            results=[],
            avg_faithfulness=0.8,
            avg_relevance=0.7,
            avg_answer_relevance=0.9,
            avg_context_precision=0.6,
            avg_overall_score=0.75
        )
        
        d = result.to_dict()
        
        assert d["num_queries"] == 0
        assert d["avg_faithfulness"] == 0.8
        assert d["avg_overall_score"] == 0.75
