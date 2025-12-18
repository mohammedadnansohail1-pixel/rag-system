"""Tests for evaluation module."""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mrr,
    hit_rate,
    ndcg_at_k,
    EvaluationResult,
    BatchEvaluationResult,
    calculate_context_relevance,
    calculate_answer_relevance_simple,
    calculate_faithfulness_simple,
    calculate_overall_score,
)
from src.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
    QueryResult,
    EvaluationResult as RetrievalEvaluationResult,
)


class TestRetrievalMetrics:
    """Tests for basic retrieval metrics."""
    
    def test_precision_at_k(self):
        """Should calculate precision correctly."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        
        assert precision_at_k(retrieved, relevant, 1) == 1.0  # a is relevant
        assert precision_at_k(retrieved, relevant, 2) == 0.5  # 1 of 2
        assert precision_at_k(retrieved, relevant, 5) == 0.6  # 3 of 5
    
    def test_recall_at_k(self):
        """Should calculate recall correctly."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e", "f"}
        
        assert recall_at_k(retrieved, relevant, 1) == 0.25  # 1 of 4
        assert recall_at_k(retrieved, relevant, 5) == 0.75  # 3 of 4
    
    def test_f1_at_k(self):
        """Should calculate F1 correctly."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        
        f1 = f1_at_k(retrieved, relevant, 3)
        # P=2/3, R=1.0, F1 = 2 * (2/3 * 1) / (2/3 + 1) = 0.8
        assert abs(f1 - 0.8) < 0.01
    
    def test_mrr(self):
        """Should calculate MRR correctly."""
        # First result is relevant
        assert mrr(["a", "b", "c"], {"a"}) == 1.0
        # Second result is relevant
        assert mrr(["a", "b", "c"], {"b"}) == 0.5
        # Third result is relevant
        assert abs(mrr(["a", "b", "c"], {"c"}) - 0.333) < 0.01
        # No relevant results
        assert mrr(["a", "b", "c"], {"d"}) == 0.0
    
    def test_hit_rate(self):
        """Should calculate hit rate correctly."""
        retrieved = ["a", "b", "c", "d", "e"]
        
        assert hit_rate(retrieved, {"a"}, 1) == 1.0
        assert hit_rate(retrieved, {"c"}, 2) == 0.0
        assert hit_rate(retrieved, {"c"}, 3) == 1.0
        assert hit_rate(retrieved, {"z"}, 5) == 0.0
    
    def test_ndcg_at_k(self):
        """Should calculate NDCG correctly."""
        retrieved = ["a", "b", "c"]
        relevance_map = {"a": 3, "b": 2, "c": 1}  # Perfect ranking
        
        # Perfect ranking should give NDCG = 1
        ndcg = ndcg_at_k(retrieved, relevance_map, 3)
        assert abs(ndcg - 1.0) < 0.01


class TestRAGMetrics:
    """Tests for RAG-specific metrics."""
    
    def test_context_relevance(self):
        """Should calculate context relevance."""
        contexts = ["relevant text", "more text"]
        scores = [0.9, 0.7]
        
        relevance = calculate_context_relevance("query", contexts, scores)
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.5  # High scores should give high relevance
    
    def test_context_relevance_empty(self):
        """Should handle empty inputs."""
        assert calculate_context_relevance("query", [], []) == 0.0
    
    def test_answer_relevance(self):
        """Should calculate answer relevance."""
        query = "What is machine learning?"
        answer = "Machine learning is a subset of AI that enables systems to learn from data."
        
        relevance = calculate_answer_relevance_simple(query, answer)
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.3  # Should have some overlap
    
    def test_answer_relevance_empty(self):
        """Should handle empty inputs."""
        assert calculate_answer_relevance_simple("", "answer") == 0.0
        assert calculate_answer_relevance_simple("query", "") == 0.0
    
    def test_faithfulness(self):
        """Should calculate faithfulness."""
        answer = "The company reported revenue growth of 15%."
        contexts = ["Annual report shows revenue growth of 15% year over year."]
        
        faithfulness = calculate_faithfulness_simple(answer, contexts)
        assert 0.0 <= faithfulness <= 1.0
        assert faithfulness > 0.1  # Should have overlap
    
    def test_faithfulness_empty(self):
        """Should handle empty inputs."""
        assert calculate_faithfulness_simple("", []) == 0.0
    
    def test_overall_score(self):
        """Should calculate weighted overall score."""
        score = calculate_overall_score(0.8, 0.7, 0.9)
        assert 0.0 <= score <= 1.0
        
        # With custom weights
        weights = {"context_relevance": 0.5, "answer_relevance": 0.3, "faithfulness": 0.2}
        score2 = calculate_overall_score(0.8, 0.7, 0.9, weights)
        assert 0.0 <= score2 <= 1.0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_creation(self):
        """Should create EvaluationResult."""
        result = EvaluationResult(
            query="test query",
            answer="test answer",
            context_relevance=0.8,
            answer_relevance=0.7,
            faithfulness=0.9,
            overall_score=0.8,
        )
        
        assert result.query == "test query"
        assert result.context_relevance == 0.8
        assert result.overall_score == 0.8
    
    def test_with_details(self):
        """Should store additional details."""
        result = EvaluationResult(
            query="test",
            answer="answer",
            context_relevance=0.8,
            answer_relevance=0.7,
            faithfulness=0.9,
            overall_score=0.8,
            details={"num_sources": 5, "confidence": "high"}
        )
        
        assert result.details["num_sources"] == 5
        assert result.details["confidence"] == "high"


class TestBatchEvaluationResult:
    """Tests for BatchEvaluationResult."""
    
    def test_creation(self):
        """Should create batch result."""
        results = [
            EvaluationResult("q1", "a1", 0.8, 0.7, 0.9, 0.8),
            EvaluationResult("q2", "a2", 0.6, 0.5, 0.7, 0.6),
        ]
        
        batch = BatchEvaluationResult(
            results=results,
            avg_context_relevance=0.7,
            avg_answer_relevance=0.6,
            avg_faithfulness=0.8,
            avg_overall=0.7,
            num_samples=2,
        )
        
        assert batch.num_samples == 2
        assert batch.avg_overall == 0.7
    
    def test_summary(self):
        """Should generate summary string."""
        batch = BatchEvaluationResult(
            results=[],
            avg_context_relevance=0.7,
            avg_answer_relevance=0.6,
            avg_faithfulness=0.8,
            avg_overall=0.7,
            num_samples=10,
        )
        
        summary = batch.summary()
        assert "RAG EVALUATION RESULTS" in summary
        assert "10" in summary


class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator."""
    
    def test_init(self):
        """Should initialize with retriever."""
        mock_retriever = MagicMock()
        evaluator = RetrievalEvaluator(mock_retriever)
        assert evaluator.retriever == mock_retriever
    
    def test_evaluate_single(self):
        """Should evaluate single query."""
        # Mock retriever
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "test content"
        mock_result.score = 0.9
        mock_result.metadata = {"id": "doc1"}
        mock_retriever.retrieve.return_value = [mock_result]
        
        evaluator = RetrievalEvaluator(mock_retriever)
        result = evaluator.evaluate_single("test query", {"doc1"}, k=5)
        
        assert result.query == "test query"
        assert result.precision == 1.0
        assert result.hit == True


class TestQueryResult:
    """Tests for QueryResult dataclass."""
    
    def test_creation(self):
        """Should create QueryResult."""
        qr = QueryResult(
            query="test",
            retrieved_ids=["a", "b"],
            retrieved_contents=["content a", "content b"],
            scores=[0.9, 0.8],
            relevant_ids={"a"},
            latency_ms=100.0,
        )
        
        assert qr.query == "test"
        assert len(qr.retrieved_ids) == 2
        assert qr.latency_ms == 100.0
