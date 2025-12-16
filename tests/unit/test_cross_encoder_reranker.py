"""Tests for cross-encoder reranker."""
import pytest
from src.reranking import RerankerFactory, RerankResult, get_registered_rerankers


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""
    
    def test_registration(self):
        """Verify cross_encoder is registered."""
        assert "cross_encoder" in get_registered_rerankers()
    
    def test_factory_create(self):
        """Can create via factory."""
        reranker = RerankerFactory.create("cross_encoder")
        assert reranker is not None
        assert "cross-encoder" in reranker.model_name
    
    def test_factory_from_config(self):
        """Can create from config dict."""
        config = {
            "type": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }
        reranker = RerankerFactory.from_config(config)
        assert reranker is not None
    
    def test_rerank_returns_sorted_results(self):
        """Reranking returns results sorted by score."""
        reranker = RerankerFactory.create("cross_encoder")
        
        results = reranker.rerank(
            query="python programming",
            documents=[
                "Java is a programming language",
                "Python is a versatile programming language",
                "Cooking recipes for beginners",
            ]
        )
        
        # Should return all 3
        assert len(results) == 3
        
        # Should be sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Python doc should be ranked first
        assert "Python" in results[0].content
    
    def test_rerank_with_top_n(self):
        """top_n limits results."""
        reranker = RerankerFactory.create("cross_encoder")
        
        results = reranker.rerank(
            query="test query",
            documents=["doc1", "doc2", "doc3", "doc4", "doc5"],
            top_n=2
        )
        
        assert len(results) == 2
    
    def test_rerank_with_metadata(self):
        """Metadata is preserved in results."""
        reranker = RerankerFactory.create("cross_encoder")
        
        results = reranker.rerank(
            query="kafka",
            documents=["Kafka streaming", "Redis caching"],
            metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}]
        )
        
        # Metadata should be present
        for r in results:
            assert "source" in r.metadata
    
    def test_rerank_empty_documents(self):
        """Empty documents returns empty list."""
        reranker = RerankerFactory.create("cross_encoder")
        
        results = reranker.rerank(query="test", documents=[])
        assert results == []
    
    def test_rank_change_property(self):
        """RerankResult.rank_change calculated correctly."""
        result = RerankResult(
            content="test",
            metadata={},
            score=0.5,
            original_rank=5,
            new_rank=2
        )
        
        # Moved up 3 positions
        assert result.rank_change == 3
    
    def test_health_check(self):
        """Health check returns True when operational."""
        reranker = RerankerFactory.create("cross_encoder")
        
        assert reranker.health_check() is True
