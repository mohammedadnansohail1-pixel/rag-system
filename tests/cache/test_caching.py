"""Tests for caching system."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.cache.embedding_cache import EmbeddingCache
from src.cache.query_cache import QueryCache


class TestEmbeddingCache:
    """Test embedding cache."""
    
    @pytest.fixture
    def cache(self):
        """Create temporary cache."""
        cache_dir = tempfile.mkdtemp()
        cache = EmbeddingCache(cache_dir=cache_dir, model_name="test")
        yield cache
        shutil.rmtree(cache_dir)
    
    def test_set_and_get(self, cache):
        """Can store and retrieve embeddings."""
        text = "hello world"
        embedding = [0.1, 0.2, 0.3]
        
        cache.set(text, embedding)
        result = cache.get(text)
        
        assert result == embedding
    
    def test_miss_returns_none(self, cache):
        """Missing key returns None."""
        result = cache.get("nonexistent")
        assert result is None
    
    def test_stats_tracking(self, cache):
        """Tracks hits and misses."""
        cache.get("miss1")
        cache.get("miss2")
        cache.set("text", [0.1])
        cache.get("text")
        
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
    
    def test_clear(self, cache):
        """Can clear cache."""
        cache.set("text", [0.1])
        cache.clear()
        
        result = cache.get("text")
        assert result is None


class TestQueryCache:
    """Test query cache."""
    
    @pytest.fixture
    def cache(self):
        """Create temporary cache."""
        cache_dir = tempfile.mkdtemp()
        cache = QueryCache(cache_dir=cache_dir, ttl_seconds=60)
        yield cache
        shutil.rmtree(cache_dir)
    
    def test_set_and_get(self, cache):
        """Can store and retrieve query results."""
        query = "test query"
        results = [{"content": "result1"}, {"content": "result2"}]
        
        cache.set(query, results, top_k=5)
        retrieved = cache.get(query, top_k=5)
        
        assert retrieved == results
    
    def test_different_top_k(self, cache):
        """Different top_k values are cached separately."""
        query = "test query"
        results_5 = [{"content": "5"}]
        results_10 = [{"content": "10"}]
        
        cache.set(query, results_5, top_k=5)
        cache.set(query, results_10, top_k=10)
        
        assert cache.get(query, top_k=5) == results_5
        assert cache.get(query, top_k=10) == results_10
    
    def test_ttl_expiry(self, cache):
        """Entries expire after TTL."""
        import time
        
        # Create cache with 1 second TTL
        cache.ttl_seconds = 1
        cache.set("query", [{"content": "test"}], top_k=5)
        
        # Should be available immediately
        assert cache.get("query", top_k=5) is not None
        
        # Wait for expiry
        time.sleep(1.5)
        
        # Should be expired
        assert cache.get("query", top_k=5) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
