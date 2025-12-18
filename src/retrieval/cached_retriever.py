"""Cached retriever wrapper."""

import logging
from typing import List, Optional, Dict, Any

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.cache.query_cache import QueryCache

logger = logging.getLogger(__name__)


class CachedRetriever(BaseRetriever):
    """
    Wrapper that adds query result caching to any retriever.
    
    Usage:
        base_retriever = HybridRetriever(...)
        retriever = CachedRetriever(base_retriever, ttl_seconds=300)
        
        # First call - computes and caches
        results = retriever.retrieve("query", top_k=5)
        
        # Second call within TTL - returns from cache
        results = retriever.retrieve("query", top_k=5)  # Cache hit!
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        cache_dir: str = ".cache/queries",
        ttl_seconds: int = 300,
        enabled: bool = True,
    ):
        self.base = base_retriever
        self.enabled = enabled
        
        if enabled:
            self.cache = QueryCache(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
        else:
            self.cache = None
        
        logger.info(f"CachedRetriever: enabled={enabled}, ttl={ttl_seconds}s")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """Add documents - clears cache since index changed."""
        self.base.add_documents(texts=texts, metadatas=metadatas, **kwargs)
        
        # Clear query cache when documents change
        if self.cache:
            self.cache.clear()
            logger.info("Cleared query cache after document update")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
        **kwargs,
    ) -> List[RetrievalResult]:
        """Retrieve with caching."""
        if not self.enabled or not use_cache:
            return self.base.retrieve(query, top_k=top_k, **kwargs)
        
        # Check cache
        cached = self.cache.get(query, top_k=top_k)
        if cached is not None:
            logger.debug(f"Query cache hit: {query[:30]}...")
            return cached
        
        # Compute and cache
        results = self.base.retrieve(query, top_k=top_k, **kwargs)
        self.cache.set(query, results, top_k=top_k)
        return results
    
    def health_check(self) -> bool:
        """Check health."""
        return self.base.health_check()
    
    @property
    def stats(self):
        """Get cache stats."""
        if self.cache:
            return self.cache.stats
        return {"enabled": False}
