"""Cache for query results with TTL."""

import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import pickle

logger = logging.getLogger(__name__)


class QueryCache:
    """
    TTL-based cache for query results.
    
    Caches retrieval results to speed up repeated queries.
    Stores results as generic dicts to avoid circular imports.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/queries",
        ttl_seconds: int = 300,
        max_entries: int = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        
        self._stats = {"hits": 0, "misses": 0}
        
        logger.info(f"Initialized QueryCache: ttl={ttl_seconds}s")
    
    def _hash_query(self, query: str, top_k: int) -> str:
        """Generate hash for query + parameters."""
        key = f"{query}::{top_k}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _get_path(self, query_hash: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{query_hash}.pkl"
    
    def get(self, query: str, top_k: int = 5) -> Optional[List[Any]]:
        """Get cached results for query."""
        query_hash = self._hash_query(query, top_k)
        cache_path = self._get_path(query_hash)
        
        if not cache_path.exists():
            self._stats["misses"] += 1
            return None
        
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            
            # Check TTL
            if time.time() - cached["timestamp"] > self.ttl_seconds:
                cache_path.unlink()
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return cached["results"]
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            self._stats["misses"] += 1
            return None
    
    def set(self, query: str, results: List[Any], top_k: int = 5) -> None:
        """Cache results for query."""
        query_hash = self._hash_query(query, top_k)
        cache_path = self._get_path(query_hash)
        
        try:
            cached = {
                "query": query,
                "top_k": top_k,
                "results": results,
                "timestamp": time.time(),
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cached, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear(self) -> None:
        """Clear all cached queries."""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        self._stats = {"hits": 0, "misses": 0}
        logger.info("Cleared query cache")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        removed = 0
        current_time = time.time()
        
        for cache_path in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if current_time - cached["timestamp"] > self.ttl_seconds:
                    cache_path.unlink()
                    removed += 1
            except Exception:
                cache_path.unlink()
                removed += 1
        
        return removed
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        entries = len(list(self.cache_dir.glob("*.pkl")))
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "entries": entries,
            "ttl_seconds": self.ttl_seconds,
        }
