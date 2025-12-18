"""Cache for embeddings to avoid recomputation."""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import pickle

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Disk-based cache for embeddings.
    
    Caches embeddings by content hash to avoid re-embedding identical text.
    Useful for:
    - Re-ingesting documents with minor changes
    - Development/testing iterations
    - Multi-document ingestion with shared content
    
    Usage:
        cache = EmbeddingCache(cache_dir=".cache/embeddings")
        
        # Check cache first
        cached = cache.get(text)
        if cached is None:
            embedding = embeddings.embed(text)
            cache.set(text, embedding)
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/embeddings",
        model_name: str = "default",
        max_size_mb: int = 500,
    ):
        """
        Args:
            cache_dir: Directory to store cache files
            model_name: Embedding model name (for cache isolation)
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir) / model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self._stats = {"hits": 0, "misses": 0}
        
        logger.info(f"Initialized EmbeddingCache: {self.cache_dir}")
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _get_path(self, text_hash: str) -> Path:
        """Get cache file path for hash."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.pkl"
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        text_hash = self._hash_text(text)
        cache_path = self._get_path(text_hash)
        
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                self._stats["hits"] += 1
                return embedding
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        self._stats["misses"] += 1
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text."""
        text_hash = self._hash_text(text)
        cache_path = self._get_path(text_hash)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def get_batch(self, texts: List[str]) -> Dict[int, List[float]]:
        """
        Get cached embeddings for multiple texts.
        
        Returns:
            Dict mapping index -> embedding for texts that have cached values
        """
        cached = {}
        for i, text in enumerate(texts):
            embedding = self.get(text)
            if embedding is not None:
                cached[i] = embedding
        return cached
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Cache embeddings for multiple texts."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0}
        logger.info("Cleared embedding cache")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        # Calculate cache size
        size_bytes = sum(f.stat().st_size for f in self.cache_dir.rglob("*.pkl"))
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "size_mb": f"{size_bytes / 1024 / 1024:.1f}",
            "cache_dir": str(self.cache_dir),
        }
