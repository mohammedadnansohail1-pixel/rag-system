"""Cached embedding wrapper."""

import logging
from typing import List

from src.embeddings.base import BaseEmbeddings
from src.cache.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class CachedEmbeddings(BaseEmbeddings):
    """
    Wrapper that adds caching to any embedding provider.
    
    Usage:
        base_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        embeddings = CachedEmbeddings(base_embeddings)
        
        # First call - computes and caches
        vec = embeddings.embed_text("hello world")
        
        # Second call - returns from cache
        vec = embeddings.embed_text("hello world")  # Cache hit!
    """
    
    def __init__(
        self,
        base_embeddings: BaseEmbeddings,
        cache_dir: str = ".cache/embeddings",
        enabled: bool = True,
    ):
        """
        Args:
            base_embeddings: Underlying embedding provider
            cache_dir: Directory for cache storage
            enabled: Whether caching is enabled
        """
        self.base = base_embeddings
        self.enabled = enabled
        
        if enabled:
            self.cache = EmbeddingCache(
                cache_dir=cache_dir, 
                model_name=base_embeddings.model_name
            )
        else:
            self.cache = None
        
        logger.info(f"CachedEmbeddings: enabled={enabled}")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text with caching."""
        if not self.enabled:
            return self.base.embed_text(text)
        
        # Check cache
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Compute and cache
        embedding = self.base.embed_text(text)
        self.cache.set(text, embedding)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch with caching."""
        if not self.enabled:
            return self.base.embed_batch(texts)
        
        # Check cache for each text
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        cache_hits = len(texts) - len(uncached_texts)
        if cache_hits > 0:
            logger.info(f"Embedding cache: {cache_hits}/{len(texts)} hits")
        
        # Compute uncached
        if uncached_texts:
            new_embeddings = self.base.embed_batch(uncached_texts)
            
            # Store results and cache
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                results[idx] = embedding
                self.cache.set(text, embedding)
        
        return results
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.base.get_dimensions()
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.base.model_name
    
    def health_check(self) -> bool:
        """Check health."""
        return self.base.health_check()
    
    @property
    def stats(self):
        """Get cache stats."""
        if self.cache:
            return self.cache.stats
        return {"enabled": False}
