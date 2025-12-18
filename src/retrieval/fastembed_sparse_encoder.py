"""FastEmbed-based sparse encoder - stateless, no vocabulary fitting needed."""
import logging
from typing import List, Optional
from fastembed import SparseTextEmbedding
from src.core.types import SparseVector

logger = logging.getLogger(__name__)


class FastEmbedSparseEncoder:
    """
    Sparse encoder using FastEmbed's BM25 model.
    
    Key advantages over rank_bm25:
    - Hash-based: No vocabulary fitting required
    - Stateless: Works across sessions without persistence
    - Qdrant-optimized: Designed for Modifier.IDF
    
    Usage:
        encoder = FastEmbedSparseEncoder()
        vectors = encoder.encode_documents(["doc1", "doc2"])
        query_vec = encoder.encode_query("search query")
    """
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Initialize FastEmbed sparse encoder.
        
        Args:
            model_name: Model to use. Options:
                - "Qdrant/bm25" (default, hash-based BM25)
                - "prithivida/Splade_PP_en_v1" (neural SPLADE)
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"Initialized FastEmbedSparseEncoder with {model_name}")
    
    @property
    def model(self) -> SparseTextEmbedding:
        """Lazy load model on first use."""
        if self._model is None:
            logger.info(f"Loading FastEmbed sparse model: {self.model_name}")
            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model
    
    def encode_documents(self, texts: List[str]) -> List[SparseVector]:
        """
        Encode documents to sparse vectors.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of SparseVector objects
        """
        if not texts:
            return []
        
        embeddings = list(self.model.embed(texts))
        
        vectors = []
        for emb in embeddings:
            vectors.append(SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist()
            ))
        
        return vectors
    
    def encode_query(self, query: str) -> SparseVector:
        """
        Encode a single query to sparse vector.
        
        Args:
            query: Query text
            
        Returns:
            SparseVector object
        """
        embeddings = list(self.model.embed([query]))
        emb = embeddings[0]
        
        return SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist()
        )
    
    def encode(self, text: str) -> SparseVector:
        """Alias for encode_query for compatibility."""
        return self.encode_query(text)

    def encode_batch(self, texts: List[str]) -> List[SparseVector]:
        """Alias for encode_documents for compatibility."""
        return self.encode_documents(texts)

    def fit(self, texts: List[str]) -> None:
        """No-op for compatibility. FastEmbed doesn't need fitting."""
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Always True - FastEmbed doesn't need fitting."""
        return True
    
    def health_check(self) -> bool:
        """Check if encoder is working."""
        try:
            _ = self.model
            return True
        except Exception:
            return False
