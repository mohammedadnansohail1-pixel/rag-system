"""Hybrid retriever combining dense and sparse search."""
import logging
from typing import List, Optional, Dict, Any

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.factory import register_retriever
from src.retrieval.sparse_encoder import SpladeEncoder
from src.core.types import SparseVector
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore

logger = logging.getLogger(__name__)


@register_retriever("hybrid")
class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense (semantic) and sparse (lexical) search.
    
    Uses:
    - Dense embeddings (Ollama/sentence-transformers) for semantic similarity
    - SPLADE sparse vectors for keyword/lexical matching
    - RRF (Reciprocal Rank Fusion) to combine results
    
    Benefits over dense-only:
    - Better recall for keyword-heavy queries
    - Handles rare terms that embeddings miss
    - More robust across query types
    
    Usage:
        retriever = HybridRetriever(
            embeddings=ollama_embeddings,
            vectorstore=qdrant_hybrid_store,
        )
        
        # Add documents (encodes both dense and sparse)
        retriever.add_documents(texts, metadatas)
        
        # Search with hybrid fusion
        results = retriever.retrieve("kafka message ordering", top_k=10)
    """
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: QdrantHybridStore,
        sparse_encoder: Optional[SpladeEncoder] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embeddings: Dense embedding provider
            vectorstore: Hybrid vector store (must support sparse)
            sparse_encoder: SPLADE encoder (created if not provided)
            top_k: Default number of results
            score_threshold: Minimum score filter
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.sparse_encoder = sparse_encoder or SpladeEncoder()
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        logger.info(
            f"Initialized HybridRetriever: "
            f"top_k={top_k}, score_threshold={score_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query text
            top_k: Number of results (overrides default)
            mode: Search mode ('hybrid', 'dense', 'sparse')
            
        Returns:
            List of RetrievalResult objects
        """
        k = top_k or self.top_k
        
        if mode == "dense":
            return self._dense_search(query, k)
        elif mode == "sparse":
            return self._sparse_search(query, k)
        else:
            return self._hybrid_search(query, k)
    
    def _dense_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Dense-only search."""
        query_embedding = self.embeddings.embed_text(query)
        
        search_results = self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=self.score_threshold,
        )
        
        return self._convert_results(search_results)
    
    def _sparse_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Sparse-only search."""
        sparse_query = self.sparse_encoder.encode(query)
        
        search_results = self.vectorstore.sparse_search(
            sparse_query=sparse_query,
            top_k=top_k,
        )
        
        return self._convert_results(search_results)
    
    def _hybrid_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Hybrid search with RRF fusion."""
        # Encode query with both methods
        dense_query = self.embeddings.embed_text(query)
        sparse_query = self.sparse_encoder.encode(query)
        
        # Search with fusion
        search_results = self.vectorstore.hybrid_search(
            dense_query=dense_query,
            sparse_query=sparse_query,
            top_k=top_k,
        )
        
        logger.debug(f"Hybrid search returned {len(search_results)} results")
        return self._convert_results(search_results)
    
    def _convert_results(self, search_results) -> List[RetrievalResult]:
        """Convert SearchResult to RetrievalResult."""
        return [
            RetrievalResult(
                content=sr.content,
                metadata=sr.metadata,
                score=sr.score,
            )
            for sr in search_results
        ]
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add documents with both dense and sparse encodings.
        
        Args:
            texts: List of text content
            metadatas: Optional metadata for each text
            
        Returns:
            List of document IDs
        """
        logger.info(f"Encoding {len(texts)} documents (dense + sparse)...")
        
        # Generate dense embeddings
        dense_embeddings = self.embeddings.embed_batch(texts)
        
        # Generate sparse vectors
        sparse_vectors = self.sparse_encoder.encode_batch(texts)
        
        # Add to hybrid store
        ids = self.vectorstore.add_hybrid(
            texts=texts,
            dense_embeddings=dense_embeddings,
            sparse_vectors=sparse_vectors,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(ids)} documents with hybrid vectors")
        return ids
    
    def health_check(self) -> bool:
        """Check if retriever is operational."""
        try:
            vs_health = self.vectorstore.health_check()
            sparse_health = self.sparse_encoder.health_check()
            return vs_health and sparse_health
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"HybridRetriever(top_k={self.top_k})"
