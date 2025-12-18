"""Hybrid retriever combining dense and sparse search."""
import logging
from typing import List, Optional, Dict, Any, Union

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.factory import register_retriever
from src.retrieval.sparse_encoder import (
    BaseSparseEncoder,
    SparseEncoderFactory,
)
from src.core.types import SparseVector
from src.embeddings.base import BaseEmbeddings
from src.retrieval.fastembed_sparse_encoder import FastEmbedSparseEncoder
from src.reranking.base import BaseReranker
from src.retrieval.classification import QueryComplexityClassifier
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore

logger = logging.getLogger(__name__)


@register_retriever("hybrid")
class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense (semantic) and sparse (lexical) search.

    Uses:
    - Dense embeddings (Ollama/sentence-transformers) for semantic similarity
    - Sparse vectors (BM25/SPLADE/TFIDF) for keyword/lexical matching
    - RRF (Reciprocal Rank Fusion) to combine results

    Sparse encoder options:
    - 'bm25': CPU-only, scales infinitely, production standard
    - 'tfidf': CPU-only, lightweight baseline
    - 'splade': GPU, best quality, higher resources

    Usage:
        # With BM25 (default, production-ready)
        retriever = HybridRetriever(
            embeddings=ollama_embeddings,
            vectorstore=qdrant_hybrid_store,
            sparse_encoder="bm25",
        )

        # With SPLADE (if you have GPU resources)
        retriever = HybridRetriever(
            embeddings=ollama_embeddings,
            vectorstore=qdrant_hybrid_store,
            sparse_encoder="splade",
        )

        # From config
        retriever = HybridRetriever(
            embeddings=ollama_embeddings,
            vectorstore=qdrant_hybrid_store,
            sparse_encoder={"type": "bm25", "k1": 1.5, "b": 0.75},
        )
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: QdrantHybridStore,
        sparse_encoder: Union[str, Dict, BaseSparseEncoder, None] = "fastembed",
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embeddings: Dense embedding provider
            vectorstore: Hybrid vector store (must support sparse)
            sparse_encoder: Sparse encoder - can be:
                - str: encoder type ('bm25', 'tfidf', 'splade')
                - dict: config with 'type' and encoder-specific params
                - BaseSparseEncoder: pre-configured encoder instance
                - None: defaults to 'bm25'
            top_k: Default number of results
            score_threshold: Minimum score filter
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.reranker = reranker
        self.query_classifier = QueryComplexityClassifier(use_llm_fallback=True)
        
        # Initialize sparse encoder
        self.sparse_encoder = self._init_sparse_encoder(sparse_encoder)
        self._corpus_fitted = False

        logger.info(
            f"Initialized HybridRetriever: "
            f"sparse={self.sparse_encoder.__class__.__name__}, reranker={reranker is not None}, "
            f"top_k={top_k}"
        )

    def _init_sparse_encoder(
        self,
        encoder: Union[str, Dict, BaseSparseEncoder, None]
    ) -> BaseSparseEncoder:
        """Initialize sparse encoder from various input types."""
        if encoder is None or encoder == "fastembed":
            return FastEmbedSparseEncoder()
        
        if isinstance(encoder, BaseSparseEncoder):
            return encoder
        
        if isinstance(encoder, str):
            return SparseEncoderFactory.create(encoder)
        
        if isinstance(encoder, dict):
            return SparseEncoderFactory.from_config(encoder)
        
        raise ValueError(f"Invalid sparse_encoder type: {type(encoder)}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query text
            top_k: Number of results (overrides default)
            mode: Search mode ('hybrid', 'dense', 'sparse')
            metadata_filter: Metadata filter (e.g., {"ticker": "AAPL"})

        Returns:
            List of RetrievalResult objects
        """
        k = top_k or self.top_k
        
        # Classify query complexity for adaptive reranking
        classification = self.query_classifier.classify_detailed(query)
        should_rerank = (
            self.reranker 
            and use_reranker 
            and classification.should_rerank
        )
        
        # Get more candidates if reranking (multiplier based on complexity)
        search_k = k * classification.candidates_multiplier if should_rerank else k
        
        logger.debug(
            f"Query '{query[:30]}...' classified as {classification.complexity.value} "
            f"(conf={classification.confidence:.2f}), rerank={should_rerank}"
        )
        
        if mode == "dense":
            results = self._dense_search(query, search_k)
        elif mode == "sparse":
            results = self._sparse_search(query, search_k)
        else:
            results = self._hybrid_search(query, search_k, metadata_filter)
        
        # Apply reranker based on classification
        if should_rerank and results:
            results = self._apply_reranker(query, results, k)
        elif len(results) > k:
            results = results[:k]
        
        return results
    
    def _apply_reranker(
        self, 
        query: str, 
        results: List[RetrievalResult], 
        top_k: int
    ) -> List[RetrievalResult]:
        """Apply reranker to results."""
        docs = [r.content for r in results]
        metas = [r.metadata for r in results]
        
        reranked = self.reranker.rerank(query, docs, metas, top_n=top_k)
        
        return [
            RetrievalResult(
                content=r.content,
                score=r.score,
                metadata=r.metadata or {},
            )
            for r in reranked
        ]

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

    def _hybrid_search(
        self, 
        query: str, 
        top_k: int,
        query_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Hybrid search with RRF fusion."""
        # Encode query with both methods
        dense_query = self.embeddings.embed_text(query)
        sparse_query = self.sparse_encoder.encode(query)

        # Search with fusion
        search_results = self.vectorstore.hybrid_search(
            dense_query=dense_query,
            sparse_query=sparse_query,
            top_k=top_k,
            query_filter=query_filter,
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
        fit_sparse: bool = True,
    ) -> List[str]:
        """
        Add documents with both dense and sparse encodings.

        Args:
            texts: List of text content
            metadatas: Optional metadata for each text
            fit_sparse: Whether to fit sparse encoder on corpus (for BM25/TFIDF)

        Returns:
            List of document IDs
        """
        logger.info(f"Encoding {len(texts)} documents (dense + sparse)...")

        # Fit sparse encoder if needed (BM25/TFIDF need corpus statistics)
        if fit_sparse and not self._corpus_fitted:
            logger.info("Fitting sparse encoder on corpus...")
            self.sparse_encoder.fit(texts)
            self._corpus_fitted = True

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

    def fit_sparse(self, corpus: List[str]) -> None:
        """
        Explicitly fit sparse encoder on corpus.
        
        Useful when you want to fit on a larger corpus than
        what you're indexing (e.g., fit on all docs, index subset).
        
        Args:
            corpus: List of documents to fit on
        """
        logger.info(f"Fitting sparse encoder on {len(corpus)} documents...")
        self.sparse_encoder.fit(corpus)
        self._corpus_fitted = True

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
        encoder_name = self.sparse_encoder.__class__.__name__
        return f"HybridRetriever(sparse={encoder_name}, top_k={self.top_k})"
