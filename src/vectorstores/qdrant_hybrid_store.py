"""Qdrant vector store with hybrid (dense + sparse) support."""
import logging
import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector as QdrantSparseVector,
    NamedVector,
    NamedSparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)

from src.vectorstores.base import BaseVectorStore, SearchResult
from src.vectorstores.factory import register_vectorstore
from src.core.types import SparseVector

logger = logging.getLogger(__name__)

DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


@register_vectorstore("qdrant_hybrid")
class QdrantHybridStore(BaseVectorStore):
    """
    Qdrant vector store with hybrid dense + sparse search.
    
    Stores both dense embeddings and SPLADE sparse vectors
    in the same collection using named vectors.
    
    Features:
    - Dense search (semantic similarity)
    - Sparse search (lexical/keyword matching)
    - Hybrid search with RRF fusion
    
    Usage:
        store = QdrantHybridStore(
            collection_name="hybrid_docs",
            dense_dimensions=768,
        )
        
        # Add with both vector types
        store.add_hybrid(
            texts=["..."],
            dense_embeddings=[[0.1, 0.2, ...]],
            sparse_vectors=[SparseVector(...)],
        )
        
        # Search with fusion
        results = store.hybrid_search(
            dense_query=[0.1, 0.2, ...],
            sparse_query=SparseVector(...),
            top_k=10
        )
    """
    
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "hybrid_documents",
        dense_dimensions: int = 768,
        distance_metric: str = "cosine",
        recreate_collection: bool = False,
    ):
        """
        Initialize hybrid vector store.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of collection
            dense_dimensions: Dimensions for dense vectors
            distance_metric: Distance metric for dense vectors
            recreate_collection: Drop and recreate collection if exists
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dense_dimensions = dense_dimensions
        self.distance_metric = distance_metric
        
        self._client = QdrantClient(host=host, port=port)
        self._ensure_collection(recreate=recreate_collection)
        
        logger.info(
            f"Initialized QdrantHybridStore: "
            f"collection={collection_name}, host={host}:{port}"
        )
    
    def _ensure_collection(self, recreate: bool = False) -> None:
        """Create collection with both dense and sparse vector configs."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists and recreate:
            self._client.delete_collection(self.collection_name)
            exists = False
            logger.info(f"Deleted existing collection: {self.collection_name}")
        
        if not exists:
            distance = DISTANCE_MAP.get(self.distance_metric, Distance.COSINE)
            
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.DENSE_VECTOR_NAME: VectorParams(
                        size=self.dense_dimensions,
                        distance=distance,
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: SparseVectorParams()
                }
            )
            logger.info(f"Created hybrid collection: {self.collection_name}")
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents with dense embeddings only.
        
        For hybrid, use add_hybrid() instead.
        """
        # Create empty sparse vectors for compatibility
        sparse_vectors = [SparseVector(indices=[], values=[]) for _ in texts]
        return self.add_hybrid(texts, embeddings, sparse_vectors, metadatas, ids)
    
    def add_hybrid(
        self,
        texts: List[str],
        dense_embeddings: List[List[float]],
        sparse_vectors: List[SparseVector],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents with both dense and sparse vectors.
        
        Args:
            texts: List of text content
            dense_embeddings: List of dense embedding vectors
            sparse_vectors: List of SPLADE sparse vectors
            metadatas: Optional metadata for each text
            ids: Optional IDs (auto-generated if not provided)
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        points = []
        for text, dense_emb, sparse_vec, metadata, doc_id in zip(
            texts, dense_embeddings, sparse_vectors, metadatas, ids
        ):
            payload = {"content": text, **metadata}
            
            point = PointStruct(
                id=doc_id,
                vector={
                    self.DENSE_VECTOR_NAME: dense_emb,
                    self.SPARSE_VECTOR_NAME: QdrantSparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    )
                },
                payload=payload,
            )
            points.append(point)
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Added {len(points)} hybrid documents to {self.collection_name}")
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Dense-only search.
        
        For hybrid, use hybrid_search() instead.
        """
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=self.DENSE_VECTOR_NAME,
            limit=top_k,
            score_threshold=score_threshold,
        ).points
        
        return self._points_to_results(results)
    
    def sparse_search(
        self,
        sparse_query: SparseVector,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Sparse-only search using SPLADE vectors.
        
        Args:
            sparse_query: SPLADE sparse vector
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=QdrantSparseVector(
                indices=sparse_query.indices,
                values=sparse_query.values,
            ),
            using=self.SPARSE_VECTOR_NAME,
            limit=top_k,
        ).points
        
        return self._points_to_results(results)
    
    def hybrid_search(
        self,
        dense_query: List[float],
        sparse_query: SparseVector,
        top_k: int = 5,
        dense_weight: float = 0.5,
        query_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse with RRF fusion.
        Args:
            dense_query: Dense embedding vector
            sparse_query: SPLADE sparse vector
            top_k: Number of results
            dense_weight: Weight for dense vs sparse (0.5 = equal)
            query_filter: Optional metadata filter (e.g., {"ticker": "AAPL"})
        Returns:
            List of SearchResult objects (RRF-fused)
        """
        # Build Qdrant filter if provided
        qdrant_filter = None
        if query_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in query_filter.items()
            ]
            qdrant_filter = Filter(must=conditions)
        
        # Use Qdrant's built-in RRF fusion via prefetch
        prefetch_limit = top_k * 4  # Get more candidates for fusion
        results = self._client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using=self.DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=QdrantSparseVector(
                        indices=sparse_query.indices,
                        values=sparse_query.values,
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=qdrant_filter,
            limit=top_k,
        ).points
        logger.debug(f"Hybrid search returned {len(results)} results")
        return self._points_to_results(results)

        
        logger.debug(f"Hybrid search returned {len(results)} results")
        return self._points_to_results(results)
    
    def _points_to_results(self, points) -> List[SearchResult]:
        """Convert Qdrant points to SearchResult objects."""
        results = []
        for point in points:
            payload = dict(point.payload)
            content = payload.pop("content", "")
            results.append(SearchResult(
                content=content,
                metadata=payload,
                score=point.score,
            ))
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )
        logger.info(f"Deleted {len(ids)} documents")
        return True
    
    def count(self) -> int:
        """Get total document count."""
        info = self._client.get_collection(self.collection_name)
        return info.points_count
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self._client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"QdrantHybridStore(collection='{self.collection_name}')"
