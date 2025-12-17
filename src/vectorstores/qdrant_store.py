"""Qdrant vector store implementation."""

import logging
import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from src.vectorstores.base import BaseVectorStore, SearchResult
from src.vectorstores.factory import register_vectorstore

logger = logging.getLogger(__name__)

DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


@register_vectorstore("qdrant")
class QdrantVectorStore(BaseVectorStore):
    """
    Vector store using Qdrant.
    
    Usage:
        store = QdrantVectorStore(
            host="localhost",
            port=6333,
            collection_name="documents",
            dimensions=768
        )
        store.add(texts, embeddings)
        results = store.search(query_embedding)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
        dimensions: int = 768,
        distance_metric: str = "cosine",
    ):
        """
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of collection to use
            dimensions: Embedding vector dimensions
            distance_metric: Distance metric ('cosine', 'euclid', 'dot')
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        
        self._client = QdrantClient(host=host, port=port)
        self._ensure_collection()
        
        logger.info(
            f"Initialized QdrantVectorStore: "
            f"collection={collection_name}, host={host}:{port}"
        )

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            distance = DISTANCE_MAP.get(self.distance_metric, Distance.COSINE)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=distance
                )
            )
            logger.info(f"Created collection: {self.collection_name}")

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts and embeddings to Qdrant.
        
        Args:
            texts: List of text content
            embeddings: List of embedding vectors
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
        for text, embedding, metadata, doc_id in zip(texts, embeddings, metadatas, ids):
            payload = {
                "content": text,
                "id": doc_id,
                **metadata
            }
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            ))
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} documents to {self.collection_name}")
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            score_threshold: Minimum score filter
            
        Returns:
            List of SearchResult objects
        """
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        ).points
        
        search_results = []
        for result in results:
            payload = dict(result.payload)
            content = payload.pop("content", "")
            search_results.append(SearchResult(
                content=content,
                metadata=payload,
                score=result.score
            ))
        
        logger.debug(f"Found {len(search_results)} results")
        return search_results

    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids: Document IDs to delete
            
        Returns:
            True if successful
        """
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
        
        logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")
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
            logger.error(f"Qdrant health check failed: {e}")
            return False
