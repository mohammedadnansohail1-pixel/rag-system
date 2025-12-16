"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SearchResult:
    """
    Represents a search result from vector store.
    
    Attributes:
        content: The text content
        metadata: Document/chunk metadata
        score: Similarity score (higher is better)
    """
    content: str
    metadata: Dict[str, Any]
    score: float

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"SearchResult(score={self.score:.3f}, content='{preview}')"


class BaseVectorStore(ABC):
    """
    Abstract base class that all vector stores must implement.
    
    Ensures consistent interface across:
    - Qdrant
    - Chroma
    - Milvus
    """

    @abstractmethod
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts and their embeddings to the store.
        
        Args:
            texts: List of text content
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            ids: Optional list of IDs (auto-generated if not provided)
            
        Returns:
            List of IDs for added documents
        """
        pass

    @abstractmethod
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
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get total number of documents in store.
        
        Returns:
            Document count
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if vector store is accessible.
        
        Returns:
            True if healthy
        """
        pass
