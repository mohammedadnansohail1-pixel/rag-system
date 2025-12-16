"""Abstract base class for retrievers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class RetrievalResult:
    """
    Represents a retrieved document chunk.
    
    Attributes:
        content: The text content
        metadata: Document/chunk metadata
        score: Relevance score (higher is better)
    """
    content: str
    metadata: Dict[str, Any]
    score: float

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"RetrievalResult(score={self.score:.3f}, content='{preview}')"


class BaseRetriever(ABC):
    """
    Abstract base class that all retrievers must implement.
    
    Ensures consistent interface across:
    - Dense retriever (embedding-based)
    - Sparse retriever (BM25)
    - Hybrid retriever (dense + sparse)
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        pass

    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add documents to the retriever's index.
        
        Args:
            texts: List of text content
            metadatas: Optional metadata for each text
            
        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if retriever is operational.
        
        Returns:
            True if healthy
        """
        pass
