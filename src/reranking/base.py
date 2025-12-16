"""Abstract base class for rerankers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class RerankResult:
    """
    Represents a reranked document.
    
    Attributes:
        content: The text content
        metadata: Document metadata (source, page, etc.)
        score: Reranker relevance score (higher is better)
        original_rank: Position before reranking (1-indexed)
        new_rank: Position after reranking (1-indexed)
    """
    content: str
    metadata: Dict[str, Any]
    score: float
    original_rank: int
    new_rank: int
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"RerankResult(score={self.score:.3f}, "
            f"rank={self.original_rank}→{self.new_rank}, "
            f"content='{preview}')"
        )
    
    @property
    def rank_change(self) -> int:
        """How many positions the document moved (positive = improved)."""
        return self.original_rank - self.new_rank


class BaseReranker(ABC):
    """
    Abstract base class that all rerankers must implement.
    
    Ensures consistent interface across:
    - Cross-encoder reranker (sentence-transformers)
    - BGE reranker (BAAI/bge-reranker)
    - LLM-based reranker (Ollama)
    - Ensemble reranker (combine multiple)
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query text
            documents: List of document texts to rerank
            metadatas: Optional metadata for each document
            top_n: Return only top N results (None = return all)
            
        Returns:
            List of RerankResult objects sorted by relevance (best first)
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if reranker is operational.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def _build_results(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        scores: List[float],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Helper to build RerankResult objects from scores.
        
        Args:
            documents: Original document texts
            metadatas: Optional metadata list
            scores: Relevance scores from reranker
            top_n: Limit results
            
        Returns:
            Sorted list of RerankResult objects
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Pair documents with scores and original rank
        scored = [
            (doc, meta, score, idx + 1)
            for idx, (doc, meta, score) in enumerate(zip(documents, metadatas, scores))
        ]
        
        # Sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)
        
        # Build results with new rank
        results = [
            RerankResult(
                content=doc,
                metadata=meta,
                score=score,
                original_rank=orig_rank,
                new_rank=new_rank + 1,
            )
            for new_rank, (doc, meta, score, orig_rank) in enumerate(scored)
        ]
        
        if top_n is not None:
            results = results[:top_n]
        
        return results
