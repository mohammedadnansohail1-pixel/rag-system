"""Hierarchical retriever using section summaries."""

import logging
from typing import List, Optional, Dict, Any, Set

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.chunkers.base import Chunk

logger = logging.getLogger(__name__)


class HierarchicalRetriever(BaseRetriever):
    """
    Two-stage retriever: summaries first, then detail chunks.
    
    Strategy:
    1. Query retrieves both summaries and detail chunks
    2. For overview queries: prioritize summaries
    3. For detail queries: use summaries to identify sections, then get detail
    4. Always return mix of summary + detail for context
    
    Benefits:
    - Better for "summarize" and "overview" queries
    - Provides section context alongside details
    - Enables drill-down from high-level to specific
    
    Usage:
        retriever = HierarchicalRetriever(
            base_retriever=hybrid_retriever,
            summary_boost=1.5,
        )
        
        # Ingest with summaries
        retriever.add_documents(texts, metadatas)  # Include summary chunks
        
        # Query
        results = retriever.retrieve("Summarize the risk factors")
        # Returns: summaries first, then supporting details
    """
    
    # Query patterns that prefer summaries
    SUMMARY_PATTERNS = [
        "summarize", "summary", "overview", "main points",
        "key points", "highlights", "brief", "high-level",
        "what are the main", "give me an overview",
    ]
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        summary_boost: float = 1.3,
        detail_boost: float = 1.0,
        max_summaries: int = 2,
        include_details_with_summaries: bool = True,
    ):
        """
        Args:
            base_retriever: Underlying retriever
            summary_boost: Score multiplier for summary chunks
            detail_boost: Score multiplier for detail chunks
            max_summaries: Max summary chunks to return
            include_details_with_summaries: Also return detail chunks
        """
        self.base_retriever = base_retriever
        self.summary_boost = summary_boost
        self.detail_boost = detail_boost
        self.max_summaries = max_summaries
        self.include_details_with_summaries = include_details_with_summaries
        
        # Track summary sections for drill-down
        self._summary_sections: Set[str] = set()
        
        logger.info(
            f"Initialized HierarchicalRetriever: "
            f"summary_boost={summary_boost}, max_summaries={max_summaries}"
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """Add documents and track which sections have summaries."""
        self.base_retriever.add_documents(texts=texts, metadatas=metadatas, **kwargs)
        
        # Track sections with summaries
        if metadatas:
            for meta in metadatas:
                if meta.get("is_summary") or meta.get("chunk_type") == "summary":
                    section = meta.get("section")
                    if section:
                        self._summary_sections.add(section)
        
        logger.info(f"Sections with summaries: {len(self._summary_sections)}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        prefer_summaries: Optional[bool] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Retrieve with hierarchical strategy.
        
        Args:
            query: Search query
            top_k: Number of results
            prefer_summaries: Force summary preference (auto-detected if None)
            
        Returns:
            List of results (summaries + details)
        """
        # Auto-detect if query prefers summaries
        if prefer_summaries is None:
            prefer_summaries = self._is_summary_query(query)
        
        # Get more results than needed for filtering
        initial_k = top_k * 3
        results = self.base_retriever.retrieve(query, top_k=initial_k, **kwargs)
        
        # Separate summaries and details
        summaries = []
        details = []
        
        for r in results:
            is_summary = (
                r.metadata.get("is_summary") or 
                r.metadata.get("chunk_type") == "summary"
            )
            
            if is_summary:
                # Boost summary scores
                boosted = RetrievalResult(
                    content=r.content,
                    metadata={**r.metadata, "retrieval_type": "summary"},
                    score=r.score * self.summary_boost,
                )
                summaries.append(boosted)
            else:
                # Apply detail boost
                boosted = RetrievalResult(
                    content=r.content,
                    metadata={**r.metadata, "retrieval_type": "detail"},
                    score=r.score * self.detail_boost,
                )
                details.append(boosted)
        
        # Combine based on query type
        if prefer_summaries:
            combined = self._prioritize_summaries(summaries, details, top_k)
        else:
            combined = self._balanced_results(summaries, details, top_k)
        
        # Sort by score
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined[:top_k]
    
    def _is_summary_query(self, query: str) -> bool:
        """Detect if query is asking for summary/overview."""
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.SUMMARY_PATTERNS)
    
    def _prioritize_summaries(
        self,
        summaries: List[RetrievalResult],
        details: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Return summaries first, then supporting details."""
        result = []
        
        # Add top summaries
        result.extend(summaries[:self.max_summaries])
        
        # Add details if enabled
        if self.include_details_with_summaries:
            # Get sections covered by summaries
            summary_sections = {
                r.metadata.get("section") for r in result
            }
            
            # Prefer details from same sections
            same_section = [
                d for d in details
                if d.metadata.get("section") in summary_sections
            ]
            other_section = [
                d for d in details
                if d.metadata.get("section") not in summary_sections
            ]
            
            remaining = top_k - len(result)
            result.extend(same_section[:remaining])
            
            if len(result) < top_k:
                remaining = top_k - len(result)
                result.extend(other_section[:remaining])
        
        return result
    
    def _balanced_results(
        self,
        summaries: List[RetrievalResult],
        details: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Return balanced mix of summaries and details."""
        result = []
        
        # Include 1 summary if available and relevant
        if summaries:
            result.append(summaries[0])
        
        # Fill rest with details
        remaining = top_k - len(result)
        result.extend(details[:remaining])
        
        return result
    
    def get_section_summary(self, section_name: str) -> Optional[RetrievalResult]:
        """Get summary for a specific section."""
        # Search for section summary directly
        results = self.base_retriever.retrieve(
            f"[SECTION SUMMARY: {section_name}]",
            top_k=5,
        )
        
        for r in results:
            if (r.metadata.get("is_summary") and 
                r.metadata.get("section") == section_name):
                return r
        
        return None
    
    def get_section_details(
        self,
        section_name: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Get detail chunks for a specific section."""
        # This is a simple filter - could be enhanced with vector search
        results = self.base_retriever.retrieve(section_name, top_k=top_k * 2)
        
        filtered = [
            r for r in results
            if r.metadata.get("section") == section_name
            and not r.metadata.get("is_summary")
        ]
        
        return filtered[:top_k]
    
    @property
    def sections_with_summaries(self) -> Set[str]:
        """Get sections that have summaries."""
        return self._summary_sections
    
    def health_check(self) -> bool:
        """Check health of underlying retriever."""
        return self.base_retriever.health_check()
