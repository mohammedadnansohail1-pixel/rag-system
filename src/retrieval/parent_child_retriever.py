"""Parent-child retriever for expanded context retrieval."""

import logging
from typing import List, Optional, Dict, Any, Set
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.factory import register_retriever

logger = logging.getLogger(__name__)


@register_retriever("parent_child")
class ParentChildRetriever(BaseRetriever):
    """
    Retriever that fetches parent chunks for expanded context.
    
    Strategy:
    1. Retrieve child chunks using base retriever
    2. Look up parent chunks for retrieved children
    3. Return children + parents for fuller context
    
    Benefits:
    - Child chunks are specific (good for matching)
    - Parent chunks provide broader context (good for answering)
    - Combines precision with context
    
    Usage:
        # Wrap any retriever
        base_retriever = HybridRetriever(...)
        pc_retriever = ParentChildRetriever(
            base_retriever=base_retriever,
            include_parents=True,
            parent_weight=0.8,
        )
        
        results = pc_retriever.retrieve("What is Meta's revenue?", top_k=5)
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        include_parents: bool = True,
        parent_weight: float = 0.9,
        replace_children_with_parents: bool = True,
        deduplicate: bool = True,
    ):
        """
        Args:
            base_retriever: Underlying retriever for initial search
            include_parents: Fetch parent chunks for retrieved children
            parent_weight: Score weight for parent chunks (0-1)
            replace_children_with_parents: Replace child with parent (vs adding both)
            deduplicate: Remove duplicate content (by chunk_id)
        """
        self.base_retriever = base_retriever
        self.include_parents = include_parents
        self.parent_weight = parent_weight
        self.replace_children_with_parents = replace_children_with_parents
        self.deduplicate = deduplicate
        
        # Cache for parent chunks (parent_id -> parent content)
        self._parent_cache: Dict[str, RetrievalResult] = {}
        
        logger.info(
            f"Initialized ParentChildRetriever: "
            f"include_parents={include_parents}, parent_weight={parent_weight}, "
            f"replace_children={replace_children_with_parents}"
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """Add documents via base retriever and cache parents."""
        self.base_retriever.add_documents(texts=texts, metadatas=metadatas, **kwargs)
        
        # Cache parent chunks for quick lookup
        if metadatas:
            for i, meta in enumerate(metadatas):
                if meta.get("is_parent") or meta.get("chunk_type") == "parent":
                    chunk_id = meta.get("chunk_id")
                    if chunk_id:
                        self._parent_cache[chunk_id] = RetrievalResult(
                            content=texts[i],
                            metadata=meta,
                            score=1.0,
                        )
        
        logger.info(f"Cached {len(self._parent_cache)} parent chunks")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Retrieve with parent context expansion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResults (children expanded to parents)
        """
        # Step 1: Get initial results from base retriever
        initial_k = top_k * 2 if self.include_parents else top_k
        initial_results = self.base_retriever.retrieve(query, top_k=initial_k, **kwargs)
        
        if not self.include_parents:
            return initial_results[:top_k]
        
        # Step 2: Expand/replace with parent chunks
        expanded_results = self._expand_with_parents(initial_results)
        
        # Step 3: Deduplicate by chunk_id
        if self.deduplicate:
            expanded_results = self._deduplicate_by_id(expanded_results)
        
        # Step 4: Sort by score and return top_k
        expanded_results.sort(key=lambda x: x.score, reverse=True)
        
        return expanded_results[:top_k]
    
    def _expand_with_parents(
        self, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Replace children with parents or add parents."""
        expanded = []
        seen_parents: Set[str] = set()
        
        for result in results:
            parent_id = result.metadata.get("parent_id")
            chunk_id = result.metadata.get("chunk_id", "")
            
            if parent_id and parent_id in self._parent_cache:
                # This is a child chunk - get parent
                parent = self._parent_cache[parent_id]
                
                if self.replace_children_with_parents:
                    # Replace child with parent (avoid duplicate parents)
                    if parent_id not in seen_parents:
                        seen_parents.add(parent_id)
                        parent_result = RetrievalResult(
                            content=parent.content,
                            metadata={
                                **parent.metadata,
                                "retrieved_as": "parent_of_match",
                                "child_chunk_id": chunk_id,
                                "original_child_score": result.score,
                            },
                            score=result.score * self.parent_weight,
                        )
                        expanded.append(parent_result)
                        logger.debug(f"Replaced child {chunk_id} with parent {parent_id}")
                else:
                    # Keep both child and parent
                    expanded.append(result)
                    if parent_id not in seen_parents:
                        seen_parents.add(parent_id)
                        parent_result = RetrievalResult(
                            content=parent.content,
                            metadata={
                                **parent.metadata,
                                "retrieved_as": "parent_of_match",
                                "child_chunk_id": chunk_id,
                            },
                            score=result.score * self.parent_weight,
                        )
                        expanded.append(parent_result)
            else:
                # Regular chunk (no parent) - keep as is
                expanded.append(result)
        
        return expanded
    
    def _deduplicate_by_id(
        self, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Remove duplicates by chunk_id."""
        seen_ids: Set[str] = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.metadata.get("chunk_id", "")
            
            # Use content hash if no chunk_id
            if not chunk_id:
                chunk_id = str(hash(result.content[:200]))
            
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def get_parent(self, parent_id: str) -> Optional[RetrievalResult]:
        """Get a specific parent chunk by ID."""
        return self._parent_cache.get(parent_id)
    
    def get_siblings(
        self, 
        parent_id: str, 
        all_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Get all children of a parent from results."""
        return [r for r in all_results if r.metadata.get("parent_id") == parent_id]
    
    def health_check(self) -> bool:
        """Check health of underlying retriever."""
        return self.base_retriever.health_check()
    
    @property
    def parent_count(self) -> int:
        """Number of cached parent chunks."""
        return len(self._parent_cache)
