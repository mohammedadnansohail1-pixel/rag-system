"""Base class for query expanders."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    expanded: str
    variations: List[str]
    metadata: dict
    
    def __repr__(self):
        return f"ExpandedQuery(original='{self.original[:30]}...', variations={len(self.variations)})"


class BaseQueryExpander(ABC):
    """Abstract base class for query expansion."""
    
    @abstractmethod
    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query to improve retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            ExpandedQuery with variations
        """
        pass
    
    def expand_batch(self, queries: List[str]) -> List[ExpandedQuery]:
        """Expand multiple queries."""
        return [self.expand(q) for q in queries]
