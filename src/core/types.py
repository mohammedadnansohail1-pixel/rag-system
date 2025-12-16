"""Shared types used across modules."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SparseVector:
    """
    Sparse vector representation.
    
    Attributes:
        indices: Token IDs with non-zero weights
        values: Weight for each token
    """
    indices: List[int]
    values: List[float]
    
    def to_dict(self) -> Dict[int, float]:
        """Convert to {token_id: weight} dict."""
        return dict(zip(self.indices, self.values))
    
    def __repr__(self) -> str:
        return f"SparseVector(nnz={len(self.indices)})"
