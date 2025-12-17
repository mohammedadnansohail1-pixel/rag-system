"""Lightweight synonym-based query expansion (no LLM needed)."""
import logging
from typing import List, Dict, Set

from src.retrieval.query_expansion.base import BaseQueryExpander, ExpandedQuery

logger = logging.getLogger(__name__)


class SynonymExpander(BaseQueryExpander):
    """
    Rule-based synonym expansion.
    
    Fast, deterministic, no LLM needed.
    Good for domain-specific terminology.
    
    Example:
        "Netflix revenue" → "Netflix revenue income earnings sales"
    """
    
    # Default financial synonyms
    FINANCIAL_SYNONYMS: Dict[str, List[str]] = {
        "revenue": ["income", "sales", "earnings", "proceeds"],
        "profit": ["income", "earnings", "gain", "margin"],
        "loss": ["deficit", "shortfall", "decline"],
        "growth": ["increase", "expansion", "rise", "gain"],
        "debt": ["liabilities", "borrowings", "loans", "obligations"],
        "assets": ["holdings", "resources", "property"],
        "employees": ["staff", "workforce", "personnel", "workers"],
        "customers": ["subscribers", "members", "users", "clients"],
        "cost": ["expense", "expenditure", "spending"],
        "cash": ["funds", "capital", "liquidity"],
        "ceo": ["chief executive", "executive officer", "leader"],
        "company": ["firm", "corporation", "enterprise", "business"],
        "stock": ["shares", "equity", "securities"],
        "quarterly": ["q1", "q2", "q3", "q4", "quarter"],
        "annual": ["yearly", "fiscal year", "fy"],
        "strategy": ["plan", "approach", "initiative"],
        "risk": ["threat", "exposure", "vulnerability"],
        "market": ["industry", "sector", "segment"],
    }
    
    def __init__(
        self,
        custom_synonyms: Dict[str, List[str]] = None,
        include_financial: bool = True,
        max_synonyms_per_term: int = 2,
    ):
        """
        Args:
            custom_synonyms: Additional synonym mappings
            include_financial: Include built-in financial synonyms
            max_synonyms_per_term: Max synonyms to add per matched term
        """
        self.synonyms: Dict[str, List[str]] = {}
        
        if include_financial:
            self.synonyms.update(self.FINANCIAL_SYNONYMS)
        
        if custom_synonyms:
            self.synonyms.update(custom_synonyms)
        
        self.max_synonyms_per_term = max_synonyms_per_term
        
        logger.info(f"Initialized SynonymExpander with {len(self.synonyms)} term mappings")
    
    def expand(self, query: str) -> ExpandedQuery:
        """Expand query with synonyms."""
        query_lower = query.lower()
        words = query_lower.split()
        
        additions: Set[str] = set()
        
        # Check each term for synonyms
        for term, syns in self.synonyms.items():
            if term in query_lower:
                # Add limited synonyms
                for syn in syns[:self.max_synonyms_per_term]:
                    if syn.lower() not in query_lower:
                        additions.add(syn)
        
        # Build expanded query
        if additions:
            expanded = f"{query} {' '.join(sorted(additions))}"
        else:
            expanded = query
        
        return ExpandedQuery(
            original=query,
            expanded=expanded,
            variations=[],
            metadata={
                "method": "synonym",
                "terms_added": list(additions),
            }
        )
