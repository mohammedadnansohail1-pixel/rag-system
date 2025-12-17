"""LLM-based query expansion - generates related terms and variations."""
import logging
from typing import List, Optional

from src.retrieval.query_expansion.base import BaseQueryExpander, ExpandedQuery
from src.generation.base import BaseLLM

logger = logging.getLogger(__name__)


class LLMQueryExpander(BaseQueryExpander):
    """
    Use LLM to expand queries with related terms.
    
    Generates:
    - Synonyms and related terms
    - Alternative phrasings
    - Domain-specific terminology
    
    Great for bridging user language to document language.
    
    Example:
        "What was Netflix revenue?" →
        "Netflix revenue income earnings financial results 2024 streaming"
    """
    
    EXPANSION_PROMPT = '''Given this search query, generate an expanded version that includes:
1. The original query terms
2. Synonyms and related terms
3. Domain-specific terminology
4. Alternative phrasings

Original query: {query}

Respond with ONLY the expanded query (single line, no explanation).
Include 5-10 additional relevant terms.

Expanded query:'''

    VARIATIONS_PROMPT = '''Generate 3 alternative ways to ask this question.
Keep the same intent but use different words.

Original: {query}

Respond with exactly 3 variations, one per line, no numbering:'''

    def __init__(
        self,
        llm: BaseLLM,
        generate_variations: bool = True,
        max_expansion_terms: int = 10,
    ):
        """
        Args:
            llm: Language model for expansion
            generate_variations: Whether to generate query variations
            max_expansion_terms: Max terms to add
        """
        self.llm = llm
        self.generate_variations = generate_variations
        self.max_expansion_terms = max_expansion_terms
        
        logger.info(f"Initialized LLMQueryExpander with {llm.model_name}")
    
    def expand(self, query: str) -> ExpandedQuery:
        """Expand query using LLM."""
        # Generate expanded query with additional terms
        expansion_prompt = self.EXPANSION_PROMPT.format(query=query)
        expanded = self.llm.generate(expansion_prompt).strip()
        
        # Clean up - remove any explanatory text
        if '\n' in expanded:
            expanded = expanded.split('\n')[0]
        
        # Generate variations if requested
        variations = []
        if self.generate_variations:
            var_prompt = self.VARIATIONS_PROMPT.format(query=query)
            var_response = self.llm.generate(var_prompt).strip()
            variations = [v.strip() for v in var_response.split('\n') if v.strip()]
            variations = variations[:3]  # Limit to 3
        
        return ExpandedQuery(
            original=query,
            expanded=expanded,
            variations=variations,
            metadata={"method": "llm", "model": self.llm.model_name}
        )
