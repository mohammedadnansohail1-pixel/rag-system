"""LLM-based query expansion for improved retrieval."""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands short/ambiguous queries using LLM.
    
    Research basis:
    - arXiv:2504.05324: Query expansion improves hybrid retrieval 15-30%
    - arXiv:2512.12694: Semantic Query Synthesis for vocabulary mismatch
    """
    
    EXPANSION_PROMPT = """Task: Add keywords to search query. Output ONLY the expanded query, nothing else.

Input: What is Azure?
Output: Azure Microsoft cloud computing platform services

Input: What is iPhone?
Output: iPhone Apple smartphone iOS mobile device

Input: What is RAG?
Output: RAG retrieval augmented generation LLM vector database

Input: {query}
Output:"""
    
    def __init__(
        self,
        llm,
        min_query_words: int = 4,
        enabled: bool = True,
    ):
        self.llm = llm
        self.min_query_words = min_query_words
        self.enabled = enabled
    
    def should_expand(self, query: str) -> bool:
        """Check if query needs expansion."""
        if not self.enabled:
            return False
        
        words = query.lower().split()
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'are', 'was', 'were', 'will', 'can', 'could', 'should', 'about', 'tell', 'me'}
        meaningful = [w for w in words if w.strip('?.,!') not in stop_words]
        
        return len(meaningful) < self.min_query_words
    
    def _clean_response(self, response: str, original_query: str) -> str:
        """Clean LLM response to extract just keywords."""
        # Take first line
        response = response.split('\n')[0].strip()
        
        # Remove common prefixes
        prefixes = ['output:', 'expanded:', 'result:', 'here are', 'here is']
        lower = response.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove quotes
        response = re.sub(r'^["\']|["\']$', '', response).strip()
        
        # If response looks like explanation, return original
        if any(response.lower().startswith(w) for w in ['i ', 'the ', 'it ', 'this ', 'yes', 'no', 'to ']):
            return original_query
        
        # If too short or too long, return original
        if len(response) < 5 or len(response) > 200:
            return original_query
        
        return response
    
    def expand(self, query: str) -> str:
        """Expand query if needed."""
        if not self.should_expand(query):
            return query
        
        try:
            prompt = self.EXPANSION_PROMPT.format(query=query)
            response = self.llm.generate(prompt).strip()
            expanded = self._clean_response(response, query)
            
            if expanded != query:
                logger.info(f"Query expanded: '{query}' → '{expanded}'")
            return expanded
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query
