"""HyDE - Hypothetical Document Embeddings.

Instead of embedding the query, generate a hypothetical answer
and embed that. Often retrieves better because the embedding
is closer to actual document language.

Paper: https://arxiv.org/abs/2212.10496
"""
import logging
from typing import List, Optional

from src.retrieval.query_expansion.base import BaseQueryExpander, ExpandedQuery
from src.generation.base import BaseLLM

logger = logging.getLogger(__name__)


class HyDEExpander(BaseQueryExpander):
    """
    Hypothetical Document Embeddings (HyDE).
    
    Generates a hypothetical document that would answer the query,
    then uses that for retrieval instead of the query itself.
    
    Why it works:
    - Queries are short, documents are long
    - Query embeddings ≠ document embeddings
    - Hypothetical doc is closer to real doc in embedding space
    
    Example:
        Query: "What was Netflix revenue in 2024?"
        
        HyDE generates:
        "Netflix reported total streaming revenue of $39 billion 
        for fiscal year 2024, representing a 16% increase from 
        the prior year. The company's revenue growth was driven 
        by subscriber additions and price increases across all 
        regions..."
        
        This hypothetical answer embeds closer to the actual 10-K text.
    """
    
    HYDE_PROMPT = '''Write a short passage (2-3 sentences) that would answer this question.
Write as if you are quoting from an official document or report.
Use specific, formal language typical of business documents.

Question: {query}

Passage:'''

    HYDE_FINANCIAL_PROMPT = '''Write a short passage (2-3 sentences) that would appear in a company's 
SEC 10-K filing to answer this question. Use formal financial language with specific 
numbers and terminology typical of annual reports.

Question: {query}

Passage from 10-K:'''

    def __init__(
        self,
        llm: BaseLLM,
        domain: str = "general",
        num_hypotheses: int = 1,
    ):
        """
        Args:
            llm: Language model for generating hypothetical docs
            domain: Domain for prompt selection ('general', 'financial', 'technical')
            num_hypotheses: Number of hypothetical docs to generate
        """
        self.llm = llm
        self.domain = domain
        self.num_hypotheses = num_hypotheses
        
        # Select prompt based on domain
        self.prompt_template = {
            "general": self.HYDE_PROMPT,
            "financial": self.HYDE_FINANCIAL_PROMPT,
        }.get(domain, self.HYDE_PROMPT)
        
        logger.info(f"Initialized HyDEExpander: domain={domain}, hypotheses={num_hypotheses}")
    
    def expand(self, query: str) -> ExpandedQuery:
        """Generate hypothetical document for query."""
        hypotheses = []
        
        for _ in range(self.num_hypotheses):
            prompt = self.prompt_template.format(query=query)
            hypothesis = self.llm.generate(prompt).strip()
            
            # Clean up
            if hypothesis.startswith('"') and hypothesis.endswith('"'):
                hypothesis = hypothesis[1:-1]
            
            hypotheses.append(hypothesis)
        
        # The "expanded" query is the hypothetical document
        # This will be embedded instead of the original query
        expanded = hypotheses[0] if hypotheses else query
        
        return ExpandedQuery(
            original=query,
            expanded=expanded,
            variations=hypotheses,
            metadata={
                "method": "hyde",
                "domain": self.domain,
                "num_hypotheses": len(hypotheses),
            }
        )
    
    def get_embedding_text(self, expanded_query: ExpandedQuery) -> str:
        """
        Get text to embed for retrieval.
        
        For HyDE, we embed the hypothetical document, not the query.
        """
        return expanded_query.expanded
