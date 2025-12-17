"""Factory for creating query expanders."""
import logging
from typing import Dict, Any, Optional, Union

from src.retrieval.query_expansion.base import BaseQueryExpander
from src.retrieval.query_expansion.llm_expander import LLMQueryExpander
from src.retrieval.query_expansion.hyde import HyDEExpander
from src.retrieval.query_expansion.synonym import SynonymExpander
from src.generation.base import BaseLLM

logger = logging.getLogger(__name__)


class QueryExpanderFactory:
    """Factory for creating query expanders."""
    
    @classmethod
    def create(
        cls,
        method: str,
        llm: Optional[BaseLLM] = None,
        **kwargs
    ) -> BaseQueryExpander:
        """
        Create a query expander.
        
        Args:
            method: Expansion method ('llm', 'hyde', 'synonym')
            llm: Language model (required for 'llm' and 'hyde')
            **kwargs: Method-specific arguments
            
        Returns:
            Configured query expander
            
        Example:
            # Synonym expansion (no LLM)
            expander = QueryExpanderFactory.create('synonym')
            
            # LLM expansion
            expander = QueryExpanderFactory.create('llm', llm=my_llm)
            
            # HyDE for financial docs
            expander = QueryExpanderFactory.create('hyde', llm=my_llm, domain='financial')
        """
        method = method.lower()
        
        if method == "synonym":
            return SynonymExpander(**kwargs)
        
        elif method == "llm":
            if llm is None:
                raise ValueError("LLM required for 'llm' expansion method")
            return LLMQueryExpander(llm=llm, **kwargs)
        
        elif method == "hyde":
            if llm is None:
                raise ValueError("LLM required for 'hyde' expansion method")
            return HyDEExpander(llm=llm, **kwargs)
        
        else:
            raise ValueError(f"Unknown expansion method: {method}. Use 'synonym', 'llm', or 'hyde'")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], llm: Optional[BaseLLM] = None) -> BaseQueryExpander:
        """
        Create expander from config dict.
        
        Example config:
            {"method": "hyde", "domain": "financial"}
            {"method": "synonym", "include_financial": True}
        """
        config = config.copy()
        method = config.pop("method", "synonym")
        return cls.create(method, llm=llm, **config)
