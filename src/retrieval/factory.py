"""Factory for creating retrievers with registry pattern."""

import logging
from typing import Dict, Type, List, Callable

from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

# Registry to hold retriever classes
_RETRIEVER_REGISTRY: Dict[str, Type[BaseRetriever]] = {}


def register_retriever(name: str) -> Callable:
    """
    Decorator to register a retriever class.
    
    Usage:
        @register_retriever("dense")
        class DenseRetriever(BaseRetriever):
            ...
    """
    def decorator(cls: Type[BaseRetriever]) -> Type[BaseRetriever]:
        if name in _RETRIEVER_REGISTRY:
            logger.warning(f"Overwriting existing retriever: {name}")
        _RETRIEVER_REGISTRY[name] = cls
        logger.debug(f"Registered retriever: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_registered_retrievers() -> List[str]:
    """Return list of registered retriever names."""
    return list(_RETRIEVER_REGISTRY.keys())


class RetrieverFactory:
    """
    Factory that creates retrievers based on config.
    
    Usage:
        # From config dict
        retriever = RetrieverFactory.from_config({
            "search_type": "dense",
            "top_k": 5
        }, embeddings=embeddings, vectorstore=vectorstore)
        
        # Or directly
        retriever = RetrieverFactory.create("dense", embeddings=..., vectorstore=...)
    """

    @classmethod
    def create(cls, search_type: str, **kwargs) -> BaseRetriever:
        """
        Create a retriever instance.
        
        Args:
            search_type: Retriever type ('dense', 'sparse', 'hybrid')
            **kwargs: Retriever-specific configuration
            
        Returns:
            Retriever instance
            
        Raises:
            ValueError: If search_type is unknown
        """
        if search_type not in _RETRIEVER_REGISTRY:
            available = get_registered_retrievers()
            raise ValueError(
                f"Unknown retriever type: '{search_type}'. "
                f"Available: {available}"
            )
        
        retriever_class = _RETRIEVER_REGISTRY[search_type]
        logger.info(f"Creating retriever: {search_type}")
        
        return retriever_class(**kwargs)

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> BaseRetriever:
        """
        Create retriever from config dict.
        
        Args:
            config: Config dict with search_type and settings
            **kwargs: Additional args (embeddings, vectorstore, etc.)
            
        Returns:
            Retriever instance
        """
        search_type = config.get("search_type", "dense")
        top_k = config.get("top_k", 5)
        score_threshold = config.get("score_threshold")
        
        logger.debug(f"Creating retriever from config: search_type={search_type}")
        
        return cls.create(
            search_type,
            top_k=top_k,
            score_threshold=score_threshold,
            **kwargs
        )
