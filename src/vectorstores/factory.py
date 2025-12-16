"""Factory for creating vector stores with registry pattern."""

import logging
from typing import Dict, Type, List, Callable

from src.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)

# Registry to hold vector store classes
_VECTORSTORE_REGISTRY: Dict[str, Type[BaseVectorStore]] = {}


def register_vectorstore(name: str) -> Callable:
    """
    Decorator to register a vector store class.
    
    Usage:
        @register_vectorstore("qdrant")
        class QdrantVectorStore(BaseVectorStore):
            ...
    """
    def decorator(cls: Type[BaseVectorStore]) -> Type[BaseVectorStore]:
        if name in _VECTORSTORE_REGISTRY:
            logger.warning(f"Overwriting existing vector store: {name}")
        _VECTORSTORE_REGISTRY[name] = cls
        logger.debug(f"Registered vector store: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_registered_vectorstores() -> List[str]:
    """Return list of registered vector store names."""
    return list(_VECTORSTORE_REGISTRY.keys())


class VectorStoreFactory:
    """
    Factory that creates vector stores based on config.
    
    Usage:
        # From config dict
        store = VectorStoreFactory.from_config({
            "provider": "qdrant",
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "documents"
            }
        })
        
        # Or directly
        store = VectorStoreFactory.create("qdrant", host="localhost", ...)
    """

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseVectorStore:
        """
        Create a vector store instance.
        
        Args:
            provider: Provider name ('qdrant', 'chroma', etc.)
            **kwargs: Provider-specific configuration
            
        Returns:
            VectorStore instance
            
        Raises:
            ValueError: If provider is unknown
        """
        if provider not in _VECTORSTORE_REGISTRY:
            available = get_registered_vectorstores()
            raise ValueError(
                f"Unknown vector store provider: '{provider}'. "
                f"Available: {available}"
            )
        
        store_class = _VECTORSTORE_REGISTRY[provider]
        logger.info(f"Creating vector store: {provider}")
        
        return store_class(**kwargs)

    @classmethod
    def from_config(cls, config: dict) -> BaseVectorStore:
        """
        Create vector store from config dict.
        
        Args:
            config: Config dict with provider and provider-specific settings
            
        Returns:
            VectorStore instance
        """
        provider = config.get("provider", "qdrant")
        provider_config = config.get(provider, {})
        
        logger.debug(f"Creating vector store from config: provider={provider}")
        
        return cls.create(provider, **provider_config)
