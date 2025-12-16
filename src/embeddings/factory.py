"""Factory for creating embedding providers with registry pattern."""

import logging
from typing import Dict, Type, List, Callable, Any

from src.embeddings.base import BaseEmbeddings

logger = logging.getLogger(__name__)

# Registry to hold embedding provider classes
_EMBEDDINGS_REGISTRY: Dict[str, Type[BaseEmbeddings]] = {}


def register_embeddings(name: str) -> Callable:
    """
    Decorator to register an embedding provider class.
    
    Usage:
        @register_embeddings("ollama")
        class OllamaEmbeddings(BaseEmbeddings):
            ...
    """
    def decorator(cls: Type[BaseEmbeddings]) -> Type[BaseEmbeddings]:
        if name in _EMBEDDINGS_REGISTRY:
            logger.warning(f"Overwriting existing embeddings provider: {name}")
        _EMBEDDINGS_REGISTRY[name] = cls
        logger.debug(f"Registered embeddings: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_registered_embeddings() -> List[str]:
    """Return list of registered embedding provider names."""
    return list(_EMBEDDINGS_REGISTRY.keys())


class EmbeddingsFactory:
    """
    Factory that creates embedding providers based on config.
    
    Usage:
        # From config dict
        embeddings = EmbeddingsFactory.from_config({
            "provider": "ollama",
            "ollama": {
                "host": "http://localhost:11434",
                "model": "nomic-embed-text"
            }
        })
        
        # Or directly
        embeddings = EmbeddingsFactory.create("ollama", host="...", model="...")
    """

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseEmbeddings:
        """
        Create an embedding provider instance.
        
        Args:
            provider: Provider name ('ollama', 'openai', etc.)
            **kwargs: Provider-specific configuration
            
        Returns:
            Embeddings instance
            
        Raises:
            ValueError: If provider is unknown
        """
        if provider not in _EMBEDDINGS_REGISTRY:
            available = get_registered_embeddings()
            raise ValueError(
                f"Unknown embeddings provider: '{provider}'. "
                f"Available: {available}"
            )
        
        embeddings_class = _EMBEDDINGS_REGISTRY[provider]
        logger.info(f"Creating embeddings provider: {provider}")
        
        return embeddings_class(**kwargs)

    @classmethod
    def from_config(cls, config: dict) -> BaseEmbeddings:
        """
        Create embeddings from config dict.
        
        Args:
            config: Config dict with provider and provider-specific settings
            
        Returns:
            Embeddings instance
        """
        provider = config.get("provider", "ollama")
        provider_config = config.get(provider, {})
        
        logger.debug(f"Creating embeddings from config: provider={provider}")
        
        return cls.create(provider, **provider_config)
