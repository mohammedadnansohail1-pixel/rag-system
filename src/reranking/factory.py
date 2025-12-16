"""Factory for creating rerankers with registry pattern."""
import logging
from typing import Dict, Type, List, Callable, Any

from src.reranking.base import BaseReranker

logger = logging.getLogger(__name__)

# Registry to hold reranker classes
_RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {}


def register_reranker(name: str) -> Callable:
    """
    Decorator to register a reranker class.
    
    Usage:
        @register_reranker("cross_encoder")
        class CrossEncoderReranker(BaseReranker):
            ...
    """
    def decorator(cls: Type[BaseReranker]) -> Type[BaseReranker]:
        if name in _RERANKER_REGISTRY:
            logger.warning(f"Overwriting existing reranker: {name}")
        _RERANKER_REGISTRY[name] = cls
        logger.debug(f"Registered reranker: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_registered_rerankers() -> List[str]:
    """Return list of registered reranker names."""
    return list(_RERANKER_REGISTRY.keys())


class RerankerFactory:
    """
    Factory that creates rerankers based on config.
    
    Usage:
        # From config dict
        reranker = RerankerFactory.from_config({
            "type": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        })
        
        # Or directly
        reranker = RerankerFactory.create("cross_encoder", model="...")
    """
    
    @classmethod
    def create(cls, reranker_type: str, **kwargs) -> BaseReranker:
        """
        Create a reranker by type name.
        
        Args:
            reranker_type: Registered name (e.g., "cross_encoder", "bge", "llm")
            **kwargs: Arguments passed to reranker constructor
            
        Returns:
            Configured reranker instance
            
        Raises:
            ValueError: If reranker type not registered
        """
        if reranker_type not in _RERANKER_REGISTRY:
            available = get_registered_rerankers()
            raise ValueError(
                f"Unknown reranker type: '{reranker_type}'. "
                f"Available: {available}"
            )
        
        reranker_cls = _RERANKER_REGISTRY[reranker_type]
        logger.info(f"Creating reranker: {reranker_type} with {kwargs}")
        
        return reranker_cls(**kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseReranker:
        """
        Create a reranker from config dictionary.
        
        Args:
            config: Dict with 'type' key and reranker-specific settings
            
        Returns:
            Configured reranker instance
        """
        config = config.copy()
        reranker_type = config.pop("type", None)
        
        if not reranker_type:
            raise ValueError("Config must include 'type' key")
        
        return cls.create(reranker_type, **config)
