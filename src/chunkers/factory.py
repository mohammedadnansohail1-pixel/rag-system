"""Factory for creating chunkers with registry pattern."""

import logging
from typing import Dict, Type, Optional, List, Callable

from src.chunkers.base import BaseChunker

logger = logging.getLogger(__name__)

# Registry to hold chunker classes
_CHUNKER_REGISTRY: Dict[str, Type[BaseChunker]] = {}


def register_chunker(name: str) -> Callable:
    """
    Decorator to register a chunker class.
    
    Usage:
        @register_chunker("my_chunker")
        class MyChunker(BaseChunker):
            ...
    """
    def decorator(cls: Type[BaseChunker]) -> Type[BaseChunker]:
        if name in _CHUNKER_REGISTRY:
            logger.warning(f"Overwriting existing chunker: {name}")
        _CHUNKER_REGISTRY[name] = cls
        logger.debug(f"Registered chunker: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_registered_chunkers() -> List[str]:
    """Return list of registered chunker names."""
    return list(_CHUNKER_REGISTRY.keys())


class ChunkerFactory:
    """
    Factory that creates chunkers based on config.
    
    Usage:
        # From config dict
        chunker = ChunkerFactory.from_config({
            "strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 50
        })
        
        # Or directly
        chunker = ChunkerFactory.create("recursive", chunk_size=512)
    """

    @classmethod
    def create(
        cls,
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        **kwargs
    ) -> BaseChunker:
        """
        Create a chunker instance.
        
        Args:
            strategy: Chunking strategy ('fixed' or 'recursive')
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: Custom separators (recursive only)
            
        Returns:
            Chunker instance
            
        Raises:
            ValueError: If strategy is unknown
        """
        if strategy not in _CHUNKER_REGISTRY:
            available = get_registered_chunkers()
            raise ValueError(
                f"Unknown strategy: '{strategy}'. "
                f"Available: {available}"
            )
        
        chunker_class = _CHUNKER_REGISTRY[strategy]
        logger.info(f"Creating chunker: {strategy} (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        # Build kwargs based on what the chunker accepts
        chunker_kwargs = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        
        if separators and strategy == "recursive":
            chunker_kwargs["separators"] = separators
        
        return chunker_class(**chunker_kwargs)

    @classmethod
    def from_config(cls, config: dict) -> BaseChunker:
        """
        Create chunker from config dict.
        
        Args:
            config: Config dict with strategy, chunk_size, etc.
            
        Returns:
            Chunker instance
        """
        logger.debug(f"Creating chunker from config: {config}")
        
        return cls.create(
            strategy=config.get("strategy", "recursive"),
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 50),
            separators=config.get("separators"),
        )
