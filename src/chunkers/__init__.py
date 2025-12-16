"""Text chunkers module - pluggable document chunking."""

# Import base and factory first (defines registry and decorator)
from src.chunkers.base import BaseChunker, Chunk
from src.chunkers.factory import (
    ChunkerFactory,
    register_chunker,
    get_registered_chunkers,
)

# Import chunkers to trigger registration
from src.chunkers.fixed_chunker import FixedChunker
from src.chunkers.recursive_chunker import RecursiveChunker

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkerFactory",
    "register_chunker",
    "get_registered_chunkers",
    "FixedChunker",
    "RecursiveChunker",
]
