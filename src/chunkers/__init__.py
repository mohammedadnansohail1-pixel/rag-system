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
from src.chunkers.structure_aware_chunker import (
    StructureAwareChunker,
    get_available_patterns,
    PATTERN_REGISTRY,
)

# Import patterns
from src.chunkers.patterns.base import BaseDocumentPattern, Section
from src.chunkers.patterns.markdown import MarkdownPattern
from src.chunkers.patterns.sec_filing import SECFilingPattern

__all__ = [
    # Base
    "BaseChunker",
    "Chunk",
    # Factory
    "ChunkerFactory",
    "register_chunker",
    "get_registered_chunkers",
    # Chunkers
    "FixedChunker",
    "RecursiveChunker",
    "StructureAwareChunker",
    # Patterns
    "BaseDocumentPattern",
    "Section",
    "MarkdownPattern",
    "SECFilingPattern",
    "get_available_patterns",
    "PATTERN_REGISTRY",
]
