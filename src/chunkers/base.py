"""Abstract base class for text chunkers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

from src.loaders.base import Document


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document.

    Attributes:
        content: The text content of the chunk
        metadata: Inherited from parent document + chunk-specific info
        chunk_id: Unique identifier for this chunk
        parent_id: Reference to parent chunk (for parent-child retrieval)
        chunk_type: Type of chunk - "content", "section_header", "table", "summary"
        section: Section name (e.g., "Risk Factors", "MD&A")
        section_hierarchy: Full path (e.g., ["Part I", "Item 1A", "Risk Factors"])
    """
    content: str
    metadata: dict
    
    # Structure-aware fields
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    chunk_type: str = "content"
    section: Optional[str] = None
    section_hierarchy: Optional[List[str]] = None

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        section_info = f", section='{self.section}'" if self.section else ""
        return f"Chunk(id='{self.chunk_id}'{section_info}, content='{preview}')"
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "chunk_type": self.chunk_type,
            "section": self.section,
            "section_hierarchy": self.section_hierarchy,
        }


class BaseChunker(ABC):
    """
    Abstract base class that all chunkers must implement.

    Ensures consistent interface across:
    - Fixed size chunker
    - Recursive chunker
    - Structure-aware chunker
    """

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        pass

    def chunk_many(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of Documents

        Returns:
            List of all Chunks from all documents
        """
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
