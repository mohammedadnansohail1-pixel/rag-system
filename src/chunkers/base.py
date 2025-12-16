"""Abstract base class for text chunkers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.loaders.base import Document


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document.
    
    Attributes:
        content: The text content of the chunk
        metadata: Inherited from parent document + chunk-specific info
    """
    content: str
    metadata: dict

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(content='{preview}', metadata={self.metadata})"


class BaseChunker(ABC):
    """
    Abstract base class that all chunkers must implement.
    
    Ensures consistent interface across:
    - Fixed size chunker
    - Recursive chunker
    - Semantic chunker
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
