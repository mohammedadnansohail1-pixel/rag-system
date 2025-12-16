"""Abstract base class for document loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Document:
    """
    Represents a loaded document.
    
    Attributes:
        content: The text content of the document
        metadata: Additional info (source path, page number, etc.)
    """
    content: str
    metadata: dict

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


class BaseLoader(ABC):
    """
    Abstract base class that all document loaders must implement.
    
    Ensures consistent interface across:
    - PDF loader
    - Markdown loader
    - Text loader
    - HTML loader
    """

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """
        Load document(s) from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            LoaderError: If file cannot be parsed
        """
        pass

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Return list of supported file extensions.
        
        Returns:
            List of extensions (e.g., ['.pdf'], ['.md', '.markdown'])
        """
        pass

    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions()
