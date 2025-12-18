"""Base class for document structure patterns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


@dataclass
class Section:
    """
    Represents a logical section extracted from a document.
    
    Attributes:
        title: Section title/header
        content: Section text content
        level: Hierarchy level (0 = top level)
        start_pos: Character position where section starts
        end_pos: Character position where section ends
        section_type: Type identifier (e.g., "item_1a", "risk_factors")
        parent_title: Title of parent section if nested
    """
    title: str
    content: str
    level: int = 0
    start_pos: int = 0
    end_pos: int = 0
    section_type: Optional[str] = None
    parent_title: Optional[str] = None
    
    @property
    def hierarchy(self) -> List[str]:
        """Return hierarchy path as list."""
        if self.parent_title:
            return [self.parent_title, self.title]
        return [self.title]


class BaseDocumentPattern(ABC):
    """
    Abstract base class for document structure patterns.
    
    Implement this to add support for new document types:
    - SEC filings (10-K, 10-Q, 8-K)
    - Legal documents
    - Markdown files
    - HTML documents
    - Academic papers
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pattern identifier (e.g., 'sec_10k', 'markdown')."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass
    
    @abstractmethod
    def detect(self, content: str, metadata: Optional[dict] = None) -> float:
        """
        Detect if this pattern matches the document.
        
        Args:
            content: Document text content
            metadata: Optional document metadata
            
        Returns:
            Confidence score 0.0 to 1.0 (0 = no match, 1 = certain match)
        """
        pass
    
    @abstractmethod
    def extract_sections(self, content: str) -> List[Section]:
        """
        Extract logical sections from document.
        
        Args:
            content: Document text content
            
        Returns:
            List of Section objects in document order
        """
        pass
    
    def preprocess(self, content: str) -> str:
        """
        Preprocess content before section extraction.
        Override for format-specific cleaning (e.g., HTML stripping).
        
        Args:
            content: Raw document content
            
        Returns:
            Cleaned content ready for section extraction
        """
        return content
    
    def extract_tables(self, content: str) -> List[Tuple[int, int, str]]:
        """
        Extract table positions from content.
        Override for formats with tables.
        
        Args:
            content: Document content
            
        Returns:
            List of (start_pos, end_pos, table_content) tuples
        """
        return []
    
    def get_fallback_separators(self) -> List[str]:
        """
        Return separators for fallback recursive chunking.
        Override to customize per document type.
        """
        return ["\n\n", "\n", ". ", " "]
