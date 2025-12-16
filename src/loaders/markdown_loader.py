"""Markdown file loader."""

import logging
from pathlib import Path
from typing import List

from src.loaders.base import BaseLoader, Document
from src.loaders.exceptions import LoaderError
from src.loaders.factory import register_loader

logger = logging.getLogger(__name__)


@register_loader("markdown")
class MarkdownLoader(BaseLoader):
    """
    Loads Markdown files (.md, .markdown).
    
    Usage:
        loader = MarkdownLoader()
        docs = loader.load(Path("README.md"))
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding

    def load(self, file_path: Path) -> List[Document]:
        """
        Load markdown file.
        
        Args:
            file_path: Path to .md file
            
        Returns:
            List containing single Document
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.can_load(file_path):
            raise LoaderError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as e:
            raise LoaderError(f"Failed to decode {file_path}: {e}")
        
        logger.info(f"Loaded {len(content)} chars from {file_path}")
        
        return [Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "filetype": "markdown"
            }
        )]

    def supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".md", ".markdown"]
