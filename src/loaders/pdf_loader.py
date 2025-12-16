"""PDF file loader."""

import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.loaders.base import BaseLoader, Document
from src.loaders.exceptions import LoaderError

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """
    Loads PDF files (.pdf).
    
    Usage:
        loader = PDFLoader()
        docs = loader.load(Path("document.pdf"))
    """

    def __init__(self, pages_as_documents: bool = False):
        """
        Args:
            pages_as_documents: If True, return each page as separate Document.
                               If False, combine all pages into one Document.
        """
        self.pages_as_documents = pages_as_documents

    def load(self, file_path: Path) -> List[Document]:
        """
        Load PDF file.
        
        Args:
            file_path: Path to .pdf file
            
        Returns:
            List of Documents (one per page or one combined)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.can_load(file_path):
            raise LoaderError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            raise LoaderError(f"Failed to read PDF {file_path}: {e}")
        
        documents = []
        
        if self.pages_as_documents:
            # Each page as separate document
            for page_num, page in enumerate(reader.pages, start=1):
                content = page.extract_text() or ""
                documents.append(Document(
                    content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "filetype": "pdf",
                        "page": page_num,
                        "total_pages": len(reader.pages)
                    }
                ))
        else:
            # Combine all pages
            all_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                all_text.append(text)
            
            content = "\n\n".join(all_text)
            documents.append(Document(
                content=content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "filetype": "pdf",
                    "total_pages": len(reader.pages)
                }
            ))
        
        logger.info(f"Loaded {len(reader.pages)} pages from {file_path}")
        
        return documents

    def supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".pdf"]
