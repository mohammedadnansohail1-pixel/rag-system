"""Document loaders module - pluggable document loading."""
from src.loaders.base import BaseLoader, Document
from src.loaders.exceptions import LoaderError, UnsupportedFileTypeError
from src.loaders.factory import (
    LoaderFactory,
    register_loader,
    get_registered_loaders,
)

# Import loaders to trigger registration
# Content-aware loaders first (they check file content, not just extension)
from src.loaders.sec_loader import SECLoader
from src.loaders.web_loader import WebLoader, CrawlConfig

# Extension-based loaders
from src.loaders.pdf_loader import PDFLoader
from src.loaders.markdown_loader import MarkdownLoader
from src.loaders.google_drive_loader import GoogleDriveLoader
from src.loaders.text_loader import TextLoader  # Fallback for .txt

__all__ = [
    "BaseLoader",
    "Document",
    "LoaderError",
    "UnsupportedFileTypeError",
    "LoaderFactory",
    "register_loader",
    "get_registered_loaders",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "GoogleDriveLoader",
    "SECLoader",
    "WebLoader",
    "CrawlConfig",
]
