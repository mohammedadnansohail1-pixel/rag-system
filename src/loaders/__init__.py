"""Document loaders module - pluggable document loading."""

from src.loaders.base import BaseLoader, Document
from src.loaders.exceptions import LoaderError, UnsupportedFileTypeError
from src.loaders.text_loader import TextLoader
from src.loaders.markdown_loader import MarkdownLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.factory import LoaderFactory

__all__ = [
    "BaseLoader",
    "Document",
    "LoaderError",
    "UnsupportedFileTypeError",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "LoaderFactory",
]
