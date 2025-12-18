"""Multi-document management module."""

from src.documents.registry import DocumentRegistry, DocumentInfo
from src.documents.multi_doc_pipeline import MultiDocumentPipeline

__all__ = ["DocumentRegistry", "DocumentInfo", "MultiDocumentPipeline"]
