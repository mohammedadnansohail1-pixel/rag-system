"""Retrieval module - pluggable document retrieval."""

# Import base and factory first (defines registry and decorator)
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.factory import (
    RetrieverFactory,
    register_retriever,
    get_registered_retrievers,
)

# Import retrievers to trigger registration
from src.retrieval.dense_retriever import DenseRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "RetrieverFactory",
    "register_retriever",
    "get_registered_retrievers",
    "DenseRetriever",
]
