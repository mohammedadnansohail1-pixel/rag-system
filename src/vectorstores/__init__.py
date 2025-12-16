"""Vector stores module - pluggable vector storage."""

# Import base and factory first (defines registry and decorator)
from src.vectorstores.base import BaseVectorStore, SearchResult
from src.vectorstores.factory import (
    VectorStoreFactory,
    register_vectorstore,
    get_registered_vectorstores,
)

# Import stores to trigger registration
from src.vectorstores.qdrant_store import QdrantVectorStore

__all__ = [
    "BaseVectorStore",
    "SearchResult",
    "VectorStoreFactory",
    "register_vectorstore",
    "get_registered_vectorstores",
    "QdrantVectorStore",
]
