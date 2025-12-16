"""Embeddings module - pluggable embedding providers."""

# Import base and factory first (defines registry and decorator)
from src.embeddings.base import BaseEmbeddings
from src.embeddings.factory import (
    EmbeddingsFactory,
    register_embeddings,
    get_registered_embeddings,
)

# Import providers to trigger registration
from src.embeddings.ollama_embeddings import OllamaEmbeddings

__all__ = [
    "BaseEmbeddings",
    "EmbeddingsFactory",
    "register_embeddings",
    "get_registered_embeddings",
    "OllamaEmbeddings",
]
