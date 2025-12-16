"""Retrieval module - dense, sparse, and hybrid retrieval."""
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_encoder import (
    BaseSparseEncoder,
    BM25Encoder,
    TFIDFEncoder,
    SpladeEncoder,
    SparseEncoderFactory,
)
from src.retrieval.factory import RetrieverFactory

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "DenseRetriever",
    "HybridRetriever",
    "BaseSparseEncoder",
    "BM25Encoder",
    "TFIDFEncoder",
    "SpladeEncoder",
    "SparseEncoderFactory",
    "RetrieverFactory",
]
