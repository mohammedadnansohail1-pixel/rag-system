"""
Reranking module for improving retrieval precision.

Usage:
    from src.reranking import RerankerFactory, RerankResult
    
    # Create from config
    reranker = RerankerFactory.from_config({
        "type": "cross_encoder",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    
    # Rerank documents
    results = reranker.rerank(
        query="what is kafka?",
        documents=["Kafka is a streaming platform...", "Redis is a cache..."],
        top_n=5
    )
"""
from src.reranking.base import BaseReranker, RerankResult
from src.reranking.factory import (
    RerankerFactory,
    register_reranker,
    get_registered_rerankers,
)

# Import implementations to trigger registration
from src.reranking import cross_encoder

__all__ = [
    "BaseReranker",
    "RerankResult",
    "RerankerFactory",
    "register_reranker",
    "get_registered_rerankers",
]

# BGE reranker
from src.reranking import bge_reranker
