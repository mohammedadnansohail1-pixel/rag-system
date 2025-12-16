"""Ollama embedding provider."""

import logging
from typing import List

import ollama

from src.embeddings.base import BaseEmbeddings
from src.embeddings.factory import register_embeddings

logger = logging.getLogger(__name__)


@register_embeddings("ollama")
class OllamaEmbeddings(BaseEmbeddings):
    """
    Generates embeddings using Ollama.
    
    Usage:
        embeddings = OllamaEmbeddings(
            host="http://localhost:11434",
            model="nomic-embed-text"
        )
        vector = embeddings.embed_text("Hello world")
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
    ):
        """
        Args:
            host: Ollama server URL
            model: Embedding model name
            dimensions: Expected embedding dimensions
        """
        self.host = host
        self.model = model
        self._dimensions = dimensions
        self._client = ollama.Client(host=host)
        
        logger.info(f"Initialized OllamaEmbeddings: model={model}, host={host}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self._client.embeddings(
            model=self.model,
            prompt=text
        )
        
        return response["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def get_dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.model
