"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """
    Abstract base class that all embedding providers must implement.
    
    Ensures consistent interface across:
    - Ollama embeddings
    - OpenAI embeddings
    - HuggingFace embeddings
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Return the embedding dimensions.
        
        Returns:
            Number of dimensions in embedding vector
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass
