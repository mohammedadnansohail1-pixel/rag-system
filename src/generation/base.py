"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLLM(ABC):
    """
    Abstract base class that all LLM providers must implement.
    
    Ensures consistent interface across:
    - Ollama
    - OpenAI
    - Anthropic
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system instructions
            
        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User question
            context: List of relevant context chunks
            system_prompt: Optional system instructions
            
        Returns:
            Generated text response
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if LLM is accessible.
        
        Returns:
            True if healthy
        """
        pass
