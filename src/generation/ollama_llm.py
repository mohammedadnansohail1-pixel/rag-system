"""Ollama LLM provider."""

import logging
from typing import List, Optional

import ollama

from src.generation.base import BaseLLM
from src.generation.factory import register_llm

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always cite the source of your information.
If the context doesn't contain the answer, say "I don't have enough information to answer this."
"""


@register_llm("ollama")
class OllamaLLM(BaseLLM):
    """
    LLM using Ollama.
    
    Usage:
        llm = OllamaLLM(
            host="http://localhost:11434",
            model="llama3.2:latest"
        )
        response = llm.generate("What is RAG?")
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            host: Ollama server URL
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Default system prompt
        """
        self.host = host
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._client = ollama.Client(host=host)
        
        logger.info(f"Initialized OllamaLLM: model={model}, host={host}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional override for system prompt
            
        Returns:
            Generated text
        """
        sys_prompt = system_prompt or self.system_prompt
        
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        )
        
        return response["message"]["content"]

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
            system_prompt: Optional override for system prompt
            
        Returns:
            Generated text
        """
        # Format context
        formatted_context = "\n\n---\n\n".join(context)
        
        # Build prompt with context
        prompt = f"""Context:
{formatted_context}

---

Question: {query}

Answer based on the context above:"""

        logger.debug(f"Generating with {len(context)} context chunks")
        return self.generate(prompt, system_prompt)

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.model

    def health_check(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            self._client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


def format_context_with_metadata(
    results,
    include_source: bool = True,
    include_section: bool = True,
    include_domain: bool = False,
    source_mapping: dict = None,
) -> list:
    """
    Format retrieval results with metadata for better LLM context.
    
    Args:
        results: List of RetrievalResult objects
        include_source: Include source identifier
        include_section: Include section name
        include_domain: Include domain name
        source_mapping: Optional dict to map source IDs to readable names
                       e.g., {"AAPL": "Apple Inc.", "NFLX": "Netflix Inc."}
    
    Returns:
        List of formatted context strings
    """
    if source_mapping is None:
        source_mapping = {}
    
    formatted = []
    for r in results:
        source = r.metadata.get('source', '')
        section = r.metadata.get('section', '')
        domain = r.metadata.get('domain', '')
        
        # Map source to readable name if available
        source_name = source_mapping.get(source, source)
        
        # Build header
        header_parts = []
        if include_source and source_name:
            header_parts.append(f"Source: {source_name}")
        if include_section and section:
            header_parts.append(f"Section: {section}")
        if include_domain and domain:
            header_parts.append(f"Domain: {domain}")
        
        if header_parts:
            header = " | ".join(header_parts)
            formatted.append(f"[{header}]\n{r.content}")
        else:
            formatted.append(r.content)
    
    return formatted
