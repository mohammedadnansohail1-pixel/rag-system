"""Generation module - pluggable LLM providers."""

# Import base and factory first (defines registry and decorator)
from src.generation.base import BaseLLM
from src.generation.factory import (
    LLMFactory,
    register_llm,
    get_registered_llms,
)

# Import LLMs to trigger registration
from src.generation.ollama_llm import OllamaLLM

__all__ = [
    "BaseLLM",
    "LLMFactory",
    "register_llm",
    "get_registered_llms",
    "OllamaLLM",
]
