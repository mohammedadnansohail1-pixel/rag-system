"""Factory for creating LLM providers with registry pattern."""

import logging
from typing import Dict, Type, List, Callable

from src.generation.base import BaseLLM

logger = logging.getLogger(__name__)

# Registry to hold LLM classes
_LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {}


def register_llm(name: str) -> Callable:
    """
    Decorator to register an LLM provider class.
    
    Usage:
        @register_llm("ollama")
        class OllamaLLM(BaseLLM):
            ...
    """
    def decorator(cls: Type[BaseLLM]) -> Type[BaseLLM]:
        if name in _LLM_REGISTRY:
            logger.warning(f"Overwriting existing LLM: {name}")
        _LLM_REGISTRY[name] = cls
        logger.debug(f"Registered LLM: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_registered_llms() -> List[str]:
    """Return list of registered LLM names."""
    return list(_LLM_REGISTRY.keys())


class LLMFactory:
    """
    Factory that creates LLMs based on config.
    
    Usage:
        # From config dict
        llm = LLMFactory.from_config({
            "provider": "ollama",
            "ollama": {
                "host": "http://localhost:11434",
                "model": "llama3.1:8b"
            }
        })
        
        # Or directly
        llm = LLMFactory.create("ollama", host="...", model="...")
    """

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseLLM:
        """
        Create an LLM instance.
        
        Args:
            provider: Provider name ('ollama', 'openai', etc.)
            **kwargs: Provider-specific configuration
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is unknown
        """
        if provider not in _LLM_REGISTRY:
            available = get_registered_llms()
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available: {available}"
            )
        
        llm_class = _LLM_REGISTRY[provider]
        logger.info(f"Creating LLM: {provider}")
        
        return llm_class(**kwargs)

    @classmethod
    def from_config(cls, config: dict) -> BaseLLM:
        """
        Create LLM from config dict.
        
        Args:
            config: Config dict with provider and provider-specific settings
            
        Returns:
            LLM instance
        """
        provider = config.get("provider", "ollama")
        provider_config = config.get(provider, {})
        system_prompt = config.get("system_prompt")
        
        if system_prompt:
            provider_config["system_prompt"] = system_prompt
        
        logger.debug(f"Creating LLM from config: provider={provider}")
        
        return cls.create(provider, **provider_config)
