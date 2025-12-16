"""Tests for LLM factory with registry."""

import pytest
from unittest.mock import patch, MagicMock

from src.generation.factory import LLMFactory, get_registered_llms
from src.generation.ollama_llm import OllamaLLM


class TestLLMFactory:
    """Tests for LLMFactory."""

    def test_get_registered_llms(self):
        """Should return list of registered LLMs."""
        # Act
        llms = get_registered_llms()

        # Assert
        assert "ollama" in llms

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_create_ollama_llm(self, mock_client):
        """Should create OllamaLLM."""
        # Act
        llm = LLMFactory.create(
            "ollama",
            host="http://localhost:11434",
            model="llama3.1:8b"
        )

        # Assert
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama3.1:8b"

    def test_create_unknown_provider(self):
        """Should raise ValueError for unknown provider."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create("unknown")

        assert "Unknown LLM provider" in str(exc_info.value)

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_from_config(self, mock_client):
        """Should create LLM from config dict."""
        # Arrange
        config = {
            "provider": "ollama",
            "ollama": {
                "host": "http://localhost:11434",
                "model": "llama3.1:8b",
                "temperature": 0.2
            }
        }

        # Act
        llm = LLMFactory.from_config(config)

        # Assert
        assert isinstance(llm, OllamaLLM)
        assert llm.temperature == 0.2

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_from_config_with_system_prompt(self, mock_client):
        """Should pass system prompt from config."""
        # Arrange
        config = {
            "provider": "ollama",
            "ollama": {
                "model": "llama3.1:8b"
            },
            "system_prompt": "Custom system prompt"
        }

        # Act
        llm = LLMFactory.from_config(config)

        # Assert
        assert llm.system_prompt == "Custom system prompt"

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_from_config_defaults(self, mock_client):
        """Should use defaults when config is minimal."""
        # Act
        llm = LLMFactory.from_config({})

        # Assert
        assert isinstance(llm, OllamaLLM)
