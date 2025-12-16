"""Tests for Ollama LLM provider."""

import pytest
from unittest.mock import patch, MagicMock

from src.generation.ollama_llm import OllamaLLM


class TestOllamaLLM:
    """Tests for OllamaLLM."""

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_init(self, mock_client):
        """Should initialize with correct settings."""
        # Act
        llm = OllamaLLM(
            host="http://localhost:11434",
            model="llama3.1:8b",
            temperature=0.5,
            max_tokens=2048
        )

        # Assert
        assert llm.host == "http://localhost:11434"
        assert llm.model == "llama3.1:8b"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2048

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_generate(self, mock_client_class):
        """Should generate response."""
        # Arrange
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "Generated response"}
        }
        mock_client_class.return_value = mock_client
        
        llm = OllamaLLM()

        # Act
        result = llm.generate("What is RAG?")

        # Assert
        assert result == "Generated response"
        mock_client.chat.assert_called_once()

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_generate_with_custom_system_prompt(self, mock_client_class):
        """Should use custom system prompt when provided."""
        # Arrange
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "Response"}
        }
        mock_client_class.return_value = mock_client
        
        llm = OllamaLLM()

        # Act
        llm.generate("Question", system_prompt="Custom prompt")

        # Assert
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "Custom prompt"

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_generate_with_context(self, mock_client_class):
        """Should format context and generate response."""
        # Arrange
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "Answer with context"}
        }
        mock_client_class.return_value = mock_client
        
        llm = OllamaLLM()

        # Act
        result = llm.generate_with_context(
            query="What is X?",
            context=["Context chunk 1", "Context chunk 2"]
        )

        # Assert
        assert result == "Answer with context"
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        # User message should contain context
        assert "Context chunk 1" in messages[1]["content"]
        assert "Context chunk 2" in messages[1]["content"]

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_model_name_property(self, mock_client):
        """Should return model name."""
        # Arrange
        llm = OllamaLLM(model="custom-model")

        # Act
        result = llm.model_name

        # Assert
        assert result == "custom-model"

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_health_check_success(self, mock_client_class):
        """Should return True when Ollama is accessible."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        llm = OllamaLLM()

        # Act
        result = llm.health_check()

        # Assert
        assert result is True

    @patch("src.generation.ollama_llm.ollama.Client")
    def test_health_check_failure(self, mock_client_class):
        """Should return False when Ollama is not accessible."""
        # Arrange
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client
        
        llm = OllamaLLM()

        # Act
        result = llm.health_check()

        # Assert
        assert result is False
