"""Tests for embeddings factory with registry."""

import pytest
from unittest.mock import patch, MagicMock

from src.embeddings.factory import EmbeddingsFactory, get_registered_embeddings
from src.embeddings.ollama_embeddings import OllamaEmbeddings


class TestEmbeddingsFactory:
    """Tests for EmbeddingsFactory."""

    def test_get_registered_embeddings(self):
        """Should return list of registered providers."""
        # Act
        providers = get_registered_embeddings()

        # Assert
        assert "ollama" in providers

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_create_ollama_embeddings(self, mock_client):
        """Should create OllamaEmbeddings."""
        # Act
        embeddings = EmbeddingsFactory.create(
            "ollama",
            host="http://localhost:11434",
            model="nomic-embed-text"
        )

        # Assert
        assert isinstance(embeddings, OllamaEmbeddings)
        assert embeddings.model == "nomic-embed-text"

    def test_create_unknown_provider(self):
        """Should raise ValueError for unknown provider."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            EmbeddingsFactory.create("unknown")

        assert "Unknown embeddings provider" in str(exc_info.value)

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_from_config(self, mock_client):
        """Should create embeddings from config dict."""
        # Arrange
        config = {
            "provider": "ollama",
            "ollama": {
                "host": "http://localhost:11434",
                "model": "nomic-embed-text",
                "dimensions": 768
            }
        }

        # Act
        embeddings = EmbeddingsFactory.from_config(config)

        # Assert
        assert isinstance(embeddings, OllamaEmbeddings)
        assert embeddings.get_dimensions() == 768

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_from_config_defaults(self, mock_client):
        """Should use defaults when config is minimal."""
        # Act
        embeddings = EmbeddingsFactory.from_config({})

        # Assert
        assert isinstance(embeddings, OllamaEmbeddings)
