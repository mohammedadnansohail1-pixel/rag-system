"""Tests for Ollama embedding provider."""

import pytest
from unittest.mock import patch, MagicMock

from src.embeddings.ollama_embeddings import OllamaEmbeddings


class TestOllamaEmbeddings:
    """Tests for OllamaEmbeddings."""

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_init(self, mock_client):
        """Should initialize with correct settings."""
        # Act
        embeddings = OllamaEmbeddings(
            host="http://localhost:11434",
            model="nomic-embed-text",
            dimensions=768
        )

        # Assert
        assert embeddings.host == "http://localhost:11434"
        assert embeddings.model == "nomic-embed-text"
        assert embeddings.get_dimensions() == 768

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_embed_text(self, mock_client_class):
        """Should generate embedding for single text."""
        # Arrange
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {
            "embedding": [0.1, 0.2, 0.3]
        }
        mock_client_class.return_value = mock_client
        
        embeddings = OllamaEmbeddings()

        # Act
        result = embeddings.embed_text("Hello world")

        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.assert_called_once_with(
            model="nomic-embed-text",
            prompt="Hello world"
        )

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_embed_batch(self, mock_client_class):
        """Should generate embeddings for multiple texts."""
        # Arrange
        mock_client = MagicMock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
            {"embedding": [0.5, 0.6]},
        ]
        mock_client_class.return_value = mock_client
        
        embeddings = OllamaEmbeddings()

        # Act
        result = embeddings.embed_batch(["text1", "text2", "text3"])

        # Assert
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_model_name_property(self, mock_client):
        """Should return model name."""
        # Arrange
        embeddings = OllamaEmbeddings(model="custom-model")

        # Act
        result = embeddings.model_name

        # Assert
        assert result == "custom-model"

    @patch("src.embeddings.ollama_embeddings.ollama.Client")
    def test_get_dimensions(self, mock_client):
        """Should return configured dimensions."""
        # Arrange
        embeddings = OllamaEmbeddings(dimensions=1024)

        # Act
        result = embeddings.get_dimensions()

        # Assert
        assert result == 1024
