"""Tests for Ollama embedding provider."""

import pytest
from unittest.mock import patch, MagicMock

from src.embeddings.ollama_embeddings import OllamaEmbeddings


class TestOllamaEmbeddings:
    """Tests for OllamaEmbeddings."""

    def test_init(self):
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

    @patch("src.embeddings.ollama_embeddings.requests.post")
    def test_embed_text(self, mock_post):
        """Should generate embedding for single text."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        embeddings = OllamaEmbeddings()

        # Act
        result = embeddings.embed_text("Hello world")

        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

    @patch("src.embeddings.ollama_embeddings.requests.post")
    def test_embed_batch(self, mock_post):
        """Should generate embeddings for multiple texts."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.side_effect = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
            {"embedding": [0.5, 0.6]},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        embeddings = OllamaEmbeddings()

        # Act
        result = embeddings.embed_batch(["text1", "text2", "text3"])

        # Assert
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]

    def test_model_name_property(self):
        """Should return model name."""
        # Arrange
        embeddings = OllamaEmbeddings(model="custom-model")

        # Act
        result = embeddings.model_name

        # Assert
        assert result == "custom-model"

    def test_get_dimensions(self):
        """Should return configured dimensions."""
        # Arrange
        embeddings = OllamaEmbeddings(dimensions=1024)

        # Act
        result = embeddings.get_dimensions()

        # Assert
        assert result == 1024

    def test_truncate_text(self):
        """Should truncate long text."""
        # Arrange
        embeddings = OllamaEmbeddings()
        long_text = "a" * 10000

        # Act
        result = embeddings._truncate_text(long_text, max_chars=100)

        # Assert
        assert len(result) == 100

    def test_clean_text(self):
        """Should clean problematic characters."""
        # Arrange
        embeddings = OllamaEmbeddings()
        dirty_text = "Hello\x00World\n\n  Multiple   spaces"

        # Act
        result = embeddings._clean_text(dirty_text)

        # Assert
        assert "\x00" not in result
        assert "  " not in result
