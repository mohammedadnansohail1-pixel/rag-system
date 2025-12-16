"""Tests for vector store factory with registry."""

import pytest
from unittest.mock import patch, MagicMock

from src.vectorstores.factory import VectorStoreFactory, get_registered_vectorstores
from src.vectorstores.qdrant_store import QdrantVectorStore


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory."""

    def test_get_registered_vectorstores(self):
        """Should return list of registered stores."""
        # Act
        stores = get_registered_vectorstores()

        # Assert
        assert "qdrant" in stores

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_create_qdrant_store(self, mock_client):
        """Should create QdrantVectorStore."""
        # Arrange
        mock_client.return_value.get_collections.return_value.collections = []

        # Act
        store = VectorStoreFactory.create(
            "qdrant",
            host="localhost",
            port=6333,
            collection_name="test"
        )

        # Assert
        assert isinstance(store, QdrantVectorStore)
        assert store.collection_name == "test"

    def test_create_unknown_provider(self):
        """Should raise ValueError for unknown provider."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            VectorStoreFactory.create("unknown")

        assert "Unknown vector store provider" in str(exc_info.value)

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_from_config(self, mock_client):
        """Should create store from config dict."""
        # Arrange
        mock_client.return_value.get_collections.return_value.collections = []
        config = {
            "provider": "qdrant",
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "documents",
                "dimensions": 768
            }
        }

        # Act
        store = VectorStoreFactory.from_config(config)

        # Assert
        assert isinstance(store, QdrantVectorStore)
        assert store.dimensions == 768

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_from_config_defaults(self, mock_client):
        """Should use defaults when config is minimal."""
        # Arrange
        mock_client.return_value.get_collections.return_value.collections = []

        # Act
        store = VectorStoreFactory.from_config({})

        # Assert
        assert isinstance(store, QdrantVectorStore)
