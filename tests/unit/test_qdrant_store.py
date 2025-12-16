"""Tests for Qdrant vector store."""

import pytest
from unittest.mock import patch, MagicMock

from src.vectorstores.qdrant_store import QdrantVectorStore
from src.vectorstores.base import SearchResult


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore."""

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_init_creates_collection(self, mock_client_class):
        """Should create collection if it doesn't exist."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        # Act
        store = QdrantVectorStore(collection_name="test_collection")

        # Assert
        mock_client.create_collection.assert_called_once()

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_init_skips_existing_collection(self, mock_client_class):
        """Should not create collection if it exists."""
        # Arrange
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.get_collections.return_value.collections = [mock_collection]
        mock_client_class.return_value = mock_client

        # Act
        store = QdrantVectorStore(collection_name="test_collection")

        # Assert
        mock_client.create_collection.assert_not_called()

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_add_documents(self, mock_client_class):
        """Should add documents to Qdrant."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore()

        # Act
        ids = store.add(
            texts=["text1", "text2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"source": "a"}, {"source": "b"}]
        )

        # Assert
        assert len(ids) == 2
        mock_client.upsert.assert_called_once()

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_add_with_custom_ids(self, mock_client_class):
        """Should use provided IDs."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore()

        # Act
        ids = store.add(
            texts=["text1"],
            embeddings=[[0.1, 0.2]],
            ids=["custom-id-1"]
        )

        # Assert
        assert ids == ["custom-id-1"]

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_search(self, mock_client_class):
        """Should return search results."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        
        mock_result = MagicMock()
        mock_result.payload = {"content": "test content", "source": "test.txt"}
        mock_result.score = 0.95
        
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_result]
        mock_client.query_points.return_value = mock_query_response
        
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore()

        # Act
        results = store.search(query_embedding=[0.1, 0.2], top_k=5)

        # Assert
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].content == "test content"
        assert results[0].score == 0.95

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_delete(self, mock_client_class):
        """Should delete documents by IDs."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore()

        # Act
        result = store.delete(["id1", "id2"])

        # Assert
        assert result is True
        mock_client.delete.assert_called_once()

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_count(self, mock_client_class):
        """Should return document count."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.get_collection.return_value.points_count = 42
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore()

        # Act
        count = store.count()

        # Assert
        assert count == 42

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_health_check_success(self, mock_client_class):
        """Should return True when Qdrant is accessible."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore()

        # Act
        result = store.health_check()

        # Assert
        assert result is True

    @patch("src.vectorstores.qdrant_store.QdrantClient")
    def test_health_check_failure(self, mock_client_class):
        """Should return False when Qdrant is not accessible."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore()
        
        # Make health check fail
        mock_client.get_collections.side_effect = Exception("Connection failed")

        # Act
        result = store.health_check()

        # Assert
        assert result is False
