"""Tests for dense retriever."""

import pytest
from unittest.mock import MagicMock

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.base import RetrievalResult
from src.vectorstores.base import SearchResult


class TestDenseRetriever:
    """Tests for DenseRetriever."""

    def test_init(self):
        """Should initialize with correct settings."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_vectorstore = MagicMock()

        # Act
        retriever = DenseRetriever(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore,
            top_k=10,
            score_threshold=0.5
        )

        # Assert
        assert retriever.top_k == 10
        assert retriever.score_threshold == 0.5

    def test_retrieve(self):
        """Should retrieve documents using embeddings."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = [0.1, 0.2, 0.3]
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.search.return_value = [
            SearchResult(content="Result 1", metadata={"source": "a.txt"}, score=0.9),
            SearchResult(content="Result 2", metadata={"source": "b.txt"}, score=0.8),
        ]
        
        retriever = DenseRetriever(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore
        )

        # Act
        results = retriever.retrieve("test query")

        # Assert
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].content == "Result 1"
        assert results[0].score == 0.9
        mock_embeddings.embed_text.assert_called_once_with("test query")

    def test_retrieve_with_custom_top_k(self):
        """Should use custom top_k when provided."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = [0.1, 0.2]
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.search.return_value = []
        
        retriever = DenseRetriever(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore,
            top_k=5
        )

        # Act
        retriever.retrieve("query", top_k=10)

        # Assert
        mock_vectorstore.search.assert_called_once()
        call_args = mock_vectorstore.search.call_args
        assert call_args.kwargs["top_k"] == 10

    def test_add_documents(self):
        """Should embed and add documents."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_embeddings.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.add.return_value = ["id1", "id2"]
        
        retriever = DenseRetriever(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore
        )

        # Act
        ids = retriever.add_documents(
            texts=["text1", "text2"],
            metadatas=[{"source": "a"}, {"source": "b"}]
        )

        # Assert
        assert ids == ["id1", "id2"]
        mock_embeddings.embed_batch.assert_called_once_with(["text1", "text2"])
        mock_vectorstore.add.assert_called_once()

    def test_health_check(self):
        """Should delegate health check to vectorstore."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_vectorstore = MagicMock()
        mock_vectorstore.health_check.return_value = True
        
        retriever = DenseRetriever(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore
        )

        # Act
        result = retriever.health_check()

        # Assert
        assert result is True
        mock_vectorstore.health_check.assert_called_once()
