"""Tests for retriever factory with registry."""

import pytest
from unittest.mock import MagicMock

from src.retrieval.factory import RetrieverFactory, get_registered_retrievers
from src.retrieval.dense_retriever import DenseRetriever


class TestRetrieverFactory:
    """Tests for RetrieverFactory."""

    def test_get_registered_retrievers(self):
        """Should return list of registered retrievers."""
        # Act
        retrievers = get_registered_retrievers()

        # Assert
        assert "dense" in retrievers

    def test_create_dense_retriever(self):
        """Should create DenseRetriever."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_vectorstore = MagicMock()

        # Act
        retriever = RetrieverFactory.create(
            "dense",
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore,
            top_k=10
        )

        # Assert
        assert isinstance(retriever, DenseRetriever)
        assert retriever.top_k == 10

    def test_create_unknown_type(self):
        """Should raise ValueError for unknown type."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            RetrieverFactory.create("unknown")

        assert "Unknown retriever type" in str(exc_info.value)

    def test_from_config(self):
        """Should create retriever from config dict."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_vectorstore = MagicMock()
        config = {
            "search_type": "dense",
            "top_k": 5,
            "score_threshold": 0.7
        }

        # Act
        retriever = RetrieverFactory.from_config(
            config,
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore
        )

        # Assert
        assert isinstance(retriever, DenseRetriever)
        assert retriever.top_k == 5
        assert retriever.score_threshold == 0.7

    def test_from_config_defaults(self):
        """Should use defaults when config is minimal."""
        # Arrange
        mock_embeddings = MagicMock()
        mock_vectorstore = MagicMock()

        # Act
        retriever = RetrieverFactory.from_config(
            {},
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore
        )

        # Assert
        assert isinstance(retriever, DenseRetriever)
        assert retriever.top_k == 5
