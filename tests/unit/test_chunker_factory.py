"""Tests for chunker factory with registry."""

import pytest

from src.chunkers.factory import ChunkerFactory, get_registered_chunkers
from src.chunkers.fixed_chunker import FixedChunker
from src.chunkers.recursive_chunker import RecursiveChunker


class TestChunkerFactory:
    """Tests for ChunkerFactory."""

    def test_get_registered_chunkers(self):
        """Should return list of registered chunkers."""
        # Act
        chunkers = get_registered_chunkers()

        # Assert
        assert "fixed" in chunkers
        assert "recursive" in chunkers

    def test_create_fixed_chunker(self):
        """Should create FixedChunker."""
        # Act
        chunker = ChunkerFactory.create("fixed", chunk_size=256)

        # Assert
        assert isinstance(chunker, FixedChunker)
        assert chunker.chunk_size == 256

    def test_create_recursive_chunker(self):
        """Should create RecursiveChunker."""
        # Act
        chunker = ChunkerFactory.create("recursive", chunk_size=512)

        # Assert
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunk_size == 512

    def test_create_with_custom_separators(self):
        """Should pass separators to recursive chunker."""
        # Act
        chunker = ChunkerFactory.create(
            "recursive",
            separators=["|", " "]
        )

        # Assert
        assert chunker.separators == ["|", " "]

    def test_create_unknown_strategy(self):
        """Should raise ValueError for unknown strategy."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkerFactory.create("unknown")

        assert "Unknown strategy" in str(exc_info.value)

    def test_from_config(self):
        """Should create chunker from config dict."""
        # Arrange
        config = {
            "strategy": "fixed",
            "chunk_size": 128,
            "chunk_overlap": 20
        }

        # Act
        chunker = ChunkerFactory.from_config(config)

        # Assert
        assert isinstance(chunker, FixedChunker)
        assert chunker.chunk_size == 128
        assert chunker.chunk_overlap == 20

    def test_from_config_defaults(self):
        """Should use defaults when config is empty."""
        # Act
        chunker = ChunkerFactory.from_config({})

        # Assert
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
