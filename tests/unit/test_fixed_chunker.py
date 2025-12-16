"""Tests for fixed size chunker."""

import pytest

from src.chunkers.fixed_chunker import FixedChunker
from src.loaders.base import Document


class TestFixedChunker:
    """Tests for FixedChunker."""

    def test_chunk_basic(self):
        """Should split text into chunks."""
        # Arrange
        doc = Document(content="a" * 100, metadata={"source": "test.txt"})
        chunker = FixedChunker(chunk_size=30, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert len(chunks) == 5
        assert all(len(c.content) <= 30 for c in chunks)

    def test_chunk_preserves_metadata(self):
        """Should preserve document metadata in chunks."""
        # Arrange
        doc = Document(
            content="Hello world this is a test",
            metadata={"source": "test.txt", "author": "test"}
        )
        chunker = FixedChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["author"] == "test"

    def test_chunk_adds_index(self):
        """Should add chunk index to metadata."""
        # Arrange
        doc = Document(content="a" * 50, metadata={"source": "test.txt"})
        chunker = FixedChunker(chunk_size=20, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[1].metadata["chunk_index"] == 1

    def test_chunk_overlap(self):
        """Should have overlapping content between chunks."""
        # Arrange
        doc = Document(content="0123456789" * 5, metadata={})
        chunker = FixedChunker(chunk_size=20, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(doc)

        # Assert - last 5 chars of chunk 0 should equal first 5 of chunk 1
        assert chunks[0].content[-5:] == chunks[1].content[:5]

    def test_empty_document(self):
        """Should return empty list for empty document."""
        # Arrange
        doc = Document(content="", metadata={})
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert chunks == []

    def test_whitespace_only_document(self):
        """Should return empty list for whitespace-only document."""
        # Arrange
        doc = Document(content="   \n\t  ", metadata={})
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert chunks == []

    def test_small_document(self):
        """Should return single chunk for small document."""
        # Arrange
        doc = Document(content="Small text", metadata={})
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert len(chunks) == 1
        assert chunks[0].content == "Small text"

    def test_invalid_overlap(self):
        """Should raise error if overlap >= chunk_size."""
        # Act & Assert
        with pytest.raises(ValueError):
            FixedChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_many(self):
        """Should chunk multiple documents."""
        # Arrange
        docs = [
            Document(content="a" * 50, metadata={"source": "a.txt"}),
            Document(content="b" * 50, metadata={"source": "b.txt"}),
        ]
        chunker = FixedChunker(chunk_size=20, chunk_overlap=5)

        # Act
        chunks = chunker.chunk_many(docs)

        # Assert
        assert len(chunks) == 8  # 4 from each doc
