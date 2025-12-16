"""Tests for recursive text chunker."""

import pytest

from src.chunkers.recursive_chunker import RecursiveChunker
from src.loaders.base import Document


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_chunk_basic(self):
        """Should split text into chunks."""
        # Arrange
        doc = Document(
            content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            metadata={"source": "test.txt"}
        )
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert len(chunks) >= 2
        assert all(len(c.content) <= 40 for c in chunks)  # Allow some flexibility

    def test_chunk_preserves_metadata(self):
        """Should preserve document metadata in chunks."""
        # Arrange
        doc = Document(
            content="Hello world.\n\nThis is a test.",
            metadata={"source": "test.txt", "author": "test"}
        )
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["author"] == "test"

    def test_chunk_adds_index(self):
        """Should add chunk index to metadata."""
        # Arrange
        doc = Document(
            content="Para one.\n\nPara two.\n\nPara three.",
            metadata={}
        )
        chunker = RecursiveChunker(chunk_size=15, chunk_overlap=3)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_empty_document(self):
        """Should return empty list for empty document."""
        # Arrange
        doc = Document(content="", metadata={})
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert chunks == []

    def test_whitespace_only_document(self):
        """Should return empty list for whitespace-only document."""
        # Arrange
        doc = Document(content="   \n\n\t  ", metadata={})
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert chunks == []

    def test_small_document(self):
        """Should return single chunk for small document."""
        # Arrange
        doc = Document(content="Small text", metadata={})
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert len(chunks) == 1

    def test_respects_paragraph_boundaries(self):
        """Should prefer splitting on paragraph boundaries."""
        # Arrange
        doc = Document(
            content="First paragraph with some text.\n\nSecond paragraph here.",
            metadata={}
        )
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(doc)

        # Assert - should split on \n\n
        assert len(chunks) >= 2

    def test_invalid_overlap(self):
        """Should raise error if overlap >= chunk_size."""
        # Act & Assert
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_custom_separators(self):
        """Should use custom separators when provided."""
        # Arrange
        doc = Document(content="one|two|three", metadata={})
        chunker = RecursiveChunker(
            chunk_size=5,
            chunk_overlap=0,
            separators=["|", ""]
        )

        # Act
        chunks = chunker.chunk(doc)

        # Assert
        assert len(chunks) >= 2

    def test_chunk_many(self):
        """Should chunk multiple documents."""
        # Arrange
        docs = [
            Document(content="Doc one.\n\nPara two.", metadata={"source": "a.txt"}),
            Document(content="Doc two.\n\nPara two.", metadata={"source": "b.txt"}),
        ]
        chunker = RecursiveChunker(chunk_size=15, chunk_overlap=3)

        # Act
        chunks = chunker.chunk_many(docs)

        # Assert
        assert len(chunks) >= 4
