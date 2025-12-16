"""Tests for text file loader."""

import pytest
from pathlib import Path

from src.loaders.text_loader import TextLoader
from src.loaders.exceptions import LoaderError


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_text_file(self, tmp_path):
        """Should load text file content."""
        # Arrange
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, world!")
        loader = TextLoader()

        # Act
        docs = loader.load(text_file)

        # Assert
        assert len(docs) == 1
        assert docs[0].content == "Hello, world!"
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["filetype"] == "txt"

    def test_load_multiline_text(self, tmp_path):
        """Should preserve multiline content."""
        # Arrange
        content = "Line 1\nLine 2\nLine 3"
        text_file = tmp_path / "multiline.txt"
        text_file.write_text(content)
        loader = TextLoader()

        # Act
        docs = loader.load(text_file)

        # Assert
        assert docs[0].content == content

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        # Arrange
        loader = TextLoader()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.txt"))

    def test_unsupported_extension(self, tmp_path):
        """Should raise LoaderError for wrong extension."""
        # Arrange
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf")
        loader = TextLoader()

        # Act & Assert
        with pytest.raises(LoaderError) as exc_info:
            loader.load(pdf_file)
        
        assert "Unsupported file type" in str(exc_info.value)

    def test_supported_extensions(self):
        """Should return .txt as supported."""
        # Arrange
        loader = TextLoader()

        # Act
        extensions = loader.supported_extensions()

        # Assert
        assert extensions == [".txt"]

    def test_can_load(self, tmp_path):
        """Should correctly identify loadable files."""
        # Arrange
        loader = TextLoader()
        txt_file = tmp_path / "test.txt"
        pdf_file = tmp_path / "test.pdf"

        # Act & Assert
        assert loader.can_load(txt_file) is True
        assert loader.can_load(pdf_file) is False
