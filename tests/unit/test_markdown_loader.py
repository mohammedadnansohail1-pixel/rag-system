"""Tests for markdown file loader."""

import pytest
from pathlib import Path

from src.loaders.markdown_loader import MarkdownLoader
from src.loaders.exceptions import LoaderError


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_load_md_file(self, tmp_path):
        """Should load .md file content."""
        # Arrange
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is markdown.")
        loader = MarkdownLoader()

        # Act
        docs = loader.load(md_file)

        # Assert
        assert len(docs) == 1
        assert "# Hello" in docs[0].content
        assert docs[0].metadata["filename"] == "test.md"
        assert docs[0].metadata["filetype"] == "markdown"

    def test_load_markdown_extension(self, tmp_path):
        """Should load .markdown extension too."""
        # Arrange
        md_file = tmp_path / "test.markdown"
        md_file.write_text("# Test")
        loader = MarkdownLoader()

        # Act
        docs = loader.load(md_file)

        # Assert
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "test.markdown"

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        # Arrange
        loader = MarkdownLoader()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.md"))

    def test_unsupported_extension(self, tmp_path):
        """Should raise LoaderError for wrong extension."""
        # Arrange
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("plain text")
        loader = MarkdownLoader()

        # Act & Assert
        with pytest.raises(LoaderError):
            loader.load(txt_file)

    def test_supported_extensions(self):
        """Should return .md and .markdown as supported."""
        # Arrange
        loader = MarkdownLoader()

        # Act
        extensions = loader.supported_extensions()

        # Assert
        assert ".md" in extensions
        assert ".markdown" in extensions

    def test_can_load(self, tmp_path):
        """Should correctly identify loadable files."""
        # Arrange
        loader = MarkdownLoader()

        # Act & Assert
        assert loader.can_load(Path("file.md")) is True
        assert loader.can_load(Path("file.markdown")) is True
        assert loader.can_load(Path("file.txt")) is False
