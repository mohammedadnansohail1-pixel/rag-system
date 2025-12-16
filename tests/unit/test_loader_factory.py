"""Tests for loader factory."""

import pytest
from pathlib import Path

from src.loaders.factory import LoaderFactory
from src.loaders.text_loader import TextLoader
from src.loaders.markdown_loader import MarkdownLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.exceptions import UnsupportedFileTypeError


class TestLoaderFactory:
    """Tests for LoaderFactory."""

    def test_get_loader_txt(self):
        """Should return TextLoader for .txt files."""
        # Arrange
        factory = LoaderFactory()

        # Act
        loader = factory.get_loader(Path("file.txt"))

        # Assert
        assert isinstance(loader, TextLoader)

    def test_get_loader_md(self):
        """Should return MarkdownLoader for .md files."""
        # Arrange
        factory = LoaderFactory()

        # Act
        loader = factory.get_loader(Path("file.md"))

        # Assert
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_pdf(self):
        """Should return PDFLoader for .pdf files."""
        # Arrange
        factory = LoaderFactory()

        # Act
        loader = factory.get_loader(Path("file.pdf"))

        # Assert
        assert isinstance(loader, PDFLoader)

    def test_get_loader_unsupported(self):
        """Should raise UnsupportedFileTypeError for unknown type."""
        # Arrange
        factory = LoaderFactory()

        # Act & Assert
        with pytest.raises(UnsupportedFileTypeError):
            factory.get_loader(Path("file.xyz"))

    def test_load_single_file(self, tmp_path):
        """Should load a single file."""
        # Arrange
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")
        factory = LoaderFactory()

        # Act
        docs = factory.load(txt_file)

        # Assert
        assert len(docs) == 1
        assert docs[0].content == "Hello, world!"

    def test_load_directory(self, tmp_path):
        """Should load all supported files from directory."""
        # Arrange
        (tmp_path / "file1.txt").write_text("Text content")
        (tmp_path / "file2.md").write_text("# Markdown")
        (tmp_path / "skip.xyz").write_text("Unsupported")
        factory = LoaderFactory()

        # Act
        docs = factory.load_directory(tmp_path)

        # Assert
        assert len(docs) == 2

    def test_load_directory_with_filter(self, tmp_path):
        """Should filter by file types."""
        # Arrange
        (tmp_path / "file1.txt").write_text("Text")
        (tmp_path / "file2.md").write_text("Markdown")
        factory = LoaderFactory()

        # Act
        docs = factory.load_directory(tmp_path, file_types=[".txt"])

        # Assert
        assert len(docs) == 1
        assert docs[0].metadata["filetype"] == "txt"

    def test_load_directory_recursive(self, tmp_path):
        """Should load from subdirectories when recursive."""
        # Arrange
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("Root file")
        (subdir / "nested.txt").write_text("Nested file")
        factory = LoaderFactory()

        # Act
        docs = factory.load_directory(tmp_path, recursive=True)

        # Assert
        assert len(docs) == 2

    def test_load_directory_not_found(self):
        """Should raise FileNotFoundError for missing directory."""
        # Arrange
        factory = LoaderFactory()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            factory.load_directory(Path("/nonexistent/dir"))
