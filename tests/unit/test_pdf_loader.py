"""Tests for PDF file loader."""

import pytest
from pathlib import Path
from pypdf import PdfWriter

from src.loaders.pdf_loader import PDFLoader
from src.loaders.exceptions import LoaderError


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Create a minimal PDF for testing."""
    pdf_path = tmp_path / "test.pdf"
    writer = PdfWriter()
    # Add two blank pages
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


class TestPDFLoader:
    """Tests for PDFLoader."""

    def test_load_pdf_combined(self, sample_pdf):
        """Should load PDF with all pages combined."""
        # Arrange
        loader = PDFLoader(pages_as_documents=False)

        # Act
        docs = loader.load(sample_pdf)

        # Assert
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "test.pdf"
        assert docs[0].metadata["filetype"] == "pdf"
        assert docs[0].metadata["total_pages"] == 2

    def test_load_pdf_pages_separate(self, sample_pdf):
        """Should load PDF with each page as separate document."""
        # Arrange
        loader = PDFLoader(pages_as_documents=True)

        # Act
        docs = loader.load(sample_pdf)

        # Assert
        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2
        assert docs[0].metadata["total_pages"] == 2

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        # Arrange
        loader = PDFLoader()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.pdf"))

    def test_unsupported_extension(self, tmp_path):
        """Should raise LoaderError for wrong extension."""
        # Arrange
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("plain text")
        loader = PDFLoader()

        # Act & Assert
        with pytest.raises(LoaderError):
            loader.load(txt_file)

    def test_supported_extensions(self):
        """Should return .pdf as supported."""
        # Arrange
        loader = PDFLoader()

        # Act
        extensions = loader.supported_extensions()

        # Assert
        assert extensions == [".pdf"]

    def test_can_load(self):
        """Should correctly identify loadable files."""
        # Arrange
        loader = PDFLoader()

        # Act & Assert
        assert loader.can_load(Path("file.pdf")) is True
        assert loader.can_load(Path("file.PDF")) is True
        assert loader.can_load(Path("file.txt")) is False
