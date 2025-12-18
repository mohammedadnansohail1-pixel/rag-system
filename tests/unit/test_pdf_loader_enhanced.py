"""Tests for enhanced PDF loader with table extraction."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.loaders.pdf_loader_enhanced import (
    EnhancedPDFLoader,
    PDFPLUMBER_AVAILABLE,
)


class TestEnhancedPDFLoader:
    """Tests for EnhancedPDFLoader."""
    
    def test_init_defaults(self):
        """Should initialize with defaults."""
        loader = EnhancedPDFLoader()
        assert loader.extract_tables == PDFPLUMBER_AVAILABLE
        assert loader.table_format == "markdown"
        assert loader.min_table_rows == 2
        assert loader.min_table_cols == 2
    
    def test_init_custom(self):
        """Should accept custom settings."""
        loader = EnhancedPDFLoader(
            pages_as_documents=True,
            table_format="html",
            min_table_rows=3,
        )
        assert loader.pages_as_documents == True
        assert loader.table_format == "html"
        assert loader.min_table_rows == 3
    
    def test_supported_extensions(self):
        """Should support PDF."""
        loader = EnhancedPDFLoader()
        assert ".pdf" in loader.supported_extensions()
    
    def test_is_valid_table_empty(self):
        """Should reject empty tables."""
        loader = EnhancedPDFLoader()
        assert loader._is_valid_table([]) == False
        assert loader._is_valid_table(None) == False
    
    def test_is_valid_table_too_small(self):
        """Should reject tables with too few rows/cols."""
        loader = EnhancedPDFLoader(min_table_rows=2, min_table_cols=2)
        
        # Too few rows
        assert loader._is_valid_table([["a"]]) == False
        
        # Too few columns
        assert loader._is_valid_table([["a"], ["b"]]) == False
    
    def test_is_valid_table_sparse(self):
        """Should reject sparse tables."""
        loader = EnhancedPDFLoader(min_cell_density=0.5)
        
        # Mostly empty cells
        sparse_table = [
            ["a", "", "", ""],
            ["", "", "", ""],
            ["", "", "", "b"],
        ]
        assert loader._is_valid_table(sparse_table) == False
    
    def test_is_valid_table_good(self):
        """Should accept valid tables."""
        loader = EnhancedPDFLoader()
        
        good_table = [
            ["Header 1", "Header 2", "Header 3"],
            ["Data 1", "Data 2", "Data 3"],
            ["Data 4", "Data 5", "Data 6"],
        ]
        assert loader._is_valid_table(good_table) == True
    
    def test_table_to_markdown(self):
        """Should convert table to markdown."""
        loader = EnhancedPDFLoader()
        
        table = [
            ["Name", "Age", "City"],
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
        ]
        
        md = loader._table_to_markdown(table)
        
        assert "| Name | Age | City |" in md
        assert "| --- | --- | --- |" in md
        assert "| Alice | 30 | NYC |" in md
    
    def test_table_to_markdown_escapes_pipes(self):
        """Should escape pipe characters in cells."""
        loader = EnhancedPDFLoader()
        
        table = [
            ["A", "B"],
            ["x|y", "z"],
        ]
        
        md = loader._table_to_markdown(table)
        assert "x\\|y" in md
    
    def test_table_to_html(self):
        """Should convert table to HTML."""
        loader = EnhancedPDFLoader(table_format="html")
        
        table = [
            ["Name", "Age"],
            ["Alice", "30"],
        ]
        
        html = loader._table_to_html(table)
        
        assert "<table" in html
        assert "<th>Name</th>" in html
        assert "<td>Alice</td>" in html


class TestTableExtractionIntegration:
    """Integration tests for table extraction."""
    
    @pytest.fixture
    def sample_pdf_with_table(self, tmp_path):
        """Create a simple PDF with a table for testing."""
        # Skip if reportlab not available
        pytest.importorskip("reportlab")
        
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        pdf_path = tmp_path / "test_table.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        
        styles = getSampleStyleSheet()
        elements = []
        
        # Add a paragraph
        elements.append(Paragraph("Financial Summary", styles['Heading1']))
        
        # Add a table
        data = [
            ['Quarter', 'Revenue', 'Profit', 'Margin'],
            ['Q1 2024', '$100M', '$20M', '20%'],
            ['Q2 2024', '$120M', '$25M', '21%'],
            ['Q3 2024', '$110M', '$22M', '20%'],
        ]
        table = Table(data)
        elements.append(table)
        
        doc.build(elements)
        return pdf_path
    
    @pytest.mark.skipif(not PDFPLUMBER_AVAILABLE, reason="pdfplumber not installed")
    def test_load_pdf_with_table(self, sample_pdf_with_table):
        """Should extract table from PDF."""
        loader = EnhancedPDFLoader()
        docs = loader.load(sample_pdf_with_table)
        
        assert len(docs) == 1
        assert docs[0].metadata["filetype"] == "pdf"
        # Table should be in content (either extracted or in text)
        assert "Revenue" in docs[0].content or "Q1" in docs[0].content
