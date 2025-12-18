"""Enhanced PDF loader with table extraction."""
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import re

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from pypdf import PdfReader
from src.loaders.base import BaseLoader, Document
from src.loaders.exceptions import LoaderError
from src.loaders.factory import register_loader

logger = logging.getLogger(__name__)


@register_loader("pdf_enhanced")
class EnhancedPDFLoader(BaseLoader):
    """
    Enhanced PDF loader with table extraction.
    
    Features:
    - Extracts tables as Markdown format
    - Preserves table structure for better RAG retrieval
    - Filters low-quality/sparse tables
    - Falls back to standard text extraction if no tables
    
    Usage:
        loader = EnhancedPDFLoader()
        docs = loader.load(Path("document.pdf"))
    """
    
    def __init__(
        self,
        pages_as_documents: bool = False,
        extract_tables: bool = True,
        table_format: str = "markdown",
        min_table_rows: int = 2,
        min_table_cols: int = 2,
        min_cell_density: float = 0.3,  # At least 30% of cells must have content
    ):
        """
        Args:
            pages_as_documents: If True, return each page as separate Document
            extract_tables: If True, detect and extract tables specially
            table_format: Output format for tables ("markdown" or "html")
            min_table_rows: Minimum rows to consider something a table
            min_table_cols: Minimum columns to consider something a table
            min_cell_density: Minimum ratio of non-empty cells
        """
        self.pages_as_documents = pages_as_documents
        self.extract_tables = extract_tables and PDFPLUMBER_AVAILABLE
        self.table_format = table_format
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        self.min_cell_density = min_cell_density
        
        if extract_tables and not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not installed. Table extraction disabled.")
    
    def load(self, file_path: Path) -> List[Document]:
        """Load PDF with table extraction."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.can_load(file_path):
            raise LoaderError(f"Unsupported file type: {file_path.suffix}")
        
        if self.extract_tables:
            return self._load_with_tables(file_path)
        else:
            return self._load_basic(file_path)
    
    def _load_with_tables(self, file_path: Path) -> List[Document]:
        """Load PDF with pdfplumber for table extraction."""
        try:
            pdf = pdfplumber.open(file_path)
        except Exception as e:
            logger.warning(f"pdfplumber failed, falling back to basic: {e}")
            return self._load_basic(file_path)
        
        documents = []
        table_count = 0
        
        try:
            if self.pages_as_documents:
                for page_num, page in enumerate(pdf.pages, start=1):
                    content, tables_found = self._extract_page_with_tables(page)
                    table_count += tables_found
                    
                    documents.append(Document(
                        content=content,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "filetype": "pdf",
                            "page": page_num,
                            "total_pages": len(pdf.pages),
                            "tables_extracted": tables_found,
                        }
                    ))
            else:
                all_content = []
                for page in pdf.pages:
                    content, tables_found = self._extract_page_with_tables(page)
                    table_count += tables_found
                    all_content.append(content)
                
                documents.append(Document(
                    content="\n\n".join(all_content),
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "filetype": "pdf",
                        "total_pages": len(pdf.pages),
                        "tables_extracted": table_count,
                    }
                ))
        finally:
            pdf.close()
        
        logger.info(f"Loaded {file_path}: {len(pdf.pages)} pages, {table_count} tables extracted")
        return documents
    
    def _is_valid_table(self, table: List[List]) -> bool:
        """Check if table meets quality thresholds."""
        if not table:
            return False
        
        # Check row count
        if len(table) < self.min_table_rows:
            return False
        
        # Check column count
        max_cols = max(len(row) for row in table)
        if max_cols < self.min_table_cols:
            return False
        
        # Check cell density (non-empty cells)
        total_cells = 0
        non_empty_cells = 0
        
        for row in table:
            for cell in row:
                total_cells += 1
                if cell is not None and str(cell).strip():
                    non_empty_cells += 1
        
        if total_cells == 0:
            return False
        
        density = non_empty_cells / total_cells
        if density < self.min_cell_density:
            return False
        
        return True
    
    def _extract_page_with_tables(self, page) -> Tuple[str, int]:
        """
        Extract page content with tables converted to markdown.
        
        Returns:
            Tuple of (content_string, num_tables_found)
        """
        # Extract tables with better settings
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
        }
        
        try:
            tables = page.extract_tables(table_settings)
        except:
            # Fallback to default settings
            tables = page.extract_tables()
        
        # Filter valid tables
        valid_tables = [t for t in tables if self._is_valid_table(t)]
        
        if not valid_tables:
            # No valid tables, just extract text
            text = page.extract_text() or ""
            return text, 0
        
        # Get full page text
        full_text = page.extract_text() or ""
        
        # Convert tables to markdown
        table_markdowns = []
        for table in valid_tables:
            md = self._table_to_markdown(table)
            if md:
                table_markdowns.append(md)
        
        tables_found = len(table_markdowns)
        
        # Append tables at the end with clear markers
        if table_markdowns:
            content_parts = [full_text]
            content_parts.append("\n\n---\n\n**📊 EXTRACTED TABLES:**\n")
            for i, table_md in enumerate(table_markdowns, 1):
                content_parts.append(f"\n**Table {i}:**\n\n{table_md}\n")
            
            return "\n".join(content_parts), tables_found
        
        return full_text, 0
    
    def _table_to_markdown(self, table: List[List]) -> Optional[str]:
        """Convert table data to markdown format."""
        if not table or len(table) < self.min_table_rows:
            return None
        
        # Clean cells
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean whitespace and newlines
                    cell_str = str(cell).strip()
                    cell_str = re.sub(r'\s+', ' ', cell_str)
                    # Escape pipe characters
                    cell_str = cell_str.replace("|", "\\|")
                    cleaned_row.append(cell_str)
            cleaned_table.append(cleaned_row)
        
        # Ensure all rows have same number of columns
        max_cols = max(len(row) for row in cleaned_table)
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # Skip if first row (header) is all empty
        if all(not cell for cell in cleaned_table[0]):
            if len(cleaned_table) > 1:
                cleaned_table = cleaned_table[1:]
            else:
                return None
        
        if self.table_format == "html":
            return self._table_to_html(cleaned_table)
        
        # Build markdown table
        lines = []
        
        # Header row
        header = cleaned_table[0]
        # Use column numbers if header is empty
        if all(not cell for cell in header):
            header = [f"Col {i+1}" for i in range(len(header))]
        
        lines.append("| " + " | ".join(header) + " |")
        
        # Separator - align based on content
        separators = []
        for cell in header:
            separators.append("---")
        lines.append("| " + " | ".join(separators) + " |")
        
        # Data rows
        for row in cleaned_table[1:]:
            # Skip completely empty rows
            if all(not cell for cell in row):
                continue
            lines.append("| " + " | ".join(row) + " |")
        
        # Return None if only header + separator
        if len(lines) <= 2:
            return None
        
        return "\n".join(lines)
    
    def _table_to_html(self, table: List[List]) -> str:
        """Convert table to HTML format."""
        lines = ["<table border='1'>"]
        
        # Header
        lines.append("  <thead><tr>")
        for cell in table[0]:
            lines.append(f"    <th>{cell}</th>")
        lines.append("  </tr></thead>")
        
        # Body
        lines.append("  <tbody>")
        for row in table[1:]:
            if all(not cell for cell in row):
                continue
            lines.append("    <tr>")
            for cell in row:
                lines.append(f"      <td>{cell}</td>")
            lines.append("    </tr>")
        lines.append("  </tbody>")
        
        lines.append("</table>")
        return "\n".join(lines)
    
    def _load_basic(self, file_path: Path) -> List[Document]:
        """Fallback to basic pypdf loading."""
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            raise LoaderError(f"Failed to read PDF {file_path}: {e}")
        
        documents = []
        
        if self.pages_as_documents:
            for page_num, page in enumerate(reader.pages, start=1):
                content = page.extract_text() or ""
                documents.append(Document(
                    content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "filetype": "pdf",
                        "page": page_num,
                        "total_pages": len(reader.pages),
                        "tables_extracted": 0,
                    }
                ))
        else:
            all_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                all_text.append(text)
            
            documents.append(Document(
                content="\n\n".join(all_text),
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "filetype": "pdf",
                    "total_pages": len(reader.pages),
                    "tables_extracted": 0,
                }
            ))
        
        return documents
    
    def supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".pdf"]


def load_pdf_with_tables(
    file_path: Path,
    pages_as_documents: bool = False,
) -> List[Document]:
    """
    Convenience function to load PDF with table extraction.
    
    Args:
        file_path: Path to PDF file
        pages_as_documents: Split into per-page documents
        
    Returns:
        List of Documents with tables as markdown
    """
    loader = EnhancedPDFLoader(
        pages_as_documents=pages_as_documents,
        extract_tables=True,
    )
    return loader.load(file_path)
