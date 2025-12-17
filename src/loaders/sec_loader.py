"""SEC EDGAR document loader for 10-K, 10-Q, 8-K filings."""
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from html.parser import HTMLParser

from src.loaders.base import BaseLoader, Document
from src.loaders.factory import register_loader

logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """Extract clean text from HTML, removing scripts and styles."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self.current_skip = set()

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.current_skip.add(tag)

    def handle_endtag(self, tag):
        self.current_skip.discard(tag)

    def handle_data(self, data):
        if not self.current_skip:
            text = data.strip()
            if text and len(text) > 1:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return ' '.join(self.text_parts)


@register_loader('.sec')
class SECLoader(BaseLoader):
    """
    Load SEC EDGAR filings (10-K, 10-Q, 8-K).

    Features:
    - Download filings from SEC EDGAR
    - Parse HTML/XBRL to clean text
    - Extract sections (Business, Risk Factors, MD&A, etc.)
    - Support for local files or direct download
    - Auto-detection of SEC filings by content

    Usage:
        loader = SECLoader()

        # Download and load
        docs = loader.download_filing("NFLX", "10-K", limit=1)

        # Load from local file
        docs = loader.load("path/to/filing.txt")

        # Load with section extraction
        docs = loader.load_with_sections("path/to/filing.txt")
    """

    # Common 10-K section headers (for extraction)
    SECTION_PATTERNS = {
        'business': r'(?:item\s*1\.?\s*business|item\s*1\b)',
        'risk_factors': r'(?:item\s*1a\.?\s*risk\s*factors|risk\s*factors)',
        'properties': r'(?:item\s*2\.?\s*properties)',
        'legal_proceedings': r'(?:item\s*3\.?\s*legal\s*proceedings)',
        'mda': r'(?:item\s*7\.?\s*management.{0,30}discussion|md&a)',
        'financial_statements': r'(?:item\s*8\.?\s*financial\s*statements)',
        'executive_compensation': r'(?:item\s*11\.?\s*executive\s*compensation)',
    }

    # Patterns to detect SEC EDGAR filings
    SEC_SIGNATURES = [
        b'<SEC-DOCUMENT>',
        b'<SEC-HEADER>',
        b'ACCESSION NUMBER:',
        b'CONFORMED SUBMISSION TYPE:',
        b'CENTRAL INDEX KEY:',
        b'sec-edgar-filings',
    ]

    def __init__(
        self,
        company_name: str = "RAGSystem",
        email: str = "user@example.com",
        download_dir: str = "data/sec_filings",
    ):
        """
        Initialize SEC loader.

        Args:
            company_name: Your company/app name (required by SEC)
            email: Your email (required by SEC for rate limiting)
            download_dir: Directory to store downloaded filings
        """
        self.company_name = company_name
        self.email = email
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self._downloader = None
        logger.info(f"Initialized SECLoader (download_dir={download_dir})")

    @property
    def downloader(self):
        """Lazy-load SEC downloader."""
        if self._downloader is None:
            try:
                from sec_edgar_downloader import Downloader
                self._downloader = Downloader(
                    self.company_name,
                    self.email,
                    str(self.download_dir)
                )
            except ImportError:
                raise ImportError(
                    "sec-edgar-downloader not installed. "
                    "Run: pip install sec-edgar-downloader"
                )
        return self._downloader

    def can_load(self, file_path: Path) -> bool:
        """
        Check if this loader can handle the given file.
        
        Uses both extension and content detection for .txt files.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if this loader should handle the file
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        # Always handle .sec extension
        if ext == '.sec':
            return True
        
        # For .txt files, check content for SEC signatures
        if ext == '.txt':
            return self._is_sec_filing(file_path)
        
        # Handle .htm/.html if they look like SEC filings
        if ext in ('.htm', '.html'):
            return self._is_sec_filing(file_path)
        
        return False

    def _is_sec_filing(self, file_path: Path) -> bool:
        """
        Detect if file is an SEC EDGAR filing by checking content.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file appears to be SEC filing
        """
        if not file_path.exists():
            return False
        
        try:
            # Read first 4KB to check for signatures
            with open(file_path, 'rb') as f:
                header = f.read(4096)
            
            # Check for SEC signatures
            matches = sum(1 for sig in self.SEC_SIGNATURES if sig in header)
            
            # Also check path for sec-edgar-filings pattern
            if 'sec-edgar-filings' in str(file_path).lower():
                matches += 2
            
            # Need at least 2 matches to be confident
            return matches >= 2
            
        except Exception as e:
            logger.debug(f"Could not check SEC signatures in {file_path}: {e}")
            return False

    def load(self, file_path: str) -> List[Document]:
        """
        Load SEC filing from local file.

        Args:
            file_path: Path to filing file

        Returns:
            List containing single Document with full text
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Filing not found: {file_path}")

        content = path.read_text(errors='ignore')
        clean_text = self._extract_text(content)

        metadata = {
            'source': str(path),
            'file_name': path.name,
            'file_type': 'sec_filing',
            'char_count': len(clean_text),
        }

        # Try to extract filing metadata from content
        metadata.update(self._extract_filing_metadata(content))

        return [Document(content=clean_text, metadata=metadata)]

    def load_with_sections(self, file_path: str) -> List[Document]:
        """
        Load SEC filing and split into sections.

        Args:
            file_path: Path to filing file

        Returns:
            List of Documents, one per section
        """
        path = Path(file_path)
        content = path.read_text(errors='ignore')
        clean_text = self._extract_text(content)

        base_metadata = {
            'source': str(path),
            'file_name': path.name,
            'file_type': 'sec_filing',
        }
        base_metadata.update(self._extract_filing_metadata(content))

        # Extract sections
        sections = self._extract_sections(clean_text)

        documents = []
        for section_name, section_text in sections.items():
            if len(section_text.strip()) > 100:  # Skip empty/tiny sections
                doc_metadata = base_metadata.copy()
                doc_metadata['section'] = section_name
                doc_metadata['char_count'] = len(section_text)
                documents.append(Document(
                    content=section_text,
                    metadata=doc_metadata
                ))

        logger.info(f"Extracted {len(documents)} sections from {path.name}")
        return documents

    def download_filing(
        self,
        ticker: str,
        filing_type: str = "10-K",
        limit: int = 1,
    ) -> List[Document]:
        """
        Download filing from SEC EDGAR and load.

        Args:
            ticker: Stock ticker (e.g., "NFLX", "AAPL")
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            limit: Number of filings to download

        Returns:
            List of Documents
        """
        logger.info(f"Downloading {filing_type} for {ticker}...")

        # Download
        self.downloader.get(filing_type, ticker, limit=limit)

        # Find downloaded files
        filing_dir = self.download_dir / "sec-edgar-filings" / ticker / filing_type

        if not filing_dir.exists():
            raise FileNotFoundError(f"No filings found for {ticker} {filing_type}")

        documents = []
        for filing_folder in filing_dir.iterdir():
            if filing_folder.is_dir():
                # Find the main filing file
                for f in filing_folder.iterdir():
                    if f.name.endswith('.txt') or f.name.endswith('.htm'):
                        docs = self.load(str(f))
                        for doc in docs:
                            doc.metadata['ticker'] = ticker
                            doc.metadata['filing_type'] = filing_type
                        documents.extend(docs)
                        break

        logger.info(f"Loaded {len(documents)} documents for {ticker}")
        return documents

    def _extract_text(self, content: str) -> str:
        """Extract clean text from HTML/XBRL content."""
        # Parse HTML
        parser = HTMLTextExtractor()
        try:
            parser.feed(content)
        except Exception as e:
            logger.warning(f"HTML parsing error: {e}")

        text = parser.get_text()

        # Clean up
        text = re.sub(r'&#\d+;', ' ', text)  # HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)  # Named entities
        text = re.sub(r'\s+', ' ', text)  # Whitespace
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Broken numbers

        # Remove common boilerplate
        boilerplate = [
            r'Table of Contents',
            r'UNITED STATES SECURITIES AND EXCHANGE COMMISSION',
            r'Washington, D\.C\. 20549',
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    def _extract_filing_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from filing content."""
        metadata = {}

        # Company name
        company_match = re.search(
            r'COMPANY CONFORMED NAME:\s*(.+?)(?:\n|$)',
            content, re.IGNORECASE
        )
        if company_match:
            metadata['company_name'] = company_match.group(1).strip()

        # Filing date
        date_match = re.search(
            r'FILED AS OF DATE:\s*(\d{8})',
            content, re.IGNORECASE
        )
        if date_match:
            date_str = date_match.group(1)
            metadata['filing_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        # CIK
        cik_match = re.search(
            r'CENTRAL INDEX KEY:\s*(\d+)',
            content, re.IGNORECASE
        )
        if cik_match:
            metadata['cik'] = cik_match.group(1)

        # Fiscal year
        fy_match = re.search(
            r'FISCAL YEAR END:\s*(\d{4})',
            content, re.IGNORECASE
        )
        if fy_match:
            metadata['fiscal_year_end'] = fy_match.group(1)
            
        # Filing type
        type_match = re.search(
            r'CONFORMED SUBMISSION TYPE:\s*(\S+)',
            content, re.IGNORECASE
        )
        if type_match:
            metadata['filing_type'] = type_match.group(1)

        return metadata

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract named sections from filing text."""
        sections = {}
        text_lower = text.lower()

        # Find section positions
        section_positions = []
        for section_name, pattern in self.SECTION_PATTERNS.items():
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                section_positions.append((match.start(), section_name))

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        # Extract text between sections
        for i, (start_pos, section_name) in enumerate(section_positions):
            # Find end position (start of next section or end of text)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(text)

            section_text = text[start_pos:end_pos]

            # Only keep if not already found (first occurrence)
            if section_name not in sections:
                sections[section_name] = section_text

        # If no sections found, return full text as 'full_document'
        if not sections:
            sections['full_document'] = text

        return sections

    def list_available_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
    ) -> List[Dict[str, Any]]:
        """
        List available filings in download directory.

        Args:
            ticker: Stock ticker
            filing_type: Type of filing

        Returns:
            List of filing metadata dicts
        """
        filing_dir = self.download_dir / "sec-edgar-filings" / ticker / filing_type

        if not filing_dir.exists():
            return []

        filings = []
        for folder in filing_dir.iterdir():
            if folder.is_dir():
                filings.append({
                    'accession_number': folder.name,
                    'path': str(folder),
                    'files': [f.name for f in folder.iterdir()],
                })

        return filings

    def supported_extensions(self) -> List[str]:
        """Return list of supported extensions."""
        return [".sec", ".txt", ".htm", ".html"]

    def health_check(self) -> bool:
        """Check if SEC downloader is available."""
        try:
            _ = self.downloader
            return True
        except ImportError:
            return False
