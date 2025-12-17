"""SEC EDGAR document loader for 10-K, 10-Q, 8-K filings."""
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
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
        if tag.lower() in self.skip_tags:
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
    - Smart section extraction that skips Table of Contents

    Usage:
        loader = SECLoader()

        # Download and load
        docs = loader.download_filing("NFLX", "10-K", limit=1)

        # Load from local file
        docs = loader.load("path/to/filing.txt")

        # Load with section extraction (recommended)
        docs = loader.load_with_sections("path/to/filing.txt")
    """

    # Section patterns for 10-K filings
    SECTION_PATTERNS_10K = [
        ('business', r'Item\s+1\.?\s*Business'),
        ('risk_factors', r'Item\s+1A\.?\s*Risk\s*Factors'),
        ('unresolved_comments', r'Item\s+1B\.?\s*Unresolved\s*Staff\s*Comments'),
        ('cybersecurity', r'Item\s+1C\.?\s*Cybersecurity'),
        ('properties', r'Item\s+2\.?\s*Properties'),
        ('legal_proceedings', r'Item\s+3\.?\s*Legal\s*Proceedings'),
        ('market_info', r'Item\s+5\.?\s*Market\s*for'),
        ('mda', r'Item\s+7\.?\s*Management.{0,5}s?\s*Discussion'),
        ('market_risk', r'Item\s+7A\.?\s*Quantitative'),
        ('financial_statements', r'Item\s+8\.?\s*Financial\s*Statements'),
        ('controls', r'Item\s+9A\.?\s*Controls'),
    ]

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
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == '.sec':
            return True
        
        if ext in ('.txt', '.htm', '.html'):
            return self._is_sec_filing(file_path)
        
        return False

    def _is_sec_filing(self, file_path: Path) -> bool:
        """Detect if file is an SEC EDGAR filing by checking content."""
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4096)
            
            matches = sum(1 for sig in self.SEC_SIGNATURES if sig in header)
            
            if 'sec-edgar-filings' in str(file_path).lower():
                matches += 2
            
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
        metadata.update(self._extract_filing_metadata(content))

        return [Document(content=clean_text, metadata=metadata)]

    def load_with_sections(self, file_path: str) -> List[Document]:
        """
        Load SEC filing and split into meaningful sections.
        
        Intelligently extracts actual section content, skipping
        Table of Contents entries.

        Args:
            file_path: Path to filing file

        Returns:
            List of Documents, one per section
        """
        path = Path(file_path)
        content = path.read_text(errors='ignore')
        
        # Get filing metadata
        base_metadata = {
            'source': str(path),
            'file_name': path.name,
            'file_type': 'sec_filing',
        }
        base_metadata.update(self._extract_filing_metadata(content))
        
        # Determine filing type and extract sections
        filing_type = base_metadata.get('filing_type', '10-K')
        sections, full_text = self._extract_sections_smart(content, filing_type)
        
        documents = []
        for section_name, section_text in sections.items():
            if len(section_text.strip()) > 200:  # Skip tiny sections
                doc_metadata = base_metadata.copy()
                doc_metadata['section'] = section_name
                doc_metadata['char_count'] = len(section_text)
                documents.append(Document(
                    content=section_text,
                    metadata=doc_metadata
                ))

        # If no sections extracted, return full document
        if not documents:
            logger.warning(f"No sections extracted, returning full document")
            return self.load(file_path)

        logger.info(f"Extracted {len(documents)} sections from {path.name}")
        return documents

    def _extract_sections_smart(
        self, 
        content: str, 
        filing_type: str = "10-K"
    ) -> Tuple[Dict[str, str], str]:
        """
        Extract sections from filing, intelligently skipping TOC.
        
        Args:
            content: Raw filing content
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            
        Returns:
            Tuple of (sections dict, full clean text)
        """
        # Extract main document (10-K, 10-Q, etc.)
        main_doc = self._extract_main_document(content, filing_type)
        
        # Parse HTML to text
        parser = HTMLTextExtractor()
        try:
            parser.feed(main_doc)
        except Exception as e:
            logger.warning(f"HTML parsing error: {e}")
        
        clean_text = parser.get_text()
        
        # Remove XBRL noise
        clean_text = re.sub(r'\b\d{10}\s+\S*:\S+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Select patterns based on filing type
        if '10-K' in filing_type:
            patterns = self.SECTION_PATTERNS_10K
        else:
            # Default patterns work for most filings
            patterns = self.SECTION_PATTERNS_10K
        
        sections = {}
        
        for section_name, pattern in patterns:
            matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))
            
            if not matches:
                continue
            
            # If multiple occurrences, skip first (likely TOC)
            if len(matches) >= 2:
                start_idx = matches[1].start()
            else:
                start_idx = matches[0].start()
            
            # Find section end (next Item header)
            search_start = start_idx + 100  # Skip past current header
            next_item = re.search(
                r'Item\s+\d+[A-C]?\.', 
                clean_text[search_start:], 
                re.IGNORECASE
            )
            
            if next_item:
                end_idx = search_start + next_item.start()
            else:
                # Cap at 100k chars if no next section
                end_idx = min(start_idx + 100000, len(clean_text))
            
            section_text = clean_text[start_idx:end_idx].strip()
            
            # Only keep substantial sections (not TOC entries)
            if len(section_text) > 500:
                sections[section_name] = section_text
        
        return sections, clean_text

    def _extract_main_document(self, content: str, doc_type: str = "10-K") -> str:
        """Extract the main document from multi-document filing."""
        # Find embedded documents
        docs = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', content, re.DOTALL)
        
        if not docs:
            return content
        
        # Find the main filing document
        for doc in docs:
            type_match = re.search(rf'<TYPE>{doc_type}\s*\n', doc, re.IGNORECASE)
            if type_match:
                return doc
        
        # Fallback to first document
        return docs[0] if docs else content

    def _extract_text(self, content: str) -> str:
        """Extract clean text from HTML/XBRL content."""
        # Try to extract main document first
        main_doc = self._extract_main_document(content)
        
        # Parse HTML
        parser = HTMLTextExtractor()
        try:
            parser.feed(main_doc)
        except Exception as e:
            logger.warning(f"HTML parsing error: {e}")

        text = parser.get_text()

        # Clean up
        text = re.sub(r'\b\d{10}\s+\S*:\S+', '', text)  # XBRL patterns
        text = re.sub(r'&#\d+;', ' ', text)  # HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)  # Named entities
        text = re.sub(r'\s+', ' ', text)  # Whitespace

        return text.strip()

    def _extract_filing_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from filing content."""
        metadata = {}

        patterns = {
            'company_name': r'COMPANY CONFORMED NAME:\s*(.+?)(?:\n|$)',
            'filing_date': r'FILED AS OF DATE:\s*(\d{8})',
            'cik': r'CENTRAL INDEX KEY:\s*(\d+)',
            'fiscal_year_end': r'FISCAL YEAR END:\s*(\d{4})',
            'filing_type': r'CONFORMED SUBMISSION TYPE:\s*(\S+)',
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field == 'filing_date' and len(value) == 8:
                    value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
                metadata[field] = value

        return metadata

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

        self.downloader.get(filing_type, ticker, limit=limit)

        filing_dir = self.download_dir / "sec-edgar-filings" / ticker / filing_type

        if not filing_dir.exists():
            raise FileNotFoundError(f"No filings found for {ticker} {filing_type}")

        documents = []
        for filing_folder in filing_dir.iterdir():
            if filing_folder.is_dir():
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

    def list_available_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
    ) -> List[Dict[str, Any]]:
        """List available filings in download directory."""
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
