"""Web/URL document loader for scraping web pages."""
import logging
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from src.loaders.base import BaseLoader, Document
from src.loaders.factory import register_loader

logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """Extract clean text from HTML, removing scripts/styles/nav."""
    
    SKIP_TAGS = {'script', 'style', 'meta', 'link', 'nav', 'footer', 'aside', 'noscript'}
    BLOCK_TAGS = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr', 'br', 'article', 'section'}
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_stack = []
        self.title = ""
        self._in_title = False
        self._in_head = False
    
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        
        # Track head section
        if tag == 'head':
            self._in_head = True
            return
        elif tag == 'body':
            self._in_head = False
            self.skip_stack.clear()  # Clear any unclosed skip tags
            return
        
        if tag in self.SKIP_TAGS:
            self.skip_stack.append(tag)
        elif tag == 'title':
            self._in_title = True
        elif tag in self.BLOCK_TAGS and self.text_parts:
            self.text_parts.append('\n')
    
    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == 'head':
            self._in_head = False
        elif tag in self.SKIP_TAGS and self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()
        elif tag == 'title':
            self._in_title = False
    
    def handle_data(self, data):
        if self._in_title:
            self.title = data.strip()
        elif not self._in_head and not self.skip_stack:
            text = data.strip()
            if text:
                self.text_parts.append(text)
    
    def get_text(self) -> str:
        text = ' '.join(self.text_parts)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


@dataclass
class CrawlConfig:
    """Configuration for web crawling."""
    max_pages: int = 50
    max_depth: int = 3
    delay_seconds: float = 1.0
    respect_robots: bool = True
    same_domain_only: bool = True
    allowed_extensions: tuple = ('.html', '.htm', '/')
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                r'/login', r'/signin', r'/signup', r'/register',
                r'/cart', r'/checkout', r'/account',
                r'\?', r'#',  # Query strings and anchors
            ]


@register_loader('.url')
class WebLoader(BaseLoader):
    """
    Load documents from web URLs.
    
    Features:
    - Single URL loading
    - Sitemap parsing
    - Recursive crawling with depth control
    - robots.txt compliance
    - Rate limiting
    - Clean text extraction from HTML
    
    Usage:
        loader = WebLoader()
        
        # Single page
        docs = loader.load("https://example.com/page")
        
        # From sitemap
        docs = loader.load_sitemap("https://example.com/sitemap.xml", max_urls=100)
        
        # Crawl entire site
        docs = loader.crawl("https://docs.example.com", max_pages=50)
    """
    
    DEFAULT_HEADERS = {
        'User-Agent': 'RAGBot/1.0 (Document Indexer; +https://github.com/rag-system)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """
        Initialize web loader.
        
        Args:
            headers: Custom HTTP headers
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.headers = headers or self.DEFAULT_HEADERS.copy()
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = None
        self._robots_cache: Dict[str, RobotFileParser] = {}
        
        logger.info("Initialized WebLoader")
    
    @property
    def session(self) -> requests.Session:
        """Lazy-load session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.headers)
        return self._session
    
    def load(self, url: str) -> List[Document]:
        """
        Load document from a single URL.
        
        Args:
            url: Web page URL
            
        Returns:
            List containing single Document
        """
        logger.info(f"Loading URL: {url}")
        
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            
            if 'text/html' in content_type:
                text, title = self._extract_html(response.text)
            elif 'text/plain' in content_type:
                text = response.text
                title = urlparse(url).path.split('/')[-1]
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                text = response.text
                title = ""
            
            metadata = {
                'source': url,
                'title': title,
                'content_type': content_type,
                'status_code': response.status_code,
                'char_count': len(text),
            }
            
            return [Document(content=text, metadata=metadata)]
            
        except requests.RequestException as e:
            logger.error(f"Failed to load {url}: {e}")
            raise
    
    def load_sitemap(
        self,
        sitemap_url: str,
        max_urls: int = 100,
        delay: float = 1.0,
    ) -> List[Document]:
        """
        Load documents from sitemap.xml.
        
        Args:
            sitemap_url: URL to sitemap.xml
            max_urls: Maximum URLs to process
            delay: Delay between requests in seconds
            
        Returns:
            List of Documents
        """
        logger.info(f"Loading sitemap: {sitemap_url}")
        
        urls = self._parse_sitemap(sitemap_url)
        logger.info(f"Found {len(urls)} URLs in sitemap")
        
        documents = []
        for i, url in enumerate(urls[:max_urls]):
            try:
                docs = self.load(url)
                documents.extend(docs)
                logger.info(f"Loaded {i+1}/{min(len(urls), max_urls)}: {url}")
                
                if delay > 0 and i < len(urls) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.warning(f"Failed to load {url}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents from sitemap")
        return documents
    
    def crawl(
        self,
        start_url: str,
        config: Optional[CrawlConfig] = None,
    ) -> List[Document]:
        """
        Crawl website starting from URL.
        
        Args:
            start_url: Starting URL
            config: Crawl configuration
            
        Returns:
            List of Documents
        """
        config = config or CrawlConfig()
        logger.info(f"Starting crawl from {start_url} (max_pages={config.max_pages})")
        
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc
        
        # Check robots.txt
        if config.respect_robots:
            robots = self._get_robots(f"{parsed_start.scheme}://{base_domain}")
        else:
            robots = None
        
        visited: Set[str] = set()
        to_visit: List[tuple] = [(start_url, 0)]  # (url, depth)
        documents: List[Document] = []
        
        while to_visit and len(documents) < config.max_pages:
            url, depth = to_visit.pop(0)
            
            # Normalize URL
            url = self._normalize_url(url)
            
            if url in visited:
                continue
            
            # Check domain
            parsed = urlparse(url)
            if config.same_domain_only and parsed.netloc != base_domain:
                continue
            
            # Check robots.txt
            if robots and not robots.can_fetch(self.headers.get('User-Agent', '*'), url):
                logger.debug(f"Blocked by robots.txt: {url}")
                continue
            
            # Check exclude patterns
            if any(re.search(p, url) for p in config.exclude_patterns):
                continue
            
            visited.add(url)
            
            try:
                # Load page
                response = self.session.get(url, timeout=self.timeout, verify=self.verify_ssl)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    continue
                
                # Extract text
                text, title = self._extract_html(response.text)
                
                if len(text.strip()) > 100:  # Skip near-empty pages
                    documents.append(Document(
                        content=text,
                        metadata={
                            'source': url,
                            'title': title,
                            'depth': depth,
                            'char_count': len(text),
                        }
                    ))
                    logger.info(f"Crawled {len(documents)}/{config.max_pages}: {url}")
                
                # Extract links for further crawling
                if depth < config.max_depth:
                    links = self._extract_links(response.text, url)
                    for link in links:
                        if link not in visited:
                            to_visit.append((link, depth + 1))
                
                # Rate limiting
                if config.delay_seconds > 0:
                    time.sleep(config.delay_seconds)
                    
            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")
                continue
        
        logger.info(f"Crawl complete: {len(documents)} documents from {len(visited)} URLs")
        return documents
    
    def _extract_html(self, html: str) -> tuple:
        """Extract clean text and title from HTML."""
        parser = HTMLTextExtractor()
        try:
            parser.feed(html)
        except Exception as e:
            logger.warning(f"HTML parsing error: {e}")
        
        return parser.get_text(), parser.title
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML."""
        links = []
        
        # Simple regex for href attributes
        href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
        
        for match in href_pattern.finditer(html):
            href = match.group(1)
            
            # Skip non-page links
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            
            # Basic filtering
            parsed = urlparse(full_url)
            if parsed.scheme in ('http', 'https'):
                links.append(full_url)
        
        return list(set(links))
    
    def _parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Parse sitemap XML for URLs."""
        response = self.session.get(sitemap_url, timeout=self.timeout)
        response.raise_for_status()
        
        urls = []
        
        # Handle sitemap index (contains other sitemaps)
        sitemap_pattern = re.compile(r'<sitemap>.*?<loc>([^<]+)</loc>.*?</sitemap>', re.DOTALL)
        for match in sitemap_pattern.finditer(response.text):
            nested_urls = self._parse_sitemap(match.group(1))
            urls.extend(nested_urls)
        
        # Handle regular sitemap (contains page URLs)
        url_pattern = re.compile(r'<url>.*?<loc>([^<]+)</loc>.*?</url>', re.DOTALL)
        for match in url_pattern.finditer(response.text):
            urls.append(match.group(1))
        
        return urls
    
    def _get_robots(self, base_url: str) -> Optional[RobotFileParser]:
        """Get robots.txt parser for domain."""
        if base_url in self._robots_cache:
            return self._robots_cache[base_url]
        
        robots_url = f"{base_url}/robots.txt"
        rp = RobotFileParser()
        
        try:
            rp.set_url(robots_url)
            rp.read()
            self._robots_cache[base_url] = rp
            logger.debug(f"Loaded robots.txt from {robots_url}")
            return rp
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {e}")
            return None
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove trailing slash, fragments
        path = parsed.path.rstrip('/')
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    
    def supported_extensions(self) -> List[str]:
        """Return supported extensions (not file-based)."""
        return ['.url', '.html', '.htm']
    
    def health_check(self) -> bool:
        """Check if loader can make requests."""
        try:
            response = self.session.get("https://httpbin.org/get", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
