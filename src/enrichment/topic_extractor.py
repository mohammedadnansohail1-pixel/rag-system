"""Topic and keyword extraction using TF-IDF and domain patterns."""

import re
import logging
from collections import Counter
from typing import List, Dict, Optional, Set
from src.enrichment.base import BaseEnricher, EnrichmentResult

logger = logging.getLogger(__name__)


class TopicExtractor(BaseEnricher):
    """
    Extracts topics and keywords from text.
    
    Methods:
    - Domain-specific topic detection (finance, legal, tech)
    - Keyword extraction based on frequency and importance
    - N-gram extraction for phrases
    
    Usage:
        extractor = TopicExtractor()
        result = extractor.enrich("Revenue increased due to advertising growth...")
        print(result.topics)    # ['revenue', 'advertising']
        print(result.keywords)  # ['revenue', 'advertising', 'growth', 'increased']
    """
    
    # Domain-specific topics
    FINANCE_TOPICS = {
        'revenue', 'profit', 'income', 'earnings', 'margin', 'growth',
        'expense', 'cost', 'cash flow', 'assets', 'liabilities',
        'debt', 'equity', 'dividend', 'stock', 'share', 'valuation',
        'ebitda', 'operating income', 'net income', 'gross margin',
    }
    
    LEGAL_TOPICS = {
        'litigation', 'lawsuit', 'legal', 'court', 'settlement',
        'regulatory', 'compliance', 'investigation', 'antitrust',
        'patent', 'intellectual property', 'trademark', 'copyright',
        'class action', 'injunction', 'consent decree', 'subpoena',
    }
    
    RISK_TOPICS = {
        'risk', 'threat', 'uncertainty', 'volatility', 'exposure',
        'cybersecurity', 'security breach', 'data breach', 'hack',
        'competition', 'competitive', 'market risk', 'credit risk',
    }
    
    TECH_TOPICS = {
        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
        'cloud', 'data center', 'infrastructure', 'platform',
        'software', 'hardware', 'semiconductor', 'chip', 'gpu',
        'algorithm', 'automation', 'digital', 'technology',
    }
    
    BUSINESS_TOPICS = {
        'acquisition', 'merger', 'partnership', 'joint venture',
        'expansion', 'restructuring', 'layoff', 'hiring',
        'strategy', 'market share', 'customer', 'user',
        'product', 'service', 'segment', 'business unit',
    }
    
    # Stopwords to filter
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'us',
        'they', 'their', 'them', 'he', 'she', 'his', 'her', 'i', 'you', 'your',
        'which', 'who', 'whom', 'what', 'where', 'when', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
        'about', 'above', 'after', 'again', 'against', 'any', 'because',
        'before', 'below', 'between', 'during', 'into', 'through', 'under',
        'until', 'while', 'over', 'out', 'up', 'down', 'off', 'further',
    }
    
    def __init__(
        self,
        max_topics: int = 5,
        max_keywords: int = 10,
        min_keyword_length: int = 3,
        include_bigrams: bool = True,
        custom_topics: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Args:
            max_topics: Maximum number of topics to return
            max_keywords: Maximum number of keywords to return
            min_keyword_length: Minimum word length for keywords
            include_bigrams: Include two-word phrases
            custom_topics: Additional topic categories
        """
        self.max_topics = max_topics
        self.max_keywords = max_keywords
        self.min_keyword_length = min_keyword_length
        self.include_bigrams = include_bigrams
        
        # Combine all topic sets
        self.all_topics = (
            self.FINANCE_TOPICS | 
            self.LEGAL_TOPICS | 
            self.RISK_TOPICS | 
            self.TECH_TOPICS |
            self.BUSINESS_TOPICS
        )
        
        # Add custom topics
        if custom_topics:
            for category, topics in custom_topics.items():
                self.all_topics |= topics
        
        # Topic categories for classification
        self.topic_categories = {
            'finance': self.FINANCE_TOPICS,
            'legal': self.LEGAL_TOPICS,
            'risk': self.RISK_TOPICS,
            'technology': self.TECH_TOPICS,
            'business': self.BUSINESS_TOPICS,
        }
        
        logger.info(f"Initialized TopicExtractor with {len(self.all_topics)} topics")
    
    @property
    def name(self) -> str:
        return "topic_extractor"
    
    def enrich(self, content: str, metadata: Optional[Dict] = None) -> EnrichmentResult:
        """Extract topics and keywords from content."""
        content_lower = content.lower()
        
        # Extract domain topics
        topics = self._extract_topics(content_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(content_lower)
        
        # Detect topic categories
        categories = self._detect_categories(content_lower)
        
        return EnrichmentResult(
            topics=topics,
            keywords=keywords,
            metadata={"topic_categories": categories} if categories else {},
        )
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract matching topics from predefined sets."""
        found = []
        
        for topic in self.all_topics:
            # Use word boundary for single words, substring for phrases
            if ' ' in topic:
                if topic in content:
                    found.append(topic)
            else:
                if re.search(r'\b' + re.escape(topic) + r'\b', content):
                    found.append(topic)
        
        # Sort by position in text (earlier = more prominent)
        found.sort(key=lambda t: content.find(t))
        
        return found[:self.max_topics]
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords based on frequency."""
        # Tokenize
        words = re.findall(r'\b[a-z]+\b', content)
        
        # Filter
        words = [
            w for w in words 
            if len(w) >= self.min_keyword_length 
            and w not in self.STOPWORDS
        ]
        
        # Count frequency
        word_counts = Counter(words)
        
        # Get top words
        keywords = [word for word, count in word_counts.most_common(self.max_keywords * 2)]
        
        # Add bigrams if enabled
        if self.include_bigrams:
            bigrams = self._extract_bigrams(content)
            keywords = bigrams + keywords
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:self.max_keywords]
    
    def _extract_bigrams(self, content: str) -> List[str]:
        """Extract meaningful two-word phrases."""
        words = re.findall(r'\b[a-z]+\b', content)
        
        bigrams = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if (len(w1) >= 3 and len(w2) >= 3 and 
                w1 not in self.STOPWORDS and w2 not in self.STOPWORDS):
                bigram = f"{w1} {w2}"
                bigrams.append(bigram)
        
        # Count and return top bigrams
        bigram_counts = Counter(bigrams)
        return [bg for bg, count in bigram_counts.most_common(5) if count >= 2]
    
    def _detect_categories(self, content: str) -> List[str]:
        """Detect which topic categories are present."""
        categories = []
        
        for category, topics in self.topic_categories.items():
            matches = sum(1 for t in topics if t in content)
            if matches >= 2:  # At least 2 matches to count
                categories.append(category)
        
        return categories
