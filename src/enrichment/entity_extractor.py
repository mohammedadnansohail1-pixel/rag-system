"""Entity extraction using regex patterns and transformers NER."""

import re
import logging
from typing import List, Dict, Optional, Set
from src.enrichment.base import BaseEnricher, EnrichmentResult

logger = logging.getLogger(__name__)


class EntityExtractor(BaseEnricher):
    """
    Extracts entities from text using regex patterns.
    
    Fast, free entity extraction for:
    - Money amounts ($1.5 billion, EUR 500 million)
    - Percentages (15%, 3.5 percent)
    - Dates (2024, Q3 2024, December 31, 2024)
    - Company names (from known list + patterns)
    - Legal/regulatory entities (FTC, SEC, DOJ)
    
    Usage:
        extractor = EntityExtractor()
        result = extractor.enrich("Revenue grew 15% to $134 billion in 2024")
        print(result.entities)
        # {'money': ['$134 billion'], 'percentage': ['15%'], 'date': ['2024']}
    """
    
    # Money patterns
    MONEY_PATTERNS = [
        r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|K)?',
        r'(?:USD|EUR|GBP)\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion)?',
        r'[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion)\s*(?:dollars|euros)?',
        r'€[\d,]+(?:\.\d+)?\s*(?:billion|million)?',
    ]
    
    # Percentage patterns
    PERCENTAGE_PATTERNS = [
        r'[\d,]+(?:\.\d+)?\s*%',
        r'[\d,]+(?:\.\d+)?\s*percent',
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'\b(?:Q[1-4])\s*\d{4}\b',  # Q1 2024
        r'\b(?:FY|CY)\s*\d{4}\b',    # FY2024
        r'\b\d{4}\b',                 # 2024
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # 12/31/2024
    ]
    
    # Known regulatory/legal entities
    REGULATORY_ENTITIES = {
        'FTC', 'SEC', 'DOJ', 'FBI', 'IRS', 'CFPB',
        'Federal Trade Commission',
        'Securities and Exchange Commission',
        'Department of Justice',
        'Internal Revenue Service',
        'Consumer Financial Protection Bureau',
        'European Commission',
        'GDPR', 'CCPA',
        'IDPC', 'Irish Data Protection Commission',
    }
    
    # Known tech companies (expandable)
    KNOWN_COMPANIES = {
        'Meta', 'Facebook', 'Instagram', 'WhatsApp', 'Threads',
        'Google', 'Alphabet', 'YouTube', 'DeepMind',
        'Apple', 'Microsoft', 'Amazon', 'AWS',
        'NVIDIA', 'Tesla', 'OpenAI', 'Anthropic',
        'Netflix', 'Twitter', 'X Corp', 'TikTok', 'ByteDance',
        'Samsung', 'Intel', 'AMD', 'Qualcomm', 'TSMC',
        'Cisco', 'Oracle', 'Salesforce', 'Adobe',
    }
    
    def __init__(
        self,
        extract_money: bool = True,
        extract_percentages: bool = True,
        extract_dates: bool = True,
        extract_organizations: bool = True,
        custom_entities: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Args:
            extract_money: Extract money amounts
            extract_percentages: Extract percentages
            extract_dates: Extract dates
            extract_organizations: Extract org names
            custom_entities: Additional entity lists by type
        """
        self.extract_money = extract_money
        self.extract_percentages = extract_percentages
        self.extract_dates = extract_dates
        self.extract_organizations = extract_organizations
        self.custom_entities = custom_entities or {}
        
        # Compile patterns
        self._money_re = [re.compile(p, re.IGNORECASE) for p in self.MONEY_PATTERNS]
        self._pct_re = [re.compile(p, re.IGNORECASE) for p in self.PERCENTAGE_PATTERNS]
        self._date_re = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        
        logger.info(f"Initialized EntityExtractor")
    
    @property
    def name(self) -> str:
        return "entity_extractor"
    
    def enrich(self, content: str, metadata: Optional[Dict] = None) -> EnrichmentResult:
        """Extract entities from content."""
        entities: Dict[str, List[str]] = {}
        
        if self.extract_money:
            money = self._extract_money(content)
            if money:
                entities["money"] = money
        
        if self.extract_percentages:
            percentages = self._extract_percentages(content)
            if percentages:
                entities["percentage"] = percentages
        
        if self.extract_dates:
            dates = self._extract_dates(content)
            if dates:
                entities["date"] = dates
        
        if self.extract_organizations:
            orgs = self._extract_organizations(content)
            if orgs:
                entities["organization"] = orgs
        
        # Extract custom entities
        for entity_type, entity_set in self.custom_entities.items():
            found = self._extract_from_set(content, entity_set)
            if found:
                entities[entity_type] = found
        
        return EnrichmentResult(entities=entities)
    
    def _extract_money(self, content: str) -> List[str]:
        """Extract money amounts."""
        found = set()
        for pattern in self._money_re:
            matches = pattern.findall(content)
            found.update(m.strip() for m in matches)
        return sorted(found, key=lambda x: len(x), reverse=True)[:10]
    
    def _extract_percentages(self, content: str) -> List[str]:
        """Extract percentages."""
        found = set()
        for pattern in self._pct_re:
            matches = pattern.findall(content)
            found.update(m.strip() for m in matches)
        return sorted(found)[:10]
    
    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates."""
        found = set()
        for pattern in self._date_re:
            matches = pattern.findall(content)
            found.update(m.strip() for m in matches)
        # Filter out likely non-dates (numbers that are too small/large for years)
        filtered = [d for d in found if not (d.isdigit() and (int(d) < 1990 or int(d) > 2030))]
        return sorted(filtered)[:10]
    
    def _extract_organizations(self, content: str) -> List[str]:
        """Extract organization names."""
        found = set()
        
        # Check regulatory entities
        for entity in self.REGULATORY_ENTITIES:
            if entity.lower() in content.lower():
                # Get original case from content
                pattern = re.compile(re.escape(entity), re.IGNORECASE)
                match = pattern.search(content)
                if match:
                    found.add(match.group())
        
        # Check known companies
        for company in self.KNOWN_COMPANIES:
            if re.search(r'\b' + re.escape(company) + r'\b', content, re.IGNORECASE):
                found.add(company)
        
        return sorted(found)[:15]
    
    def _extract_from_set(self, content: str, entity_set: Set[str]) -> List[str]:
        """Extract entities from a custom set."""
        found = set()
        content_lower = content.lower()
        for entity in entity_set:
            if entity.lower() in content_lower:
                found.add(entity)
        return sorted(found)
