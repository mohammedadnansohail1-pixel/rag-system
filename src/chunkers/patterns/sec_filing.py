"""SEC filing pattern - 10-K, 10-Q, 8-K section detection."""

import re
from typing import List, Optional, Tuple
from html import unescape

from src.chunkers.patterns.base import BaseDocumentPattern, Section


class SECFilingPattern(BaseDocumentPattern):
    """
    Detects and extracts sections from SEC filings.
    
    Supports:
    - 10-K (Annual Reports)
    - 10-Q (Quarterly Reports)
    - 8-K (Current Reports)
    
    Handles:
    - HTML-encoded filings
    - Plain text filings
    - XBRL inline tags
    - Multiple DOCUMENT blocks
    """
    
    # SEC document markers
    SEC_HEADER_PATTERN = re.compile(r'<SEC-DOCUMENT>|<SEC-HEADER>', re.IGNORECASE)
    DOCUMENT_BLOCK_PATTERN = re.compile(
        r'<DOCUMENT>\s*<TYPE>([^<\n]+)', 
        re.IGNORECASE
    )
    
    # 10-K/10-Q Item patterns (flexible matching)
    ITEM_PATTERNS = [
        # Standard format: "Item 1." or "ITEM 1."
        re.compile(r'(?:^|\n)\s*(?:ITEM|Item)\s+(\d+[A-Za-z]?)\.?\s*[-–—]?\s*([^\n<]{3,80})', re.MULTILINE),
        # HTML format with tags
        re.compile(r'>\s*(?:ITEM|Item)\s+(\d+[A-Za-z]?)\.?\s*[-–—]?\s*([^<]{3,80})<', re.IGNORECASE),
        # Bold/formatted
        re.compile(r'(?:ITEM|Item)\s+(\d+[A-Za-z]?)[\.\s]+([A-Z][^<\n]{3,80})', re.MULTILINE),
    ]
    
    # Part patterns
    PART_PATTERNS = [
        re.compile(r'(?:^|\n)\s*(PART\s+[IVX]+)\s*[-–—.]?\s*([^\n<]*)', re.IGNORECASE | re.MULTILINE),
        re.compile(r'>\s*(PART\s+[IVX]+)\s*[-–—.]?\s*([^<]*)<', re.IGNORECASE),
    ]
    
    # HTML tag stripping
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    HTML_COMMENT_PATTERN = re.compile(r'<!--[\s\S]*?-->')
    XBRL_TAG_PATTERN = re.compile(r'</?ix:[^>]+>')
    STYLE_PATTERN = re.compile(r'<style[^>]*>[\s\S]*?</style>', re.IGNORECASE)
    SCRIPT_PATTERN = re.compile(r'<script[^>]*>[\s\S]*?</script>', re.IGNORECASE)
    
    # Standard SEC 10-K sections
    SECTION_NAMES = {
        "1": "Business",
        "1A": "Risk Factors",
        "1B": "Unresolved Staff Comments",
        "1C": "Cybersecurity",
        "2": "Properties",
        "3": "Legal Proceedings",
        "4": "Mine Safety Disclosures",
        "5": "Market for Registrant's Common Equity",
        "6": "Reserved",
        "7": "Management's Discussion and Analysis",
        "7A": "Quantitative and Qualitative Disclosures About Market Risk",
        "8": "Financial Statements and Supplementary Data",
        "9": "Changes in and Disagreements with Accountants",
        "9A": "Controls and Procedures",
        "9B": "Other Information",
        "9C": "Disclosure Regarding Foreign Jurisdictions",
        "10": "Directors, Executive Officers and Corporate Governance",
        "11": "Executive Compensation",
        "12": "Security Ownership",
        "13": "Certain Relationships and Related Transactions",
        "14": "Principal Accountant Fees and Services",
        "15": "Exhibits and Financial Statement Schedules",
        "16": "Form 10-K Summary",
    }
    
    @property
    def name(self) -> str:
        return "sec_filing"
    
    @property
    def description(self) -> str:
        return "SEC filings (10-K, 10-Q, 8-K)"
    
    def detect(self, content: str, metadata: Optional[dict] = None) -> float:
        """
        Detect if content is an SEC filing.
        
        Signals:
        - SEC-DOCUMENT or SEC-HEADER tags
        - DOCUMENT TYPE markers (10-K, 10-Q, 8-K)
        - ITEM patterns
        - Company metadata (CIK, ACCESSION NUMBER)
        """
        score = 0.0
        content_upper = content[:5000].upper()  # Check first 5k chars
        
        # Strong signals
        if self.SEC_HEADER_PATTERN.search(content):
            score += 0.4
        
        if '<TYPE>10-K' in content_upper or '<TYPE>10-Q' in content_upper:
            score += 0.3
        elif '10-K' in content_upper[:1000] or '10-Q' in content_upper[:1000]:
            score += 0.15
        
        # Check for SEC-specific markers
        sec_markers = ['ACCESSION NUMBER', 'CENTRAL INDEX KEY', 'CONFORMED SUBMISSION TYPE']
        for marker in sec_markers:
            if marker in content_upper:
                score += 0.1
        
        # Check for ITEM patterns
        item_count = 0
        for pattern in self.ITEM_PATTERNS:
            item_count += len(pattern.findall(content[:20000]))
        if item_count >= 5:
            score += 0.3
        elif item_count >= 2:
            score += 0.15
        
        # Metadata check
        if metadata:
            source = str(metadata.get("source", "")).upper()
            doc_type = str(metadata.get("type", "")).upper()
            if any(t in doc_type for t in ["10-K", "10-Q", "8-K"]):
                score += 0.3
            if any(t in source for t in ["SEC", "EDGAR"]):
                score += 0.1
        
        return min(score, 1.0)
    
    def preprocess(self, content: str) -> str:
        """
        Clean SEC filing content.
        
        Steps:
        1. Extract main 10-K document (skip exhibits)
        2. Remove HTML/XBRL tags
        3. Decode HTML entities
        4. Normalize whitespace
        """
        # Try to extract main document (not exhibits)
        main_content = self._extract_main_document(content)
        
        # Remove style and script blocks
        main_content = self.STYLE_PATTERN.sub('', main_content)
        main_content = self.SCRIPT_PATTERN.sub('', main_content)
        
        # Remove HTML comments
        main_content = self.HTML_COMMENT_PATTERN.sub('', main_content)
        
        # Remove XBRL tags but keep content
        main_content = self.XBRL_TAG_PATTERN.sub('', main_content)
        
        # Replace common block elements with newlines
        main_content = re.sub(r'</(?:div|p|tr|li|h[1-6])>', '\n', main_content, flags=re.IGNORECASE)
        main_content = re.sub(r'<(?:br|hr)[^>]*/?>', '\n', main_content, flags=re.IGNORECASE)
        
        # Remove remaining HTML tags
        main_content = self.HTML_TAG_PATTERN.sub(' ', main_content)
        
        # Decode HTML entities
        main_content = unescape(main_content)
        
        # Normalize whitespace
        main_content = re.sub(r'[ \t]+', ' ', main_content)
        main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
        main_content = main_content.strip()
        
        return main_content
    
    def _extract_main_document(self, content: str) -> str:
        """Extract the main 10-K/10-Q document, skipping exhibits."""
        # Find all DOCUMENT blocks
        doc_matches = list(self.DOCUMENT_BLOCK_PATTERN.finditer(content))
        
        if not doc_matches:
            return content
        
        # Find the main document (10-K or 10-Q, not EX-*)
        for i, match in enumerate(doc_matches):
            doc_type = match.group(1).strip().upper()
            if doc_type in ['10-K', '10-Q', '10-K/A', '10-Q/A']:
                # Get content until next DOCUMENT or end
                start = match.end()
                if i + 1 < len(doc_matches):
                    end = doc_matches[i + 1].start()
                else:
                    end = content.find('</DOCUMENT>', start)
                    if end == -1:
                        end = len(content)
                return content[start:end]
        
        # Fallback: return first document
        start = doc_matches[0].end()
        if len(doc_matches) > 1:
            end = doc_matches[1].start()
        else:
            end = len(content)
        return content[start:end]
    
    def extract_sections(self, content: str) -> List[Section]:
        """
        Extract sections based on ITEM and PART markers.
        
        Strategy:
        1. Preprocess to clean HTML
        2. Find PART markers (Part I, Part II, etc.)
        3. Find ITEM markers within parts
        4. Extract content between markers
        """
        # Preprocess content
        clean_content = self.preprocess(content)
        
        sections = []
        markers = []
        
        # Find PART markers
        for pattern in self.PART_PATTERNS:
            for match in pattern.finditer(clean_content):
                part_name = match.group(1).strip().upper()
                part_title = match.group(2).strip() if match.group(2) else ""
                markers.append({
                    "type": "part",
                    "name": part_name,
                    "title": part_title,
                    "pos": match.start(),
                    "end": match.end(),
                    "level": 0,
                })
        
        # Find ITEM markers
        for pattern in self.ITEM_PATTERNS:
            for match in pattern.finditer(clean_content):
                item_num = match.group(1).strip().upper()
                item_title = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                # Use standard name if title looks incomplete
                if len(item_title) < 5 or item_title.lower() in ['', 'none', '-']:
                    item_title = self.SECTION_NAMES.get(item_num, item_title)
                
                markers.append({
                    "type": "item",
                    "name": f"Item {item_num}",
                    "title": item_title,
                    "pos": match.start(),
                    "end": match.end(),
                    "level": 1,
                    "item_num": item_num,
                })
        
        # Remove duplicates (same position)
        seen_positions = set()
        unique_markers = []
        for m in markers:
            # Allow markers within 50 chars of each other to be considered same
            pos_key = m["pos"] // 50
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_markers.append(m)
        
        # Sort by position
        unique_markers.sort(key=lambda x: x["pos"])
        
        if not unique_markers:
            # No structure found - return as single section
            if clean_content.strip():
                sections.append(Section(
                    title="Document Content",
                    content=clean_content.strip(),
                    level=0,
                    start_pos=0,
                    end_pos=len(clean_content),
                    section_type="unstructured",
                ))
            return sections
        
        # Content before first marker
        if unique_markers[0]["pos"] > 100:
            preamble = clean_content[:unique_markers[0]["pos"]].strip()
            if len(preamble) > 50:
                sections.append(Section(
                    title="Header",
                    content=preamble,
                    level=0,
                    start_pos=0,
                    end_pos=unique_markers[0]["pos"],
                    section_type="header",
                ))
        
        # Extract sections between markers
        current_part = None
        for i, marker in enumerate(unique_markers):
            # Track current part for hierarchy
            if marker["type"] == "part":
                current_part = marker["name"]
            
            # Determine content end
            if i + 1 < len(unique_markers):
                content_end = unique_markers[i + 1]["pos"]
            else:
                content_end = len(clean_content)
            
            section_content = clean_content[marker["end"]:content_end].strip()
            
            # Skip empty sections
            if len(section_content) < 20:
                continue
            
            # Build section title
            if marker["title"]:
                full_title = f"{marker['name']} - {marker['title']}"
            else:
                full_title = marker["name"]
            
            section_type = marker.get("item_num", marker["type"])
            
            sections.append(Section(
                title=full_title,
                content=section_content,
                level=marker["level"],
                start_pos=marker["pos"],
                end_pos=content_end,
                section_type=section_type,
                parent_title=current_part if marker["type"] == "item" else None,
            ))
        
        return sections
    
    def get_fallback_separators(self) -> List[str]:
        """SEC-appropriate separators."""
        return ["\n\n\n", "\n\n", "\n", ". ", " "]
