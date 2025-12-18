"""Markdown document pattern - header-based section detection."""

import re
from typing import List, Optional

from src.chunkers.patterns.base import BaseDocumentPattern, Section


class MarkdownPattern(BaseDocumentPattern):
    """
    Detects and extracts sections from Markdown documents.
    
    Recognizes:
    - ATX headers: # H1, ## H2, ### H3, etc.
    - Setext headers: underlined with === or ---
    - Code blocks (preserved as single units)
    """
    
    # ATX style: # Header
    ATX_HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*#*)?$', re.MULTILINE)
    
    # Setext style: Header\n=====
    SETEXT_H1_PATTERN = re.compile(r'^(.+)\n={3,}\s*$', re.MULTILINE)
    SETEXT_H2_PATTERN = re.compile(r'^(.+)\n-{3,}\s*$', re.MULTILINE)
    
    # Code blocks
    FENCED_CODE_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    @property
    def name(self) -> str:
        return "markdown"
    
    @property
    def description(self) -> str:
        return "Markdown documents with header-based sections"
    
    def detect(self, content: str, metadata: Optional[dict] = None) -> float:
        """
        Detect if content is Markdown.
        
        Signals:
        - File extension .md or .markdown
        - ATX headers present
        - Markdown-specific syntax (```, *, _, [])
        """
        score = 0.0
        
        # Check metadata for file extension
        if metadata:
            filename = metadata.get("filename", "") or metadata.get("source", "")
            if filename.lower().endswith((".md", ".markdown")):
                score += 0.5
        
        # Check for ATX headers
        atx_matches = len(self.ATX_HEADER_PATTERN.findall(content))
        if atx_matches >= 2:
            score += 0.3
        elif atx_matches >= 1:
            score += 0.15
        
        # Check for code blocks
        if self.FENCED_CODE_PATTERN.search(content):
            score += 0.1
        
        # Check for other markdown syntax
        md_indicators = [
            r'\[.+\]\(.+\)',  # Links
            r'^\s*[-*+]\s',   # Unordered lists
            r'^\s*\d+\.\s',   # Ordered lists
            r'\*\*.+\*\*',    # Bold
            r'__.+__',        # Bold alt
        ]
        for pattern in md_indicators:
            if re.search(pattern, content, re.MULTILINE):
                score += 0.05
        
        return min(score, 1.0)
    
    def extract_sections(self, content: str) -> List[Section]:
        """
        Extract sections based on headers.
        
        Strategy:
        1. Find all headers with positions
        2. Content between headers belongs to preceding header
        3. Track hierarchy based on header level
        """
        sections = []
        
        # Find all headers with positions
        headers = []
        
        # ATX headers
        for match in self.ATX_HEADER_PATTERN.finditer(content):
            level = len(match.group(1))  # Number of #
            title = match.group(2).strip()
            headers.append({
                "title": title,
                "level": level,
                "start": match.start(),
                "header_end": match.end(),
            })
        
        # Setext H1
        for match in self.SETEXT_H1_PATTERN.finditer(content):
            headers.append({
                "title": match.group(1).strip(),
                "level": 1,
                "start": match.start(),
                "header_end": match.end(),
            })
        
        # Setext H2
        for match in self.SETEXT_H2_PATTERN.finditer(content):
            headers.append({
                "title": match.group(1).strip(),
                "level": 2,
                "start": match.start(),
                "header_end": match.end(),
            })
        
        # Sort by position
        headers.sort(key=lambda x: x["start"])
        
        if not headers:
            # No headers found - return entire content as one section
            if content.strip():
                sections.append(Section(
                    title="content",
                    content=content.strip(),
                    level=0,
                    start_pos=0,
                    end_pos=len(content),
                    section_type="body",
                ))
            return sections
        
        # Content before first header
        if headers[0]["start"] > 0:
            preamble = content[:headers[0]["start"]].strip()
            if preamble:
                sections.append(Section(
                    title="preamble",
                    content=preamble,
                    level=0,
                    start_pos=0,
                    end_pos=headers[0]["start"],
                    section_type="preamble",
                ))
        
        # Track parent hierarchy
        parent_stack = []  # [(level, title), ...]
        
        # Extract sections between headers
        for i, header in enumerate(headers):
            # Determine content end
            if i + 1 < len(headers):
                content_end = headers[i + 1]["start"]
            else:
                content_end = len(content)
            
            section_content = content[header["header_end"]:content_end].strip()
            
            # Update parent stack
            while parent_stack and parent_stack[-1][0] >= header["level"]:
                parent_stack.pop()
            
            parent_title = parent_stack[-1][1] if parent_stack else None
            
            sections.append(Section(
                title=header["title"],
                content=section_content,
                level=header["level"],
                start_pos=header["start"],
                end_pos=content_end,
                section_type=f"h{header['level']}",
                parent_title=parent_title,
            ))
            
            parent_stack.append((header["level"], header["title"]))
        
        return sections
    
    def get_fallback_separators(self) -> List[str]:
        """Markdown-appropriate separators."""
        return ["\n\n", "\n", ". ", " "]
