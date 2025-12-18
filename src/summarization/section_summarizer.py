"""Generate summaries for document sections."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import defaultdict

from src.chunkers.base import Chunk
from src.generation.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class SectionSummary:
    """Summary of a document section."""
    
    section_name: str
    summary: str
    key_points: List[str] = field(default_factory=list)
    chunk_count: int = 0
    total_chars: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_chunk(self) -> Chunk:
        """Convert to a Chunk for storage/retrieval."""
        content = f"[SECTION SUMMARY: {self.section_name}]\n\n{self.summary}"
        
        if self.key_points:
            content += "\n\nKey Points:\n" + "\n".join(f"- {p}" for p in self.key_points)
        
        return Chunk(
            content=content,
            metadata={
                "section": self.section_name,
                "chunk_type": "summary",
                "is_summary": True,
                "chunk_count": self.chunk_count,
                "total_chars": self.total_chars,
                **self.metadata,
            },
            chunk_type="summary",
            section=self.section_name,
        )


class SectionSummarizer:
    """
    Generate hierarchical summaries for document sections.
    
    Creates concise summaries for each major section, enabling:
    - Quick overview retrieval
    - Better handling of "summarize" queries
    - Drill-down from summary to detail
    
    Usage:
        summarizer = SectionSummarizer(llm=llm)
        summaries = summarizer.summarize_sections(chunks)
        
        # Convert to chunks for storage
        summary_chunks = [s.to_chunk() for s in summaries]
    """
    
    SUMMARY_PROMPT = """Summarize the following content from the "{section}" section of a document.
Provide a concise summary (2-3 paragraphs) capturing the key information.

Content:
{content}

Summary:"""

    KEY_POINTS_PROMPT = """Extract 3-5 key points from the following content.
Return only the key points, one per line.

Content:
{content}

Key Points:"""

    def __init__(
        self,
        llm: BaseLLM,
        max_content_for_summary: int = 8000,
        min_chunks_for_summary: int = 3,
        generate_key_points: bool = True,
    ):
        """
        Args:
            llm: LLM for generating summaries
            max_content_for_summary: Max chars to send to LLM
            min_chunks_for_summary: Min chunks needed to create summary
            generate_key_points: Also extract key points
        """
        self.llm = llm
        self.max_content_for_summary = max_content_for_summary
        self.min_chunks_for_summary = min_chunks_for_summary
        self.generate_key_points = generate_key_points
        
        logger.info(f"Initialized SectionSummarizer: max_content={max_content_for_summary}")
    
    def _match_section(self, section_name: str, target: str) -> bool:
        """Check if section_name matches target (partial match)."""
        # Normalize both
        section_lower = section_name.lower().strip()
        target_lower = target.lower().strip()
        
        # Exact match
        if section_lower == target_lower:
            return True
        
        # Section starts with target
        if section_lower.startswith(target_lower):
            return True
        
        # Target is contained in section
        if target_lower in section_lower:
            return True
        
        return False
    
    def _group_chunks_by_section(
        self, 
        chunks: List[Chunk],
        sections_to_summarize: Optional[List[str]] = None,
    ) -> Dict[str, List[Chunk]]:
        """Group chunks by section with partial matching."""
        sections = defaultdict(list)
        
        for chunk in chunks:
            if not chunk.section:
                continue
            
            if sections_to_summarize:
                # Find matching target section
                for target in sections_to_summarize:
                    if self._match_section(chunk.section, target):
                        # Use the target name (normalized)
                        sections[target].append(chunk)
                        break
            else:
                # Use actual section name
                sections[chunk.section].append(chunk)
        
        return dict(sections)
    
    def summarize_sections(
        self,
        chunks: List[Chunk],
        sections_to_summarize: Optional[List[str]] = None,
    ) -> List[SectionSummary]:
        """
        Generate summaries for document sections.
        
        Args:
            chunks: All document chunks
            sections_to_summarize: Specific sections (None = all major sections)
            
        Returns:
            List of SectionSummary objects
        """
        # Group chunks by section
        sections = self._group_chunks_by_section(chunks, sections_to_summarize)
        
        # Filter to sections with enough content
        sections = {
            k: v for k, v in sections.items()
            if len(v) >= self.min_chunks_for_summary
        }
        
        logger.info(f"Generating summaries for {len(sections)} sections: {list(sections.keys())}")
        
        summaries = []
        for section_name, section_chunks in sections.items():
            try:
                summary = self._summarize_section(section_name, section_chunks)
                summaries.append(summary)
                logger.info(f"Generated summary for: {section_name[:40]}")
            except Exception as e:
                logger.warning(f"Failed to summarize {section_name}: {e}")
        
        return summaries
    
    def _summarize_section(
        self,
        section_name: str,
        chunks: List[Chunk],
    ) -> SectionSummary:
        """Generate summary for a single section."""
        # Combine chunk content
        combined_content = "\n\n".join(c.content for c in chunks)
        total_chars = len(combined_content)
        
        # Truncate if needed (take from beginning and end)
        if len(combined_content) > self.max_content_for_summary:
            half = self.max_content_for_summary // 2
            combined_content = (
                combined_content[:half] + 
                "\n\n[...content truncated...]\n\n" + 
                combined_content[-half:]
            )
        
        # Generate summary
        prompt = self.SUMMARY_PROMPT.format(
            section=section_name,
            content=combined_content,
        )
        
        summary_text = self.llm.generate(
            prompt,
            system_prompt="You are a document summarizer. Be concise and factual."
        )
        
        # Generate key points
        key_points = []
        if self.generate_key_points:
            key_points = self._extract_key_points(combined_content)
        
        # Collect metadata from chunks
        topics = set()
        entities = {}
        for chunk in chunks:
            if chunk.metadata.get("topics"):
                topics.update(chunk.metadata["topics"])
            if chunk.metadata.get("entities"):
                for etype, values in chunk.metadata["entities"].items():
                    if etype not in entities:
                        entities[etype] = set()
                    entities[etype].update(values)
        
        # Convert entity sets to lists
        entities = {k: list(v)[:10] for k, v in entities.items()}
        
        return SectionSummary(
            section_name=section_name,
            summary=summary_text.strip(),
            key_points=key_points,
            chunk_count=len(chunks),
            total_chars=total_chars,
            metadata={
                "topics": list(topics)[:10],
                "entities": entities,
            }
        )
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Use shorter content for key points
        truncated = content[:self.max_content_for_summary // 2]
        
        prompt = self.KEY_POINTS_PROMPT.format(content=truncated)
        
        response = self.llm.generate(
            prompt,
            system_prompt="Extract key points only. Be concise."
        )
        
        # Parse response into list
        points = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove common prefixes
            for prefix in ["- ", "• ", "* ", "1.", "2.", "3.", "4.", "5."]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line and len(line) > 10:
                points.append(line)
        
        return points[:5]
