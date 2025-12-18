"""Structure-aware chunker - respects document boundaries and sections."""

import logging
from typing import List, Optional, Dict, Type

from src.chunkers.base import BaseChunker, Chunk
from src.chunkers.factory import register_chunker
from src.chunkers.recursive_chunker import RecursiveChunker
from src.chunkers.patterns.base import BaseDocumentPattern, Section
from src.chunkers.patterns.markdown import MarkdownPattern
from src.chunkers.patterns.sec_filing import SECFilingPattern
from src.loaders.base import Document

logger = logging.getLogger(__name__)


# Pattern registry
PATTERN_REGISTRY: Dict[str, Type[BaseDocumentPattern]] = {
    "markdown": MarkdownPattern,
    "sec_filing": SECFilingPattern,
}


def get_available_patterns() -> List[str]:
    """Return list of available pattern names."""
    return list(PATTERN_REGISTRY.keys())


@register_chunker("structure_aware")
class StructureAwareChunker(BaseChunker):
    """
    Chunks documents based on logical structure.
    
    Flow:
    1. Auto-detect document type using patterns
    2. Extract sections using matched pattern
    3. Chunk within sections (respecting boundaries)
    4. Fall back to recursive chunking if no structure found
    
    Features:
    - Pluggable pattern system
    - Section-aware metadata
    - Parent-child chunk generation (optional)
    - Preserves document hierarchy
    
    Usage:
        chunker = StructureAwareChunker(
            chunk_size=1024,
            patterns=["sec_filing", "markdown"],
        )
        chunks = chunker.chunk(document)
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        patterns: Optional[List[str]] = None,
        min_detection_confidence: float = 0.3,
        fallback_strategy: str = "recursive",
        generate_parent_chunks: bool = False,
        parent_chunk_size: int = 2048,
        min_section_size: int = 100,
    ):
        """
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks within same section
            patterns: List of pattern names to try (default: all)
            min_detection_confidence: Minimum confidence to use a pattern
            fallback_strategy: Strategy when no pattern matches
            generate_parent_chunks: Create parent chunks for retrieval
            parent_chunk_size: Size of parent chunks
            min_section_size: Minimum section size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_detection_confidence = min_detection_confidence
        self.fallback_strategy = fallback_strategy
        self.generate_parent_chunks = generate_parent_chunks
        self.parent_chunk_size = parent_chunk_size
        self.min_section_size = min_section_size
        
        # Initialize patterns
        pattern_names = patterns or list(PATTERN_REGISTRY.keys())
        self.patterns: List[BaseDocumentPattern] = []
        for name in pattern_names:
            if name in PATTERN_REGISTRY:
                self.patterns.append(PATTERN_REGISTRY[name]())
            else:
                logger.warning(f"Unknown pattern: {name}, skipping")
        
        # Fallback chunker
        self._fallback_chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        logger.info(
            f"StructureAwareChunker initialized with patterns: "
            f"{[p.name for p in self.patterns]}"
        )
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk document using structure-aware approach.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunks with section metadata
        """
        content = document.content
        metadata = document.metadata
        
        if not content.strip():
            return []
        
        # Detect document type
        pattern, confidence = self._detect_pattern(content, metadata)
        
        if pattern and confidence >= self.min_detection_confidence:
            logger.info(
                f"Detected pattern '{pattern.name}' with confidence {confidence:.2f} "
                f"for {metadata.get('source', 'unknown')}"
            )
            return self._chunk_with_pattern(document, pattern)
        else:
            logger.info(
                f"No pattern matched (best: {confidence:.2f}), "
                f"using fallback for {metadata.get('source', 'unknown')}"
            )
            return self._chunk_fallback(document)
    
    def _detect_pattern(
        self, 
        content: str, 
        metadata: dict
    ) -> tuple[Optional[BaseDocumentPattern], float]:
        """Detect best matching pattern for content."""
        best_pattern = None
        best_confidence = 0.0
        
        for pattern in self.patterns:
            try:
                confidence = pattern.detect(content, metadata)
                logger.debug(f"Pattern '{pattern.name}' confidence: {confidence:.2f}")
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_pattern = pattern
            except Exception as e:
                logger.warning(f"Pattern '{pattern.name}' detection failed: {e}")
        
        return best_pattern, best_confidence
    
    def _chunk_with_pattern(
        self, 
        document: Document, 
        pattern: BaseDocumentPattern
    ) -> List[Chunk]:
        """Chunk document using detected pattern."""
        chunks = []
        
        # Extract sections
        try:
            sections = pattern.extract_sections(document.content)
        except Exception as e:
            logger.error(f"Section extraction failed: {e}, falling back")
            return self._chunk_fallback(document)
        
        if not sections:
            logger.warning("No sections extracted, falling back")
            return self._chunk_fallback(document)
        
        logger.info(f"Extracted {len(sections)} sections")
        
        # Process each section
        for section in sections:
            section_chunks = self._chunk_section(
                section=section,
                document=document,
                pattern=pattern,
            )
            chunks.extend(section_chunks)
        
        # Generate parent chunks if enabled
        if self.generate_parent_chunks:
            chunks = self._add_parent_chunks(chunks, document)
        
        logger.info(
            f"Created {len(chunks)} chunks from {len(sections)} sections"
        )
        return chunks
    
    def _chunk_section(
        self,
        section: Section,
        document: Document,
        pattern: BaseDocumentPattern,
    ) -> List[Chunk]:
        """Chunk a single section, respecting boundaries."""
        section_chunks = []
        content = section.content
        
        # Skip tiny sections
        if len(content) < self.min_section_size:
            return []
        
        # If section fits in one chunk, keep it together
        if len(content) <= self.chunk_size:
            chunk = Chunk(
                content=content,
                metadata={
                    **document.metadata,
                    "pattern": pattern.name,
                    "section_type": section.section_type,
                },
                section=section.title,
                section_hierarchy=section.hierarchy,
                chunk_type="content",
            )
            section_chunks.append(chunk)
        else:
            # Split section using recursive chunking
            sub_chunks = self._split_large_section(content, pattern)
            
            for i, sub_content in enumerate(sub_chunks):
                chunk = Chunk(
                    content=sub_content,
                    metadata={
                        **document.metadata,
                        "pattern": pattern.name,
                        "section_type": section.section_type,
                        "section_chunk_index": i,
                    },
                    section=section.title,
                    section_hierarchy=section.hierarchy,
                    chunk_type="content",
                )
                section_chunks.append(chunk)
        
        return section_chunks
    
    def _split_large_section(
        self, 
        content: str, 
        pattern: BaseDocumentPattern
    ) -> List[str]:
        """Split large section content respecting natural boundaries."""
        separators = pattern.get_fallback_separators()
        
        # Use recursive splitting logic
        splits = self._recursive_split(content, separators)
        
        # Merge splits respecting chunk_size
        return self._merge_splits(splits)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text on separators."""
        if not separators:
            return [text] if text else []
        
        separator = separators[0]
        remaining = separators[1:]
        
        if separator == "":
            return list(text)
        
        parts = text.split(separator)
        result = []
        
        for part in parts:
            if not part.strip():
                continue
            
            if len(part) <= self.chunk_size:
                result.append(part.strip())
            else:
                result.extend(self._recursive_split(part, remaining))
        
        return result
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits into chunks."""
        if not splits:
            return []
        
        chunks = []
        current = []
        current_len = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_len + split_len + 1 <= self.chunk_size:
                current.append(split)
                current_len += split_len + 1
            else:
                if current:
                    chunks.append(" ".join(current))
                
                # Handle overlap
                if self.chunk_overlap > 0 and current:
                    overlap = []
                    overlap_len = 0
                    for piece in reversed(current):
                        if overlap_len + len(piece) <= self.chunk_overlap:
                            overlap.insert(0, piece)
                            overlap_len += len(piece) + 1
                        else:
                            break
                    current = overlap + [split]
                    current_len = sum(len(p) for p in current) + len(current)
                else:
                    current = [split]
                    current_len = split_len
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    def _add_parent_chunks(
        self, 
        chunks: List[Chunk], 
        document: Document
    ) -> List[Chunk]:
        """Add parent chunks for parent-child retrieval."""
        if not chunks:
            return chunks
        
        # Group chunks by section
        section_chunks: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            section = chunk.section or "default"
            if section not in section_chunks:
                section_chunks[section] = []
            section_chunks[section].append(chunk)
        
        result = []
        
        for section, sec_chunks in section_chunks.items():
            # Create parent chunk from section content
            combined_content = " ".join(c.content for c in sec_chunks)
            
            if len(combined_content) <= self.parent_chunk_size:
                # Section is small enough to be its own parent
                parent = Chunk(
                    content=combined_content,
                    metadata=sec_chunks[0].metadata.copy(),
                    section=section,
                    section_hierarchy=sec_chunks[0].section_hierarchy,
                    chunk_type="parent",
                )
                parent.metadata["is_parent"] = True
                result.append(parent)
                
                # Link children to parent
                for child in sec_chunks:
                    child.parent_id = parent.chunk_id
                    child.chunk_type = "child"
                    result.append(child)
            else:
                # Section too large, create multiple parents
                # For now, just add chunks without parents
                result.extend(sec_chunks)
        
        return result
    
    def _chunk_fallback(self, document: Document) -> List[Chunk]:
        """Fall back to recursive chunking."""
        return self._fallback_chunker.chunk(document)
