"""Recursive text chunker - splits on natural boundaries."""

import logging
from typing import List, Optional

from src.chunkers.base import BaseChunker, Chunk
from src.chunkers.factory import register_chunker
from src.loaders.base import Document

logger = logging.getLogger(__name__)


@register_chunker("recursive")
class RecursiveChunker(BaseChunker):
    """
    Recursively splits text on natural boundaries.
    
    Tries separators in order (paragraphs -> newlines -> sentences -> words)
    until chunks are small enough.
    
    Usage:
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(document)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            separators: List of separators to try, in order
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document recursively on natural boundaries.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunks
        """
        text = document.content
        
        if not text.strip():
            return []
        
        # Split recursively
        split_texts = self._split_text(text, self.separators)
        
        # Merge small pieces and create chunks with overlap
        merged_texts = self._merge_splits(split_texts)
        
        chunks = []
        for i, chunk_text in enumerate(merged_texts):
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                }
            ))
        
        logger.debug(f"Created {len(chunks)} chunks from {document.metadata.get('filename', 'unknown')}")
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            # No more separators, just return text as-is
            return [text] if text else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Last resort: split by character
            return list(text)
        
        splits = text.split(separator)
        
        result = []
        for split in splits:
            if not split:
                continue
            
            # Add separator back (except for last split)
            split_with_sep = split + separator if separator != " " else split + " "
            
            if len(split_with_sep) <= self.chunk_size:
                result.append(split_with_sep.rstrip())
            else:
                # Too big, recurse with next separator
                result.extend(self._split_text(split, remaining_separators))
        
        return result

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits into chunks respecting size and overlap."""
        if not splits:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_length + split_len <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_len + 1  # +1 for space
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last pieces that fit in overlap
                    overlap_chunk = []
                    overlap_length = 0
                    for piece in reversed(current_chunk):
                        if overlap_length + len(piece) <= self.chunk_overlap:
                            overlap_chunk.insert(0, piece)
                            overlap_length += len(piece) + 1
                        else:
                            break
                    current_chunk = overlap_chunk + [split]
                    current_length = sum(len(p) for p in current_chunk) + len(current_chunk)
                else:
                    current_chunk = [split]
                    current_length = split_len
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
