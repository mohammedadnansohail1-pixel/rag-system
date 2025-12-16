"""Fixed size text chunker."""

import logging
from typing import List

from src.chunkers.base import BaseChunker, Chunk
from src.chunkers.factory import register_chunker
from src.loaders.base import Document

logger = logging.getLogger(__name__)


@register_chunker("fixed")
class FixedChunker(BaseChunker):
    """
    Splits text into fixed-size chunks with overlap.
    
    Usage:
        chunker = FixedChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(document)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into fixed-size chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunks
        """
        text = document.content
        
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": min(end, len(text)),
                }
            ))
            
            chunk_index += 1
            start += self.chunk_size - self.chunk_overlap
        
        logger.debug(f"Created {len(chunks)} chunks from {document.metadata.get('filename', 'unknown')}")
        return chunks
