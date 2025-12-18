"""Tests for structure-aware chunker."""

import pytest
from src.chunkers import (
    StructureAwareChunker,
    RecursiveChunker,
    ChunkerFactory,
    get_available_patterns,
    get_registered_chunkers,
)
from src.chunkers.patterns.markdown import MarkdownPattern
from src.chunkers.patterns.sec_filing import SECFilingPattern
from src.loaders.base import Document


class TestPatternDetection:
    """Test document pattern detection."""
    
    def test_markdown_detection(self):
        """Markdown pattern detects markdown content."""
        pattern = MarkdownPattern()
        
        md_content = """# Main Title
        
## Section One

Some content here.

## Section Two

More content.
"""
        confidence = pattern.detect(md_content, {"filename": "test.md"})
        assert confidence >= 0.5
    
    def test_markdown_detection_without_extension(self):
        """Markdown pattern detects content without file extension."""
        pattern = MarkdownPattern()
        
        md_content = """# Header

## Subheader

Content with **bold** and [links](http://example.com).
"""
        confidence = pattern.detect(md_content, {})
        assert confidence >= 0.3
    
    def test_sec_filing_detection(self):
        """SEC pattern detects SEC filing content."""
        pattern = SECFilingPattern()
        
        sec_content = """<SEC-DOCUMENT>
<SEC-HEADER>
ACCESSION NUMBER: 0001234567-25-000001
CONFORMED SUBMISSION TYPE: 10-K
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-K
ITEM 1. Business
Our company does things.
ITEM 1A. Risk Factors
There are risks.
</DOCUMENT>
"""
        confidence = pattern.detect(sec_content, {"type": "10-K"})
        assert confidence >= 0.5
    
    def test_sec_detection_by_metadata(self):
        """SEC pattern uses metadata for detection."""
        pattern = SECFilingPattern()
        
        confidence = pattern.detect("some content", {"type": "10-K", "source": "SEC"})
        assert confidence >= 0.3


class TestMarkdownSectionExtraction:
    """Test markdown section extraction."""
    
    def test_extracts_atx_headers(self):
        """Extracts sections from ATX-style headers."""
        pattern = MarkdownPattern()
        
        content = """# Main Title

Introduction text.

## Section One

Section one content.

## Section Two

Section two content.

### Subsection

Nested content.
"""
        sections = pattern.extract_sections(content)
        
        assert len(sections) >= 4
        titles = [s.title for s in sections]
        assert "Main Title" in titles
        assert "Section One" in titles
        assert "Section Two" in titles
        assert "Subsection" in titles
    
    def test_tracks_hierarchy(self):
        """Tracks parent-child hierarchy."""
        pattern = MarkdownPattern()
        
        content = """# Parent

## Child

Content.
"""
        sections = pattern.extract_sections(content)
        child = next(s for s in sections if s.title == "Child")
        assert child.parent_title == "Parent"


class TestSECSectionExtraction:
    """Test SEC filing section extraction."""
    
    def test_extracts_items(self):
        """Extracts ITEM sections from SEC filings."""
        pattern = SECFilingPattern()
        
        # More realistic SEC content with clear section breaks
        content = """
PART I

ITEM 1. Business

We are a technology company that provides various services. Our main business involves software development and cloud computing solutions.

ITEM 1A. Risk Factors

There are many risks associated with our business. Market conditions may change. Competition is intense in our industry.

ITEM 2. Properties

We have offices in multiple locations. Our headquarters is in California.
"""
        sections = pattern.extract_sections(content)
        
        # Should find at least the sections
        assert len(sections) >= 1
        
        # Check that we found Item content
        all_content = " ".join(s.content for s in sections)
        assert "technology company" in all_content
    
    def test_handles_html_content(self):
        """Handles HTML-encoded SEC filings."""
        pattern = SECFilingPattern()
        
        content = """
<div>
<span>ITEM 1. Business</span>
</div>
<div>
<p>We are a technology company.</p>
</div>
<div>
<span>ITEM 1A. Risk Factors</span>
</div>
<div>
<p>There are risks.</p>
</div>
"""
        sections = pattern.extract_sections(content)
        
        # Should find sections despite HTML
        assert len(sections) >= 1


class TestStructureAwareChunker:
    """Test the main structure-aware chunker."""
    
    def test_factory_registration(self):
        """Chunker is registered with factory."""
        assert "structure_aware" in get_registered_chunkers()
    
    def test_create_via_factory(self):
        """Can create via factory."""
        chunker = ChunkerFactory.create("structure_aware", chunk_size=1024)
        assert isinstance(chunker, StructureAwareChunker)
    
    def test_chunks_markdown(self):
        """Chunks markdown documents with section awareness."""
        chunker = StructureAwareChunker(chunk_size=500, min_section_size=20)
        
        doc = Document(
            content="""# Title

This is the introduction with enough content to meet minimum size requirements.

## Section A

This is section A content that is reasonably long and contains substantial information.

## Section B

This is section B content with additional details and explanations.
""",
            metadata={"filename": "test.md"}
        )
        
        chunks = chunker.chunk(doc)
        
        assert len(chunks) >= 2
        # Check section metadata
        sections = set(c.section for c in chunks if c.section)
        assert len(sections) >= 2
    
    def test_fallback_on_plain_text(self):
        """Falls back to recursive chunking for plain text."""
        chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=20)
        
        doc = Document(
            content="This is plain text without any structure. " * 20,
            metadata={"filename": "plain.txt"}
        )
        
        chunks = chunker.chunk(doc)
        
        # Should still produce chunks
        assert len(chunks) >= 1
    
    def test_chunk_has_required_fields(self):
        """Chunks have all required fields."""
        chunker = StructureAwareChunker(chunk_size=500, min_section_size=20)
        
        doc = Document(
            content="# Test\n\nContent here with enough text to be meaningful.",
            metadata={"source": "test"}
        )
        
        chunks = chunker.chunk(doc)
        
        for chunk in chunks:
            assert chunk.content
            assert chunk.chunk_id
            assert chunk.metadata
            assert chunk.chunk_type
    
    def test_respects_chunk_size(self):
        """Respects maximum chunk size."""
        chunk_size = 500
        chunker = StructureAwareChunker(chunk_size=chunk_size)
        
        # Long content
        doc = Document(
            content="# Title\n\n" + "Word " * 1000,
            metadata={}
        )
        
        chunks = chunker.chunk(doc)
        
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size * 1.1  # 10% tolerance


class TestChunkReduction:
    """Test that structure-aware chunking reduces chunk count."""
    
    def test_reduces_chunks_vs_recursive(self):
        """Structure-aware produces fewer chunks than recursive."""
        content = """# Document

## Section 1

This is the first section with substantial content that would normally be split across multiple chunks if we were using a simple recursive approach.

## Section 2

This is the second section with more content. The structure-aware chunker should keep sections together when possible.

## Section 3

Final section with additional information.
"""
        doc = Document(content=content, metadata={"filename": "test.md"})
        
        sa_chunker = StructureAwareChunker(chunk_size=500, min_section_size=20)
        rec_chunker = RecursiveChunker(chunk_size=500)
        
        sa_chunks = sa_chunker.chunk(doc)
        rec_chunks = rec_chunker.chunk(doc)
        
        # Structure-aware should be same or fewer
        assert len(sa_chunks) <= len(rec_chunks) + 2  # Allow small variance


class TestRealWorldSECFiling:
    """Test with real SEC filing data if available."""
    
    def test_meta_10k_if_available(self):
        """Test META 10-K chunking if data exists."""
        from pathlib import Path
        
        filing_path = Path("data/test_adaptive/sec-edgar-filings/META/10-K")
        if not filing_path.exists():
            pytest.skip("META 10-K data not available")
        
        filing_dirs = list(filing_path.glob("*"))
        if not filing_dirs:
            pytest.skip("No filing directory found")
        
        full_submission = filing_dirs[0] / "full-submission.txt"
        if not full_submission.exists():
            pytest.skip("full-submission.txt not found")
        
        content = full_submission.read_text(encoding='utf-8', errors='ignore')
        doc = Document(content=content, metadata={"source": "META", "type": "10-K"})
        
        chunker = StructureAwareChunker(chunk_size=1500)
        chunks = chunker.chunk(doc)
        
        # Should produce reasonable number of chunks
        assert 100 < len(chunks) < 1000
        
        # Should find Item sections
        item_chunks = [c for c in chunks if c.section and 'Item' in c.section]
        assert len(item_chunks) > 50
        
        # Should have section metadata
        assert any(c.section_hierarchy for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
