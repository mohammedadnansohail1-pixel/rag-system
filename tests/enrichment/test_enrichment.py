"""Tests for enrichment module."""

import pytest
from src.enrichment import (
    EnrichmentPipeline,
    EnrichmentConfig,
    EntityExtractor,
    TopicExtractor,
    EnrichmentResult,
)
from src.chunkers.base import Chunk


class TestEntityExtractor:
    """Test entity extraction."""
    
    def test_extracts_money(self):
        """Extracts money amounts."""
        extractor = EntityExtractor()
        result = extractor.enrich("Revenue was $134.9 billion in 2024")
        
        assert "money" in result.entities
        assert any("134" in m for m in result.entities["money"])
    
    def test_extracts_percentages(self):
        """Extracts percentages."""
        extractor = EntityExtractor()
        result = extractor.enrich("Growth increased 15% year over year")
        
        assert "percentage" in result.entities
        assert "15%" in result.entities["percentage"]
    
    def test_extracts_dates(self):
        """Extracts dates."""
        extractor = EntityExtractor()
        result = extractor.enrich("As of December 31, 2024, we had 72,000 employees")
        
        assert "date" in result.entities
        assert "2024" in result.entities["date"] or "December 31, 2024" in result.entities["date"]
    
    def test_extracts_organizations(self):
        """Extracts organization names."""
        extractor = EntityExtractor()
        result = extractor.enrich("The FTC investigation into Meta continues")
        
        assert "organization" in result.entities
        assert "FTC" in result.entities["organization"]
        assert "Meta" in result.entities["organization"]
    
    def test_handles_empty_content(self):
        """Handles empty content gracefully."""
        extractor = EntityExtractor()
        result = extractor.enrich("")
        
        assert result.entities == {}


class TestTopicExtractor:
    """Test topic extraction."""
    
    def test_extracts_finance_topics(self):
        """Extracts finance-related topics."""
        extractor = TopicExtractor()
        result = extractor.enrich("Revenue growth and profit margins improved significantly")
        
        assert len(result.topics) > 0
        assert any(t in result.topics for t in ["revenue", "growth", "profit", "margin"])
    
    def test_extracts_legal_topics(self):
        """Extracts legal topics."""
        extractor = TopicExtractor()
        result = extractor.enrich("The litigation and regulatory investigation continues")
        
        assert any(t in result.topics for t in ["litigation", "regulatory", "legal"])
    
    def test_extracts_keywords(self):
        """Extracts relevant keywords."""
        extractor = TopicExtractor()
        result = extractor.enrich("The cybersecurity program protects our infrastructure")
        
        assert len(result.keywords) > 0
    
    def test_detects_categories(self):
        """Detects topic categories."""
        extractor = TopicExtractor()
        result = extractor.enrich("Revenue increased due to advertising growth in our platform")
        
        categories = result.metadata.get("topic_categories", [])
        assert "finance" in categories or "technology" in categories


class TestEnrichmentResult:
    """Test EnrichmentResult dataclass."""
    
    def test_to_dict(self):
        """Converts to dictionary."""
        result = EnrichmentResult(
            entities={"money": ["$100"]},
            topics=["revenue"],
            keywords=["growth"],
        )
        
        d = result.to_dict()
        assert d["entities"] == {"money": ["$100"]}
        assert d["topics"] == ["revenue"]
        assert d["keywords"] == ["growth"]
    
    def test_merge(self):
        """Merges two results."""
        result1 = EnrichmentResult(
            entities={"money": ["$100"]},
            topics=["revenue"],
        )
        result2 = EnrichmentResult(
            entities={"date": ["2024"]},
            topics=["growth"],
            summary="Test summary",
        )
        
        merged = result1.merge(result2)
        
        assert "money" in merged.entities
        assert "date" in merged.entities
        assert "revenue" in merged.topics
        assert "growth" in merged.topics
        assert merged.summary == "Test summary"


class TestEnrichmentConfig:
    """Test enrichment configurations."""
    
    def test_fast_config(self):
        """Fast config has no LLM."""
        config = EnrichmentConfig.fast()
        
        assert config.extract_entities is True
        assert config.extract_topics is True
        assert config.use_llm is False
    
    def test_full_config(self):
        """Full config includes LLM."""
        config = EnrichmentConfig.full()
        
        assert config.extract_entities is True
        assert config.extract_topics is True
        assert config.use_llm is True
        assert config.generate_summary is True
    
    def test_minimal_config(self):
        """Minimal config has limited extraction."""
        config = EnrichmentConfig.minimal()
        
        assert config.extract_entities is True
        assert config.extract_topics is False
        assert config.use_llm is False


class TestEnrichmentPipeline:
    """Test enrichment pipeline."""
    
    def test_creates_with_fast_config(self):
        """Creates pipeline with fast config."""
        pipeline = EnrichmentPipeline(config=EnrichmentConfig.fast())
        
        assert len(pipeline.enrichers) == 2  # entity + topic
    
    def test_enrich_content(self):
        """Enriches text content."""
        pipeline = EnrichmentPipeline(config=EnrichmentConfig.fast())
        result = pipeline.enrich("Revenue grew 15% to $134 billion in 2024")
        
        assert result.entities.get("money")
        assert result.entities.get("percentage")
    
    def test_enrich_chunk(self):
        """Enriches a chunk object."""
        pipeline = EnrichmentPipeline(config=EnrichmentConfig.fast())
        
        chunk = Chunk(
            content="Revenue grew 15% to $134 billion in 2024",
            metadata={"source": "test"},
            section="Item 7 - MD&A",
            section_hierarchy=["Part II", "Item 7"],
        )
        
        enriched = pipeline.enrich_chunk(chunk)
        
        # Original fields preserved
        assert enriched.content == chunk.content
        assert enriched.section == "Item 7 - MD&A"
        
        # Section in metadata
        assert enriched.metadata.get("section") == "Item 7 - MD&A"
        
        # Enrichment added
        assert enriched.metadata.get("entities")
    
    def test_enrich_chunks_batch(self):
        """Enriches multiple chunks."""
        pipeline = EnrichmentPipeline(config=EnrichmentConfig.fast())
        
        chunks = [
            Chunk(content="Revenue was $100 billion", metadata={}, section="Business"),
            Chunk(content="Legal risks include litigation", metadata={}, section="Risk Factors"),
        ]
        
        enriched = pipeline.enrich_chunks(chunks, show_progress=False)
        
        assert len(enriched) == 2
        assert all(c.metadata.get("section") for c in enriched)
    
    def test_from_config_dict(self):
        """Creates from config dictionary."""
        pipeline = EnrichmentPipeline.from_config({
            "extract_entities": True,
            "extract_topics": False,
            "use_llm": False,
        })
        
        assert len(pipeline.enrichers) == 1  # Only entity extractor
    
    def test_get_stats(self):
        """Gets enrichment statistics."""
        pipeline = EnrichmentPipeline(config=EnrichmentConfig.fast())
        
        chunks = [
            Chunk(content="Revenue was $100 billion", metadata={}, section="Business"),
        ]
        enriched = pipeline.enrich_chunks(chunks, show_progress=False)
        
        stats = pipeline.get_stats(enriched)
        
        assert stats["total_chunks"] == 1
        assert "entity_types" in stats
        assert "sections" in stats


class TestRealWorldEnrichment:
    """Test with real SEC filing data if available."""
    
    def test_sec_filing_enrichment(self):
        """Test enrichment on SEC filing chunks."""
        from pathlib import Path
        from src.loaders.base import Document
        from src.chunkers import StructureAwareChunker
        
        filing_path = Path("data/test_adaptive/sec-edgar-filings/META/10-K")
        if not filing_path.exists():
            pytest.skip("META 10-K data not available")
        
        filing_dirs = list(filing_path.glob("*"))
        if not filing_dirs:
            pytest.skip("No filing directory found")
        
        full_sub = filing_dirs[0] / "full-submission.txt"
        if not full_sub.exists():
            pytest.skip("full-submission.txt not found")
        
        # Load and chunk
        content = full_sub.read_text(encoding='utf-8', errors='ignore')
        doc = Document(content=content, metadata={"source": "META", "type": "10-K"})
        
        chunker = StructureAwareChunker(chunk_size=1500)
        chunks = chunker.chunk(doc)
        content_chunks = [c for c in chunks if c.section and 'Item' in c.section][:20]
        
        # Enrich
        pipeline = EnrichmentPipeline(config=EnrichmentConfig.fast())
        enriched = pipeline.enrich_chunks(content_chunks, show_progress=False)
        
        # Verify enrichment
        stats = pipeline.get_stats(enriched)
        
        assert stats["chunks_with_entities"] > 10  # Most should have entities
        assert stats["chunks_with_topics"] > 10    # Most should have topics
        assert len(stats["sections"]) >= 3          # Multiple sections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
