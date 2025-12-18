"""Tests for hierarchical summarization."""

import pytest
from src.summarization import SectionSummary, HierarchicalRetriever
from src.retrieval.base import RetrievalResult


class TestSectionSummary:
    """Test SectionSummary dataclass."""
    
    def test_to_chunk(self):
        """Converts to chunk correctly."""
        summary = SectionSummary(
            section_name="Item 1 - Business",
            summary="This is the business summary.",
            key_points=["Point 1", "Point 2"],
            chunk_count=10,
            total_chars=5000,
        )
        
        chunk = summary.to_chunk()
        
        assert "[SECTION SUMMARY: Item 1 - Business]" in chunk.content
        assert "This is the business summary." in chunk.content
        assert "Point 1" in chunk.content
        assert chunk.metadata["is_summary"] is True
        assert chunk.metadata["chunk_type"] == "summary"
        assert chunk.chunk_type == "summary"


class MockBaseRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, results=None):
        self.results = results or []
    
    def add_documents(self, texts, metadatas=None, **kwargs):
        pass
    
    def retrieve(self, query, top_k=5, **kwargs):
        return self.results[:top_k]
    
    def health_check(self):
        return True


class TestHierarchicalRetriever:
    """Test HierarchicalRetriever."""
    
    def test_detects_summary_query(self):
        """Detects summary-type queries."""
        base = MockBaseRetriever()
        retriever = HierarchicalRetriever(base_retriever=base)
        
        assert retriever._is_summary_query("Summarize the risk factors")
        assert retriever._is_summary_query("Give me an overview")
        assert retriever._is_summary_query("What are the main points?")
        assert not retriever._is_summary_query("What was the revenue?")
        assert not retriever._is_summary_query("How many employees?")
    
    def test_boosts_summary_scores(self):
        """Boosts summary chunk scores."""
        summary_result = RetrievalResult(
            content="Summary content",
            metadata={"is_summary": True, "section": "Item 1"},
            score=0.5,
        )
        detail_result = RetrievalResult(
            content="Detail content",
            metadata={"is_summary": False, "section": "Item 1"},
            score=0.6,
        )
        
        base = MockBaseRetriever(results=[detail_result, summary_result])
        retriever = HierarchicalRetriever(
            base_retriever=base,
            summary_boost=1.5,
        )
        
        results = retriever.retrieve("Summarize Item 1", top_k=2)
        
        # Summary should be boosted above detail
        assert results[0].metadata.get("is_summary") is True
        assert results[0].score == 0.75  # 0.5 * 1.5
    
    def test_tracks_summary_sections(self):
        """Tracks which sections have summaries."""
        base = MockBaseRetriever()
        retriever = HierarchicalRetriever(base_retriever=base)
        
        metadatas = [
            {"is_summary": True, "section": "Item 1"},
            {"is_summary": False, "section": "Item 1"},
            {"is_summary": True, "section": "Item 2"},
        ]
        
        retriever.add_documents(texts=["a", "b", "c"], metadatas=metadatas)
        
        assert "Item 1" in retriever.sections_with_summaries
        assert "Item 2" in retriever.sections_with_summaries
        assert len(retriever.sections_with_summaries) == 2
    
    def test_balanced_results_for_detail_query(self):
        """Returns balanced results for non-summary queries."""
        summary_result = RetrievalResult(
            content="Summary",
            metadata={"is_summary": True, "section": "Item 1"},
            score=0.8,
        )
        detail_result = RetrievalResult(
            content="Detail",
            metadata={"is_summary": False, "section": "Item 1"},
            score=0.7,
        )
        
        base = MockBaseRetriever(results=[summary_result, detail_result])
        retriever = HierarchicalRetriever(base_retriever=base, summary_boost=1.0)
        
        # Non-summary query
        results = retriever.retrieve("What is the revenue?", top_k=2)
        
        # Should have both but details more prominent
        types = [r.metadata.get("is_summary") for r in results]
        assert True in types  # Has summary
        assert False in types or None in types  # Has detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
