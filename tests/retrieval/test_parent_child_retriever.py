"""Tests for parent-child retriever."""

import pytest
from src.retrieval.base import RetrievalResult
from src.retrieval.parent_child_retriever import ParentChildRetriever


class MockBaseRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, results=None):
        self.results = results or []
        self.documents = []
    
    def add_documents(self, texts, metadatas=None, **kwargs):
        metadatas = metadatas or [{}] * len(texts)
        self.documents = list(zip(texts, metadatas))
    
    def retrieve(self, query, top_k=5, **kwargs):
        return self.results[:top_k]
    
    def health_check(self):
        return True


class TestParentChildRetriever:
    """Test parent-child retriever."""
    
    def test_initialization(self):
        """Initializes correctly."""
        base = MockBaseRetriever()
        pc = ParentChildRetriever(base_retriever=base)
        
        assert pc.include_parents is True
        assert pc.parent_count == 0
    
    def test_caches_parents(self):
        """Caches parent chunks on add_documents."""
        base = MockBaseRetriever()
        pc = ParentChildRetriever(base_retriever=base)
        
        texts = ["parent content", "child content"]
        metadatas = [
            {"chunk_id": "p1", "chunk_type": "parent"},
            {"chunk_id": "c1", "parent_id": "p1"},
        ]
        
        pc.add_documents(texts=texts, metadatas=metadatas)
        
        assert pc.parent_count == 1
        assert pc.get_parent("p1") is not None
    
    def test_replaces_child_with_parent(self):
        """Replaces child with parent in results."""
        # Create results where child has parent_id
        child_result = RetrievalResult(
            content="child content",
            metadata={"chunk_id": "c1", "parent_id": "p1", "chunk_type": "child"},
            score=0.9,
        )
        
        base = MockBaseRetriever(results=[child_result])
        pc = ParentChildRetriever(
            base_retriever=base,
            replace_children_with_parents=True,
            parent_weight=0.95,
        )
        
        # Add parent to cache
        pc.add_documents(
            texts=["parent content with more context"],
            metadatas=[{"chunk_id": "p1", "chunk_type": "parent"}],
        )
        
        results = pc.retrieve("test query", top_k=5)
        
        # Should return parent instead of child
        assert len(results) == 1
        assert results[0].content == "parent content with more context"
        assert results[0].metadata.get("retrieved_as") == "parent_of_match"
    
    def test_keeps_regular_chunks(self):
        """Keeps regular chunks without parents."""
        regular_result = RetrievalResult(
            content="regular content",
            metadata={"chunk_id": "r1"},
            score=0.8,
        )
        
        base = MockBaseRetriever(results=[regular_result])
        pc = ParentChildRetriever(base_retriever=base)
        
        results = pc.retrieve("test query", top_k=5)
        
        assert len(results) == 1
        assert results[0].content == "regular content"
    
    def test_deduplicates_by_id(self):
        """Removes duplicate chunks by ID."""
        result1 = RetrievalResult(
            content="same content",
            metadata={"chunk_id": "c1"},
            score=0.9,
        )
        result2 = RetrievalResult(
            content="same content",
            metadata={"chunk_id": "c1"},
            score=0.8,
        )
        
        base = MockBaseRetriever(results=[result1, result2])
        pc = ParentChildRetriever(base_retriever=base, deduplicate=True)
        
        results = pc.retrieve("test query", top_k=5)
        
        assert len(results) == 1
    
    def test_parent_weight_applied(self):
        """Applies weight to parent scores."""
        child_result = RetrievalResult(
            content="child",
            metadata={"chunk_id": "c1", "parent_id": "p1"},
            score=1.0,
        )
        
        base = MockBaseRetriever(results=[child_result])
        pc = ParentChildRetriever(
            base_retriever=base,
            parent_weight=0.8,
            replace_children_with_parents=True,
        )
        
        pc.add_documents(
            texts=["parent"],
            metadatas=[{"chunk_id": "p1", "chunk_type": "parent"}],
        )
        
        results = pc.retrieve("test", top_k=5)
        
        assert results[0].score == 0.8  # 1.0 * 0.8
    
    def test_health_check_delegates(self):
        """Health check delegates to base retriever."""
        base = MockBaseRetriever()
        pc = ParentChildRetriever(base_retriever=base)
        
        assert pc.health_check() is True
    
    def test_without_parent_expansion(self):
        """Works without parent expansion."""
        result = RetrievalResult(
            content="content",
            metadata={"chunk_id": "c1"},
            score=0.9,
        )
        
        base = MockBaseRetriever(results=[result])
        pc = ParentChildRetriever(base_retriever=base, include_parents=False)
        
        results = pc.retrieve("test", top_k=5)
        
        assert len(results) == 1
        assert results[0].content == "content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
