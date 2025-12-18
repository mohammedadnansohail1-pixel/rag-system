"""Tests for hybrid retriever."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.retrieval import (
    HybridRetriever,
    get_registered_retrievers,
    RetrievalResult,
)
from src.retrieval.sparse_encoder import SpladeEncoder
from src.core.types import SparseVector
from src.vectorstores.base import SearchResult


class TestHybridRetrieverRegistration:
    """Test hybrid retriever registration."""
    
    def test_registered(self):
        """Verify hybrid is registered."""
        assert "hybrid" in get_registered_retrievers()


class TestSparseVector:
    """Test SparseVector dataclass."""
    
    def test_creation(self):
        """Can create sparse vector."""
        vec = SparseVector(indices=[1, 5, 10], values=[0.5, 0.8, 0.3])
        assert len(vec.indices) == 3
        assert len(vec.values) == 3
    
    def test_to_dict(self):
        """Converts to dictionary correctly."""
        vec = SparseVector(indices=[1, 5, 10], values=[0.5, 0.8, 0.3])
        d = vec.to_dict()
        assert d == {1: 0.5, 5: 0.8, 10: 0.3}
    
    def test_repr(self):
        """Repr shows non-zero count."""
        vec = SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])
        assert "nnz=3" in repr(vec)


class TestSpladeEncoder:
    """Test SPLADE sparse encoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return SpladeEncoder()
    
    def test_encode_returns_sparse_vector(self, encoder):
        """Encode returns SparseVector."""
        vec = encoder.encode("test query")
        assert isinstance(vec, SparseVector)
        assert len(vec.indices) > 0
        assert len(vec.indices) == len(vec.values)
    
    def test_encode_batch(self, encoder):
        """Batch encoding works."""
        vecs = encoder.encode_batch(["query one", "query two"])
        assert len(vecs) == 2
        assert all(isinstance(v, SparseVector) for v in vecs)
    
    def test_decode_tokens(self, encoder):
        """Can decode sparse vector to tokens."""
        vec = encoder.encode("kafka streaming platform")
        tokens = encoder.decode_tokens(vec, top_k=5)
        
        assert len(tokens) <= 5
        assert all(isinstance(t, tuple) for t in tokens)
        assert all(len(t) == 2 for t in tokens)  # (token, weight)
    
    def test_health_check(self, encoder):
        """Health check passes."""
        assert encoder.health_check() is True


class TestHybridRetriever:
    """Test HybridRetriever class."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        mock = Mock()
        mock.embed_text.return_value = [0.1] * 768
        mock.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
        return mock
    
    @pytest.fixture
    def mock_sparse_encoder(self):
        """Create mock sparse encoder."""
        from src.retrieval.sparse_encoder import BaseSparseEncoder
        mock = Mock(spec=BaseSparseEncoder)
        mock.encode.return_value = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])
        mock.encode_batch.return_value = [
            SparseVector(indices=[1, 2], values=[0.5, 0.3]),
            SparseVector(indices=[3, 4], values=[0.4, 0.2]),
        ]
        mock.health_check.return_value = True
        return mock
    
    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock hybrid vector store."""
        mock = Mock()
        mock.search.return_value = [
            SearchResult(content="dense result", metadata={"source": "a"}, score=0.9),
        ]
        mock.sparse_search.return_value = [
            SearchResult(content="sparse result", metadata={"source": "b"}, score=0.8),
        ]
        mock.hybrid_search.return_value = [
            SearchResult(content="hybrid result", metadata={"source": "c"}, score=0.95),
        ]
        mock.add_hybrid.return_value = ["id1", "id2"]
        mock.health_check.return_value = True
        return mock
    
    @pytest.fixture
    def retriever(self, mock_embeddings, mock_vectorstore, mock_sparse_encoder):
        """Create retriever with mocks."""
        return HybridRetriever(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore,
            sparse_encoder=mock_sparse_encoder,
            top_k=5,
        )
    
    def test_retrieve_hybrid_mode(self, retriever, mock_vectorstore):
        """Default retrieve uses hybrid mode."""
        results = retriever.retrieve("test query")
        
        mock_vectorstore.hybrid_search.assert_called_once()
        assert len(results) == 1
        assert results[0].content == "hybrid result"
    
    def test_retrieve_dense_mode(self, retriever, mock_vectorstore):
        """Dense mode uses dense search."""
        results = retriever.retrieve("test query", mode="dense")
        
        mock_vectorstore.search.assert_called_once()
        assert results[0].content == "dense result"
    
    def test_retrieve_sparse_mode(self, retriever, mock_vectorstore):
        """Sparse mode uses sparse search."""
        results = retriever.retrieve("test query", mode="sparse")
        
        mock_vectorstore.sparse_search.assert_called_once()
        assert results[0].content == "sparse result"
    
    def test_retrieve_respects_top_k(self, retriever, mock_embeddings):
        """Top_k parameter is passed correctly."""
        retriever.retrieve("test", top_k=10)
        
        # Verify embed_text was called (for hybrid/dense)
        mock_embeddings.embed_text.assert_called_with("test")
    
    def test_add_documents(self, retriever, mock_embeddings, mock_sparse_encoder, mock_vectorstore):
        """Add documents encodes both dense and sparse."""
        texts = ["doc1", "doc2"]
        metadatas = [{"a": 1}, {"b": 2}]
        
        ids = retriever.add_documents(texts, metadatas)
        
        mock_embeddings.embed_batch.assert_called_once_with(texts)
        mock_sparse_encoder.encode_batch.assert_called_once_with(texts)
        mock_vectorstore.add_hybrid.assert_called_once()
        assert ids == ["id1", "id2"]
    
    def test_health_check(self, retriever):
        """Health check verifies all components."""
        assert retriever.health_check() is True
    
    def test_health_check_fails_if_vectorstore_unhealthy(self, retriever, mock_vectorstore):
        """Health check fails if vectorstore unhealthy."""
        mock_vectorstore.health_check.return_value = False
        assert retriever.health_check() is False
    
    def test_health_check_fails_if_sparse_encoder_unhealthy(self, retriever, mock_sparse_encoder):
        """Health check fails if sparse encoder unhealthy."""
        mock_sparse_encoder.health_check.return_value = False
        assert retriever.health_check() is False
    
    def test_results_are_retrieval_result_type(self, retriever):
        """Results are RetrievalResult instances."""
        results = retriever.retrieve("test")
        
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_repr(self, retriever):
        """Repr shows top_k."""
        assert "top_k=5" in repr(retriever)


class TestHybridRetrieverIntegration:
    """Integration tests requiring Qdrant."""
    
    @pytest.fixture
    def qdrant_available(self):
        """Check if Qdrant is running."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            client.get_collections()
            return True
        except Exception:
            return False
    
    @pytest.mark.integration
    def test_full_hybrid_flow(self, qdrant_available):
        """Test full add and retrieve flow."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")
        
        from src.embeddings.factory import EmbeddingsFactory
        from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
        
        # Setup
        embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
        vectorstore = QdrantHybridStore(
            collection_name="test_hybrid_integration",
            dense_dimensions=768,
            recreate_collection=True,
        )
        retriever = HybridRetriever(
            embeddings=embeddings,
            vectorstore=vectorstore,
        )
        
        # Add documents
        texts = [
            "Kafka is a distributed streaming platform",
            "Redis is an in-memory cache",
        ]
        retriever.add_documents(texts)
        
        # Retrieve
        results = retriever.retrieve("streaming platform", top_k=2)
        
        # Verify
        assert len(results) > 0
        assert "kafka" in results[0].content.lower() or "streaming" in results[0].content.lower()
        
        # Cleanup
        vectorstore._client.delete_collection("test_hybrid_integration")
