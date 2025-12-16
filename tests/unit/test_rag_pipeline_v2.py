"""Tests for RAGPipelineV2 with hybrid retrieval and reranking."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.pipeline import RAGPipelineV2, RAGResponseV2
from src.retrieval.base import RetrievalResult
from src.reranking.base import RerankResult


class TestRAGResponseV2:
    """Test RAGResponseV2 dataclass."""
    
    def test_creation(self):
        """Can create response."""
        response = RAGResponseV2(
            answer="Test answer",
            sources=[{"content": "doc1", "score": 0.9}],
            query="test query",
        )
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.query == "test query"
    
    def test_default_fields(self):
        """Default fields are set correctly."""
        response = RAGResponseV2(
            answer="Test",
            sources=[],
            query="query",
        )
        assert response.retrieval_scores == []
        assert response.rerank_scores is None
        assert response.metadata == {}
    
    def test_repr_without_reranking(self):
        """Repr shows not reranked."""
        response = RAGResponseV2(
            answer="Test answer",
            sources=[],
            query="query",
        )
        assert "not reranked" in repr(response)
    
    def test_repr_with_reranking(self):
        """Repr shows reranked."""
        response = RAGResponseV2(
            answer="Test answer",
            sources=[],
            query="query",
            rerank_scores=[0.9, 0.8],
        )
        assert "reranked" in repr(response)
        assert "not reranked" not in repr(response)


class TestRAGPipelineV2:
    """Test RAGPipelineV2 class."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        mock = Mock()
        mock.embed_text.return_value = [0.1] * 768
        mock.embed_batch.return_value = [[0.1] * 768]
        return mock
    
    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vector store."""
        mock = Mock()
        mock.health_check.return_value = True
        return mock
    
    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever."""
        mock = Mock()
        mock.retrieve.return_value = [
            RetrievalResult(content="doc1 content", metadata={"source": "a.md"}, score=0.9),
            RetrievalResult(content="doc2 content", metadata={"source": "b.md"}, score=0.8),
            RetrievalResult(content="doc3 content", metadata={"source": "c.md"}, score=0.7),
        ]
        mock.add_documents.return_value = ["id1", "id2"]
        mock.health_check.return_value = True
        return mock
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        mock = Mock()
        mock.generate_with_context.return_value = "Generated answer based on context."
        mock.health_check.return_value = True
        return mock
    
    @pytest.fixture
    def mock_reranker(self):
        """Create mock reranker."""
        mock = Mock()
        mock.rerank.return_value = [
            RerankResult(content="doc1 content", metadata={"source": "a.md"}, score=5.0, original_rank=1, new_rank=1),
            RerankResult(content="doc3 content", metadata={"source": "c.md"}, score=3.0, original_rank=3, new_rank=2),
        ]
        mock.health_check.return_value = True
        return mock
    
    @pytest.fixture
    def pipeline_without_reranker(self, mock_embeddings, mock_vectorstore, mock_retriever, mock_llm):
        """Create pipeline without reranker."""
        return RAGPipelineV2(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore,
            retriever=mock_retriever,
            llm=mock_llm,
            reranker=None,
        )
    
    @pytest.fixture
    def pipeline_with_reranker(self, mock_embeddings, mock_vectorstore, mock_retriever, mock_llm, mock_reranker):
        """Create pipeline with reranker."""
        return RAGPipelineV2(
            embeddings=mock_embeddings,
            vectorstore=mock_vectorstore,
            retriever=mock_retriever,
            llm=mock_llm,
            reranker=mock_reranker,
        )
    
    # Query tests
    def test_query_without_reranker(self, pipeline_without_reranker, mock_retriever, mock_llm):
        """Query works without reranker."""
        response = pipeline_without_reranker.query("test question", rerank_top_n=3)
        
        mock_retriever.retrieve.assert_called_once()
        mock_llm.generate_with_context.assert_called_once()
        
        assert response.answer == "Generated answer based on context."
        assert len(response.sources) == 3
        assert response.rerank_scores is None
        assert response.metadata["reranked"] is False
    
    def test_query_with_reranker(self, pipeline_with_reranker, mock_retriever, mock_reranker, mock_llm):
        """Query uses reranker when available."""
        response = pipeline_with_reranker.query(
            "test question",
            retrieval_top_k=10,
            rerank_top_n=2,
        )
        
        # Should retrieve more candidates for reranking
        mock_retriever.retrieve.assert_called_with("test question", top_k=10)
        mock_reranker.rerank.assert_called_once()
        mock_llm.generate_with_context.assert_called_once()
        
        assert len(response.sources) == 2
        assert response.rerank_scores is not None
        assert response.metadata["reranked"] is True
    
    def test_query_can_disable_reranker(self, pipeline_with_reranker, mock_retriever, mock_reranker):
        """Can disable reranker per query."""
        response = pipeline_with_reranker.query(
            "test question",
            use_reranker=False,
        )
        
        mock_reranker.rerank.assert_not_called()
        assert response.metadata["reranked"] is False
    
    def test_query_can_force_reranker(self, pipeline_without_reranker):
        """Force reranker raises no error if None."""
        # When use_reranker=True but no reranker, should still work
        # (reranker is None so nothing happens)
        response = pipeline_without_reranker.query(
            "test question",
            use_reranker=True,
        )
        assert response.metadata["reranked"] is False
    
    def test_query_empty_results(self, pipeline_without_reranker, mock_retriever):
        """Handles empty retrieval results."""
        mock_retriever.retrieve.return_value = []
        
        response = pipeline_without_reranker.query("test question")
        
        assert "couldn't find" in response.answer.lower()
        assert len(response.sources) == 0
    
    def test_query_preserves_retrieval_scores(self, pipeline_without_reranker):
        """Retrieval scores are captured."""
        response = pipeline_without_reranker.query("test question")
        
        assert response.retrieval_scores == [0.9, 0.8, 0.7]
    
    # Query compare tests
    def test_query_compare(self, pipeline_with_reranker):
        """query_compare returns both versions."""
        results = pipeline_with_reranker.query_compare("test question", top_k=2)
        
        assert "without_reranking" in results
        assert "with_reranking" in results
        assert results["without_reranking"].metadata["reranked"] is False
        assert results["with_reranking"].metadata["reranked"] is True
    
    def test_query_compare_raises_without_reranker(self, pipeline_without_reranker):
        """query_compare raises if no reranker."""
        with pytest.raises(ValueError, match="No reranker"):
            pipeline_without_reranker.query_compare("test question")
    
    # Ingest tests
    def test_ingest_file(self, pipeline_without_reranker, mock_retriever):
        """ingest_file processes documents."""
        with patch('src.pipeline.rag_pipeline_v2.LoaderFactory') as mock_loader:
            with patch.object(pipeline_without_reranker.chunker, 'chunk') as mock_chunk:
                from src.loaders.base import Document
                from src.chunkers.base import Chunk
                
                mock_loader.load.return_value = [Document(content="test content", metadata={})]
                mock_chunk.return_value = [Chunk(content="chunk1", metadata={})]
                
                count = pipeline_without_reranker.ingest_file("test.txt")
                
                assert count == 1
                mock_retriever.add_documents.assert_called_once()
    
    def test_ingest_file_no_chunks(self, pipeline_without_reranker, mock_retriever):
        """ingest_file handles empty chunks."""
        with patch('src.pipeline.rag_pipeline_v2.LoaderFactory') as mock_loader:
            with patch.object(pipeline_without_reranker.chunker, 'chunk') as mock_chunk:
                from src.loaders.base import Document
                
                mock_loader.load.return_value = [Document(content="", metadata={})]
                mock_chunk.return_value = []
                
                count = pipeline_without_reranker.ingest_file("empty.txt")
                
                assert count == 0
                mock_retriever.add_documents.assert_not_called()
    
    # Health check tests
    def test_health_check_without_reranker(self, pipeline_without_reranker):
        """Health check works without reranker."""
        status = pipeline_without_reranker.health_check()
        
        assert "vectorstore" in status
        assert "llm" in status
        assert "retriever" in status
        assert "reranker" not in status
    
    def test_health_check_with_reranker(self, pipeline_with_reranker):
        """Health check includes reranker."""
        status = pipeline_with_reranker.health_check()
        
        assert "reranker" in status
        assert status["reranker"] is True
    
    def test_health_check_detects_failures(self, pipeline_with_reranker, mock_vectorstore):
        """Health check detects unhealthy components."""
        mock_vectorstore.health_check.return_value = False
        
        status = pipeline_with_reranker.health_check()
        
        assert status["vectorstore"] is False
    
    # Repr test
    def test_repr(self, pipeline_with_reranker):
        """Repr shows retriever and reranker."""
        repr_str = repr(pipeline_with_reranker)
        
        assert "RAGPipelineV2" in repr_str
        assert "retriever=" in repr_str
        assert "reranker=" in repr_str


class TestRAGPipelineV2Integration:
    """Integration tests requiring external services."""
    
    @pytest.fixture
    def services_available(self):
        """Check if required services are running."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            client.get_collections()
            
            import ollama
            ollama.list()
            
            return True
        except Exception:
            return False
    
    @pytest.mark.integration
    def test_full_pipeline_flow(self, services_available):
        """Test full ingest and query flow."""
        if not services_available:
            pytest.skip("Required services not available")
        
        from src.embeddings.factory import EmbeddingsFactory
        from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.reranking import RerankerFactory
        from src.generation.factory import LLMFactory
        
        # Setup
        embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
        vectorstore = QdrantHybridStore(
            collection_name="test_pipeline_v2",
            dense_dimensions=768,
            recreate_collection=True,
        )
        retriever = HybridRetriever(embeddings=embeddings, vectorstore=vectorstore)
        reranker = RerankerFactory.create("cross_encoder")
        llm = LLMFactory.create("ollama", model="llama3.2")
        
        pipeline = RAGPipelineV2(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )
        
        # Add test documents
        retriever.add_documents(
            texts=[
                "Kafka is a distributed streaming platform.",
                "Redis is an in-memory cache.",
            ],
            metadatas=[{"source": "kafka.md"}, {"source": "redis.md"}],
        )
        
        # Query
        response = pipeline.query(
            "What is Kafka?",
            retrieval_top_k=5,
            rerank_top_n=2,
        )
        
        # Verify
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.metadata["reranked"] is True
        
        # Cleanup
        vectorstore._client.delete_collection("test_pipeline_v2")
