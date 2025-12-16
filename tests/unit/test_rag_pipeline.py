"""Tests for RAG pipeline."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse
from src.loaders.base import Document
from src.chunkers.base import Chunk
from src.retrieval.base import RetrievalResult


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embeddings = MagicMock()
        self.mock_vectorstore = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()

    def test_init(self):
        """Should initialize with all components."""
        # Act
        pipeline = RAGPipeline(
            embeddings=self.mock_embeddings,
            vectorstore=self.mock_vectorstore,
            retriever=self.mock_retriever,
            llm=self.mock_llm
        )

        # Assert
        assert pipeline.embeddings == self.mock_embeddings
        assert pipeline.vectorstore == self.mock_vectorstore
        assert pipeline.retriever == self.mock_retriever
        assert pipeline.llm == self.mock_llm

    @patch("src.pipeline.rag_pipeline.LoaderFactory")
    def test_ingest_file(self, mock_loader_factory):
        """Should load, chunk, and index a file."""
        # Arrange
        mock_loader_factory.load.return_value = [
            Document(content="Test content", metadata={"filename": "test.txt"})
        ]
        
        self.mock_retriever.add_documents.return_value = ["id1"]
        
        pipeline = RAGPipeline(
            embeddings=self.mock_embeddings,
            vectorstore=self.mock_vectorstore,
            retriever=self.mock_retriever,
            llm=self.mock_llm,
            chunker_config={"strategy": "fixed", "chunk_size": 100}
        )

        # Act
        count = pipeline.ingest_file("test.txt")

        # Assert
        assert count > 0
        self.mock_retriever.add_documents.assert_called_once()

    @patch("src.pipeline.rag_pipeline.LoaderFactory")
    def test_ingest_directory(self, mock_loader_factory):
        """Should load all documents from directory."""
        # Arrange
        mock_loader_factory.load_directory.return_value = [
            Document(content="Doc 1 content", metadata={"filename": "doc1.txt"}),
            Document(content="Doc 2 content", metadata={"filename": "doc2.txt"}),
        ]
        
        self.mock_retriever.add_documents.return_value = ["id1", "id2"]
        
        pipeline = RAGPipeline(
            embeddings=self.mock_embeddings,
            vectorstore=self.mock_vectorstore,
            retriever=self.mock_retriever,
            llm=self.mock_llm
        )

        # Act
        count = pipeline.ingest_directory("./docs")

        # Assert
        assert count > 0
        mock_loader_factory.load_directory.assert_called_once()

    def test_query(self):
        """Should retrieve context and generate answer."""
        # Arrange
        self.mock_retriever.retrieve.return_value = [
            RetrievalResult(
                content="Relevant chunk 1",
                metadata={"source": "doc1.txt"},
                score=0.9
            ),
            RetrievalResult(
                content="Relevant chunk 2",
                metadata={"source": "doc2.txt"},
                score=0.8
            ),
        ]
        
        self.mock_llm.generate_with_context.return_value = "Generated answer"
        
        pipeline = RAGPipeline(
            embeddings=self.mock_embeddings,
            vectorstore=self.mock_vectorstore,
            retriever=self.mock_retriever,
            llm=self.mock_llm
        )

        # Act
        response = pipeline.query("What is RAG?")

        # Assert
        assert isinstance(response, RAGResponse)
        assert response.answer == "Generated answer"
        assert response.query == "What is RAG?"
        assert len(response.sources) == 2
        self.mock_retriever.retrieve.assert_called_once_with("What is RAG?", top_k=5)

    def test_query_no_results(self):
        """Should handle empty retrieval results."""
        # Arrange
        self.mock_retriever.retrieve.return_value = []
        
        pipeline = RAGPipeline(
            embeddings=self.mock_embeddings,
            vectorstore=self.mock_vectorstore,
            retriever=self.mock_retriever,
            llm=self.mock_llm
        )

        # Act
        response = pipeline.query("Unknown topic")

        # Assert
        assert "couldn't find" in response.answer.lower()
        assert len(response.sources) == 0
        self.mock_llm.generate_with_context.assert_not_called()

    def test_query_custom_top_k(self):
        """Should pass custom top_k to retriever."""
        # Arrange
        self.mock_retriever.retrieve.return_value = []
        
        pipeline = RAGPipeline(
            embeddings=self.mock_embeddings,
            vectorstore=self.mock_vectorstore,
            retriever=self.mock_retriever,
            llm=self.mock_llm
        )

        # Act
        pipeline.query("Question", top_k=10)

        # Assert
        self.mock_retriever.retrieve.assert_called_once_with("Question", top_k=10)

    def test_health_check(self):
        """Should check health of all components."""
        # Arrange
        self.mock_vectorstore.health_check.return_value = True
        self.mock_llm.health_check.return_value = True
        self.mock_retriever.health_check.return_value = True
        
        pipeline = RAGPipeline(
            embeddings=self.mock_embeddings,
            vectorstore=self.mock_vectorstore,
            retriever=self.mock_retriever,
            llm=self.mock_llm
        )

        # Act
        health = pipeline.health_check()

        # Assert
        assert health["vectorstore"] is True
        assert health["llm"] is True
        assert health["retriever"] is True
