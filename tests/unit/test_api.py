"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class TestAPIModels:
    """Tests for API Pydantic models."""

    def test_query_request_validation(self):
        """Should validate query request."""
        from src.api.models import QueryRequest
        
        # Valid request
        req = QueryRequest(question="What is ML?")
        assert req.question == "What is ML?"
        assert req.top_k == 10  # default

    def test_query_request_custom_top_k(self):
        """Should accept custom top_k."""
        from src.api.models import QueryRequest
        
        req = QueryRequest(question="Test", top_k=5)
        assert req.top_k == 5

    def test_query_response_model(self):
        """Should create valid query response."""
        from src.api.models import QueryResponse, SourceDocument
        
        resp = QueryResponse(
            answer="Test answer",
            query="Test query",
            confidence="high",
            confidence_emoji="🟢",
            avg_score=0.75,
            sources=[SourceDocument(content="test", score=0.8, metadata={})],
            validation_passed=True,
        )
        
        assert resp.answer == "Test answer"
        assert resp.confidence == "high"
        assert len(resp.sources) == 1

    def test_ingest_response_model(self):
        """Should create valid ingest response."""
        from src.api.models import IngestResponse
        
        resp = IngestResponse(
            success=True,
            chunks_indexed=100,
            message="Done"
        )
        
        assert resp.success is True
        assert resp.chunks_indexed == 100

    def test_health_response_model(self):
        """Should create valid health response."""
        from src.api.models import HealthResponse
        
        resp = HealthResponse(
            status="healthy",
            components={"vectorstore": True, "llm": True}
        )
        
        assert resp.status == "healthy"
        assert resp.components["vectorstore"] is True


class TestAPIEndpoints:
    """Tests for API endpoints using TestClient."""

    def test_root_endpoint(self):
        """Should return API info."""
        import src.api.main as api_main
        
        # Mock the pipeline
        mock_pipeline = MagicMock()
        api_main.pipeline = mock_pipeline
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()

    def test_health_endpoint(self):
        """Should return health status."""
        import src.api.main as api_main
        
        mock_pipeline = MagicMock()
        mock_pipeline.health_check.return_value = {
            "vectorstore": True,
            "llm": True,
            "retriever": True,
        }
        api_main.pipeline = mock_pipeline
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_endpoint_degraded(self):
        """Should return degraded when component unhealthy."""
        import src.api.main as api_main
        
        mock_pipeline = MagicMock()
        mock_pipeline.health_check.return_value = {
            "vectorstore": True,
            "llm": False,
            "retriever": True,
        }
        api_main.pipeline = mock_pipeline
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_query_endpoint(self):
        """Should process query."""
        import src.api.main as api_main
        
        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.query = "Test question"
        mock_response.confidence = "high"
        mock_response.confidence_emoji = "🟢"
        mock_response.avg_score = 0.75
        mock_response.sources = []
        mock_response.validation_passed = True
        mock_response.rejection_reason = None
        
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = mock_response
        api_main.pipeline = mock_pipeline
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.post("/query", json={"question": "Test question"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Test answer"

    def test_query_endpoint_no_pipeline(self):
        """Should return 503 when pipeline not initialized."""
        import src.api.main as api_main
        
        api_main.pipeline = None
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.post("/query", json={"question": "Test"})
        assert response.status_code == 503

    def test_ingest_file_success(self):
        """Should ingest file successfully."""
        import src.api.main as api_main
        
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = 50
        api_main.pipeline = mock_pipeline
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.post("/ingest/file", json={"file_path": "test.pdf"})
        assert response.status_code == 200
        assert response.json()["chunks_indexed"] == 50

    def test_ingest_file_not_found(self):
        """Should return 404 for missing file."""
        import src.api.main as api_main
        
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.side_effect = FileNotFoundError()
        api_main.pipeline = mock_pipeline
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.post("/ingest/file", json={"file_path": "nonexistent.pdf"})
        assert response.status_code == 404

    def test_ingest_directory_success(self):
        """Should ingest directory successfully."""
        import src.api.main as api_main
        
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_directory.return_value = 200
        api_main.pipeline = mock_pipeline
        
        client = TestClient(api_main.app, raise_server_exceptions=False)
        
        response = client.post("/ingest/directory", json={"directory": "data/docs"})
        assert response.status_code == 200
        assert response.json()["chunks_indexed"] == 200
