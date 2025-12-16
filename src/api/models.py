"""Pydantic models for API request/response schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ============== Request Models ==============

class QueryRequest(BaseModel):
    """Request body for RAG query."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": "What is gradient descent?",
            "top_k": 10
        }
    })
    
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")


class IngestFileRequest(BaseModel):
    """Request body for file ingestion."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "file_path": "data/documents/example.pdf"
        }
    })
    
    file_path: str = Field(..., description="Path to file to ingest")


class IngestDirectoryRequest(BaseModel):
    """Request body for directory ingestion."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "directory": "data/documents",
            "recursive": True,
            "file_types": [".pdf", ".txt", ".md"]
        }
    })
    
    directory: str = Field(..., description="Path to directory")
    recursive: bool = Field(default=True, description="Search subdirectories")
    file_types: Optional[List[str]] = Field(
        default=None, 
        description="Filter by extensions (e.g., ['.pdf', '.txt'])"
    )


# ============== Response Models ==============

class SourceDocument(BaseModel):
    """A source document used in the response."""
    content: str = Field(..., description="Chunk content (truncated)")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response from RAG query."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "answer": "Gradient descent is an optimization algorithm...",
            "query": "What is gradient descent?",
            "confidence": "high",
            "confidence_emoji": "🟢",
            "avg_score": 0.73,
            "sources": [
                {"content": "Gradient descent...", "score": 0.75, "metadata": {}}
            ],
            "validation_passed": True,
            "rejection_reason": None
        }
    })
    
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original question")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    confidence_emoji: str = Field(..., description="Visual confidence indicator")
    avg_score: float = Field(..., description="Average relevance score")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    validation_passed: bool = Field(..., description="Whether guardrails passed")
    rejection_reason: Optional[str] = Field(default=None, description="Why validation failed")


class IngestResponse(BaseModel):
    """Response from ingestion."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "chunks_indexed": 156,
            "message": "Successfully indexed 156 chunks from data/documents"
        }
    })
    
    success: bool = Field(..., description="Whether ingestion succeeded")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "components": {
                "vectorstore": True,
                "llm": True,
                "retriever": True
            }
        }
    })
    
    status: str = Field(..., description="Overall status")
    components: Dict[str, bool] = Field(..., description="Component health status")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional details")
