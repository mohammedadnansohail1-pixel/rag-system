"""FastAPI application for RAG system."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QueryRequest,
    QueryResponse,
    IngestFileRequest,
    IngestDirectoryRequest,
    IngestResponse,
    HealthResponse,
    SourceDocument,
    ErrorResponse,
)
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline_production import ProductionRAGPipeline
from src.guardrails.config import GuardrailsConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[ProductionRAGPipeline] = None


def get_config_from_env() -> dict:
    """Load configuration from environment variables."""
    return {
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "collection_name": os.getenv("COLLECTION_NAME", "rag_production"),
        "embedding_dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
        # Guardrails config
        "score_threshold": float(os.getenv("SCORE_THRESHOLD", "0.4")),
        "min_sources": int(os.getenv("MIN_SOURCES", "2")),
        "min_avg_score": float(os.getenv("MIN_AVG_SCORE", "0.5")),
    }


def initialize_pipeline() -> ProductionRAGPipeline:
    """Initialize the RAG pipeline with all components."""
    config = get_config_from_env()
    
    logger.info("Initializing RAG pipeline...")
    
    embeddings = OllamaEmbeddings(
        host=config["ollama_host"],
        model=config["embedding_model"],
        dimensions=config["embedding_dimensions"],
    )
    logger.info(f"✓ Embeddings: {config['embedding_model']}")
    
    vectorstore = QdrantVectorStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        collection_name=config["collection_name"],
        dimensions=config["embedding_dimensions"],
    )
    logger.info(f"✓ VectorStore: Qdrant ({config['collection_name']})")
    
    retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        top_k=10,
    )
    logger.info("✓ Retriever: Dense")
    
    llm = OllamaLLM(
        host=config["ollama_host"],
        model=config["ollama_model"],
        temperature=0.1,
    )
    logger.info(f"✓ LLM: {config['ollama_model']}")
    
    guardrails_config = GuardrailsConfig(
        score_threshold=config["score_threshold"],
        min_sources=config["min_sources"],
        min_avg_score=config["min_avg_score"],
    )
    logger.info("✓ Guardrails configured")
    
    pipeline = ProductionRAGPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        chunker_config={
            "strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 100,
        },
        guardrails_config=guardrails_config,
    )
    
    logger.info("✓ Pipeline ready")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global pipeline
    
    # Startup
    logger.info("Starting RAG API...")
    try:
        pipeline = initialize_pipeline()
        logger.info("RAG API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API...")


# Create FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Production-ready Retrieval-Augmented Generation API with guardrails",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== API Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check health of all components."""
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    health = pipeline.health_check()
    all_healthy = all(health.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        components=health,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Retrieves relevant documents and generates an answer with guardrails.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        response = pipeline.query(
            question=request.question,
            top_k=request.top_k,
        )
        
        return QueryResponse(
            answer=response.answer,
            query=response.query,
            confidence=response.confidence,
            confidence_emoji=response.confidence_emoji,
            avg_score=response.avg_score,
            sources=[
                SourceDocument(
                    content=s["content"],
                    score=s["score"],
                    metadata=s["metadata"],
                )
                for s in response.sources
            ],
            validation_passed=response.validation_passed,
            rejection_reason=response.rejection_reason,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(request: IngestFileRequest):
    """
    Ingest a single file into the RAG system.
    
    Supports: PDF, TXT, MD files.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        chunks = pipeline.ingest_file(request.file_path)
        
        return IngestResponse(
            success=True,
            chunks_indexed=chunks,
            message=f"Successfully indexed {chunks} chunks from {request.file_path}",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/ingest/directory", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_directory(request: IngestDirectoryRequest):
    """
    Ingest all documents from a directory.
    
    Supports: PDF, TXT, MD files.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        chunks = pipeline.ingest_directory(
            directory=request.directory,
            recursive=request.recursive,
            file_types=request.file_types,
        )
        
        return IngestResponse(
            success=True,
            chunks_indexed=chunks,
            message=f"Successfully indexed {chunks} chunks from {request.directory}",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {request.directory}",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# Run with: uvicorn src.api.main:app --reload
