"""Production-ready RAG pipeline with guardrails."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.loaders.factory import LoaderFactory
from src.chunkers.factory import ChunkerFactory
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore
from src.retrieval.base import BaseRetriever
from src.generation.base import BaseLLM
from src.guardrails.config import GuardrailsConfig, PRODUCTION_CONFIG
from src.guardrails.validator import GuardrailsValidator, get_uncertainty_response
from src.guardrails.prompts import (
    SYSTEM_PROMPT_STRICT,
    build_prompt_with_context,
    get_confidence_guidance,
)

logger = logging.getLogger(__name__)


@dataclass
class ProductionRAGResponse:
    """
    Production response with full metadata.
    
    Attributes:
        answer: Generated answer
        sources: List of source documents used
        query: Original query
        confidence: Confidence level (low/medium/high)
        avg_score: Average relevance score
        is_uncertain: Whether response indicates uncertainty
        validation_passed: Whether guardrails validation passed
        rejection_reason: Why validation failed (if applicable)
    """
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: str = "low"
    avg_score: float = 0.0
    is_uncertain: bool = False
    validation_passed: bool = True
    rejection_reason: Optional[str] = None
    
    @property
    def confidence_emoji(self) -> str:
        return {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(self.confidence, "⚪")
    
    def __repr__(self) -> str:
        preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        return (
            f"ProductionRAGResponse("
            f"confidence={self.confidence}, "
            f"sources={len(self.sources)}, "
            f"answer='{preview}')"
        )


class ProductionRAGPipeline:
    """
    Production-ready RAG pipeline with guardrails.
    
    Features:
    - Score threshold filtering
    - Minimum source requirements
    - Confidence assessment
    - Uncertainty handling
    - Full observability
    
    Usage:
        pipeline = ProductionRAGPipeline(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            guardrails_config=PRODUCTION_CONFIG
        )
        
        response = pipeline.query("What is machine learning?")
        print(f"{response.confidence_emoji} Confidence: {response.confidence}")
        print(f"Answer: {response.answer}")
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        llm: BaseLLM,
        chunker_config: Optional[dict] = None,
        guardrails_config: Optional[GuardrailsConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            embeddings: Embedding provider
            vectorstore: Vector store for documents
            retriever: Retriever for search
            llm: LLM for generation
            chunker_config: Config for chunker
            guardrails_config: Guardrails configuration
            system_prompt: Custom system prompt (uses strict default)
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.llm = llm
        self.chunker = ChunkerFactory.from_config(chunker_config or {})
        self.guardrails = GuardrailsValidator(guardrails_config or PRODUCTION_CONFIG)
        self.system_prompt = system_prompt or SYSTEM_PROMPT_STRICT
        
        logger.info("Initialized ProductionRAGPipeline with guardrails")

    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file into the pipeline."""
        path = Path(file_path)
        documents = LoaderFactory.load(str(path))
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning(f"No chunks created from {file_path}")
            return 0
        
        texts = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        
        self.retriever.add_documents(texts=texts, metadatas=metadatas)
        
        logger.info(f"Ingested {len(all_chunks)} chunks from {file_path}")
        return len(all_chunks)

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
    ) -> int:
        """Ingest all documents from a directory."""
        documents = LoaderFactory.load_directory(
            directory,
            recursive=recursive,
            file_types=file_types
        )
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning(f"No chunks created from {directory}")
            return 0
        
        texts = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        
        self.retriever.add_documents(texts=texts, metadatas=metadatas)
        
        logger.info(f"Ingested {len(all_chunks)} chunks from {directory}")
        return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: int = 10,
    ) -> ProductionRAGResponse:
        """
        Query the pipeline with guardrails.
        
        Args:
            question: User question
            top_k: Initial number of chunks to retrieve (before filtering)
            
        Returns:
            ProductionRAGResponse with answer and metadata
        """
        # Step 1: Retrieve more chunks than needed (guardrails will filter)
        results = self.retriever.retrieve(question, top_k=top_k)
        
        # Step 2: Validate through guardrails
        validation = self.guardrails.validate(results)
        
        # Step 3: Handle validation failure
        if not validation.is_valid:
            logger.warning(f"Guardrails rejected query: {validation.rejection_reason}")
            return ProductionRAGResponse(
                answer=get_uncertainty_response(validation.rejection_reason or ""),
                sources=[],
                query=question,
                confidence="low",
                avg_score=validation.avg_score,
                is_uncertain=True,
                validation_passed=False,
                rejection_reason=validation.rejection_reason,
            )
        
        # Step 4: Build context from filtered results
        filtered_results = validation.filtered_results
        context_chunks = filtered_results
        
        # Step 5: Build prompt with confidence context
        prompt = build_prompt_with_context(
            query=question,
            context_chunks=context_chunks,
            confidence=validation.confidence,
            include_scores=True
        )
        
        # Step 6: Add confidence guidance to system prompt
        confidence_guidance = get_confidence_guidance(validation.confidence)
        enhanced_system_prompt = f"{self.system_prompt}\n\nCONFIDENCE LEVEL: {validation.confidence.upper()}\n{confidence_guidance}"
        
        # Step 7: Generate answer
        answer = self.llm.generate(prompt, system_prompt=enhanced_system_prompt)
        
        # Step 8: Build response with full metadata
        sources = [
            {
                "content": r.content[:300] + "..." if len(r.content) > 300 else r.content,
                "metadata": r.metadata,
                "score": r.score
            }
            for r in filtered_results
        ]
        
        logger.info(
            f"Generated response: confidence={validation.confidence}, "
            f"sources={len(sources)}, avg_score={validation.avg_score:.2f}"
        )
        
        return ProductionRAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            confidence=validation.confidence,
            avg_score=validation.avg_score,
            is_uncertain=False,
            validation_passed=True,
        )

    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        return {
            "vectorstore": self.vectorstore.health_check(),
            "llm": self.llm.health_check(),
            "retriever": self.retriever.health_check(),
        }
