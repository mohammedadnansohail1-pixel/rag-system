"""Enhanced RAG pipeline with hybrid retrieval and reranking."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.loaders.base import Document
from src.loaders.factory import LoaderFactory
from src.chunkers.base import Chunk
from src.chunkers.factory import ChunkerFactory
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.generation.base import BaseLLM
from src.reranking.base import BaseReranker, RerankResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponseV2:
    """
    Enhanced response from the RAG pipeline.

    Attributes:
        answer: Generated answer
        sources: List of source documents used
        query: Original query
        retrieval_scores: Scores from initial retrieval
        rerank_scores: Scores after reranking (if enabled)
        metadata: Additional pipeline metadata
    """
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    retrieval_scores: List[float] = field(default_factory=list)
    rerank_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        reranked = "reranked" if self.rerank_scores else "not reranked"
        return f"RAGResponseV2(sources={len(self.sources)}, {reranked}, answer='{preview}')"


class RAGPipelineV2:
    """
    Enhanced RAG pipeline with hybrid retrieval and reranking.

    Two-stage retrieval:
    1. Hybrid retrieval (dense + sparse) - high recall, get candidates
    2. Reranking (cross-encoder/BGE) - high precision, best to top

    Usage:
        pipeline = RAGPipelineV2(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            reranker=reranker,  # Optional
        )

        # Index documents
        pipeline.ingest_directory("./data/documents")

        # Query with reranking
        response = pipeline.query(
            "What is RAG?",
            retrieval_top_k=20,  # Get 20 candidates
            rerank_top_n=5,      # Rerank to top 5
        )
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        llm: BaseLLM,
        reranker: Optional[BaseReranker] = None,
        chunker_config: Optional[dict] = None,
    ):
        """
        Args:
            embeddings: Embedding provider
            vectorstore: Vector store for documents
            retriever: Retriever for search (dense or hybrid)
            llm: LLM for generation
            reranker: Optional reranker for precision (cross-encoder/BGE)
            chunker_config: Config for chunker
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.chunker = ChunkerFactory.from_config(chunker_config or {})

        logger.info(
            f"Initialized RAGPipelineV2 "
            f"(reranker={'enabled' if reranker else 'disabled'})"
        )

    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file into the pipeline.

        Args:
            file_path: Path to file

        Returns:
            Number of chunks indexed
        """
        path = Path(file_path)

        # Load document
        documents = LoaderFactory.load(str(path))

        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning(f"No chunks created from {file_path}")
            return 0

        # Add to retriever
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
        """
        Ingest all documents from a directory.

        Args:
            directory: Path to directory
            recursive: Search subdirectories
            file_types: Filter by extensions

        Returns:
            Total number of chunks indexed
        """
        documents = LoaderFactory.load_directory(
            directory,
            recursive=recursive,
            file_types=file_types
        )

        all_chunks: List[Chunk] = []
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
        retrieval_top_k: int = 20,
        rerank_top_n: int = 5,
        use_reranker: Optional[bool] = None,
    ) -> RAGResponseV2:
        """
        Query the RAG pipeline with optional reranking.

        Two-stage process:
        1. Retrieve top_k candidates (fast, high recall)
        2. Rerank to top_n (slow, high precision)

        Args:
            question: User question
            retrieval_top_k: Candidates to retrieve (stage 1)
            rerank_top_n: Final results after reranking (stage 2)
            use_reranker: Override reranker usage (None = use if available)

        Returns:
            RAGResponseV2 with answer, sources, and scores
        """
        # Determine if we should rerank
        should_rerank = (use_reranker if use_reranker is not None else True) and (self.reranker is not None)
        
        # Stage 1: Retrieve candidates
        if should_rerank:
            # Get more candidates for reranking
            results = self.retriever.retrieve(question, top_k=retrieval_top_k)
        else:
            # No reranking, just get final count
            results = self.retriever.retrieve(question, top_k=rerank_top_n)

        if not results:
            return RAGResponseV2(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=question,
                metadata={"retrieval_count": 0, "reranked": False}
            )

        retrieval_scores = [r.score for r in results]

        # Stage 2: Rerank if enabled
        rerank_scores = None
        if should_rerank and self.reranker:
            logger.debug(f"Reranking {len(results)} candidates to top {rerank_top_n}")
            
            reranked = self.reranker.rerank(
                query=question,
                documents=[r.content for r in results],
                metadatas=[r.metadata for r in results],
                top_n=rerank_top_n,
            )
            
            # Convert back to format for generation
            results = [
                RetrievalResult(
                    content=r.content,
                    metadata=r.metadata,
                    score=r.score,
                )
                for r in reranked
            ]
            rerank_scores = [r.score for r in reranked]

        # Build context and sources
        context = [r.content for r in results]
        sources = [
            {
                "content": r.content[:200],
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in results
        ]

        # Stage 3: Generate answer
        answer = self.llm.generate_with_context(
            query=question,
            context=context
        )

        logger.info(
            f"Generated response for: {question[:50]}... "
            f"(reranked={should_rerank}, sources={len(sources)})"
        )

        return RAGResponseV2(
            answer=answer,
            sources=sources,
            query=question,
            retrieval_scores=retrieval_scores,
            rerank_scores=rerank_scores,
            metadata={
                "retrieval_count": len(retrieval_scores),
                "final_count": len(sources),
                "reranked": should_rerank,
            }
        )

    def query_compare(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, RAGResponseV2]:
        """
        Query with and without reranking for comparison.

        Useful for A/B testing and demonstrating reranker value.

        Args:
            question: User question
            top_k: Number of final results

        Returns:
            Dict with 'with_reranking' and 'without_reranking' responses
        """
        if not self.reranker:
            raise ValueError("No reranker configured for comparison")

        return {
            "without_reranking": self.query(
                question,
                retrieval_top_k=top_k,
                rerank_top_n=top_k,
                use_reranker=False,
            ),
            "with_reranking": self.query(
                question,
                retrieval_top_k=top_k * 4,  # Get more candidates
                rerank_top_n=top_k,
                use_reranker=True,
            ),
        }

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.

        Returns:
            Dict with component health status
        """
        status = {
            "vectorstore": self.vectorstore.health_check(),
            "llm": self.llm.health_check(),
            "retriever": self.retriever.health_check(),
        }
        
        if self.reranker:
            status["reranker"] = self.reranker.health_check()
        
        return status

    def __repr__(self) -> str:
        return (
            f"RAGPipelineV2("
            f"retriever={self.retriever.__class__.__name__}, "
            f"reranker={self.reranker.__class__.__name__ if self.reranker else 'None'})"
        )
