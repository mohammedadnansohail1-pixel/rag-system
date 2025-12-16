"""Main RAG pipeline that ties all components together."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.loaders.base import Document
from src.loaders.factory import LoaderFactory
from src.chunkers.base import Chunk
from src.chunkers.factory import ChunkerFactory
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore
from src.retrieval.base import BaseRetriever
from src.generation.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Response from the RAG pipeline.
    
    Attributes:
        answer: Generated answer
        sources: List of source documents used
        query: Original query
    """
    answer: str
    sources: List[Dict[str, Any]]
    query: str

    def __repr__(self) -> str:
        preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        return f"RAGResponse(sources={len(self.sources)}, answer='{preview}')"


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    
    Orchestrates:
    - Document loading
    - Chunking
    - Embedding and indexing
    - Retrieval
    - Answer generation
    
    Usage:
        pipeline = RAGPipeline(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            chunker_config={"strategy": "recursive", "chunk_size": 512}
        )
        
        # Index documents
        pipeline.ingest_directory("./data/documents")
        
        # Query
        response = pipeline.query("What is RAG?")
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        llm: BaseLLM,
        chunker_config: Optional[dict] = None,
    ):
        """
        Args:
            embeddings: Embedding provider
            vectorstore: Vector store for documents
            retriever: Retriever for search
            llm: LLM for generation
            chunker_config: Config for chunker (strategy, chunk_size, etc.)
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.llm = llm
        self.chunker = ChunkerFactory.from_config(chunker_config or {})
        
        logger.info("Initialized RAGPipeline")

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
        
        # Add to retriever (handles embedding + vectorstore)
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
            file_types: Filter by extensions (e.g., ['.txt', '.pdf'])
            
        Returns:
            Total number of chunks indexed
        """
        # Load all documents
        documents = LoaderFactory.load_directory(
            directory,
            recursive=recursive,
            file_types=file_types
        )
        
        # Chunk all documents
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning(f"No chunks created from {directory}")
            return 0
        
        # Add to retriever
        texts = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        
        self.retriever.add_documents(texts=texts, metadatas=metadatas)
        
        logger.info(f"Ingested {len(all_chunks)} chunks from {directory}")
        return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            
        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve relevant chunks
        results = self.retriever.retrieve(question, top_k=top_k)
        
        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=question
            )
        
        # Extract context and sources
        context = [r.content for r in results]
        sources = [
            {"content": r.content[:200], "metadata": r.metadata, "score": r.score}
            for r in results
        ]
        
        # Generate answer
        answer = self.llm.generate_with_context(
            query=question,
            context=context
        )
        
        logger.info(f"Generated response for: {question[:50]}...")
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question
        )

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.
        
        Returns:
            Dict with component health status
        """
        return {
            "vectorstore": self.vectorstore.health_check(),
            "llm": self.llm.health_check(),
            "retriever": self.retriever.health_check(),
        }
