"""
Domain-aware RAG Pipeline.

Configures entire pipeline based on document domain.
Single entry point for production use.

Usage:
    from src.rag_pipeline import RAGPipeline
    from src.config import FINANCIAL
    
    pipeline = RAGPipeline(domain=FINANCIAL)
    pipeline.index_documents(documents)
    answer = pipeline.query("What was revenue in 2024?")
"""
import logging
import hashlib
from typing import List, Optional, Dict, Any

from src.config import get_domain_config, DomainConfig, GENERAL
from src.loaders.base import Document
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.base import RetrievalResult
from src.generation.factory import LLMFactory
from src.generation.base import BaseLLM

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Domain-aware RAG pipeline.
    
    Automatically configures:
    - Chunking strategy
    - Embedding model
    - Sparse encoder
    - Query expansion
    - Retrieval settings
    - Generation prompts
    
    Example:
        # Financial documents
        pipeline = RAGPipeline(domain="financial", collection_name="company_10k")
        pipeline.index_documents(sec_documents)
        result = pipeline.query("What was the operating margin?")
        
        # Technical docs
        pipeline = RAGPipeline(domain="technical", collection_name="api_docs")
        pipeline.index_documents(docs)
        result = pipeline.query("How do I authenticate?")
    """
    
    def __init__(
        self,
        domain: str = GENERAL,
        collection_name: str = "documents",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        config_override: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            domain: Domain type (financial, technical, legal, general)
            collection_name: Vector store collection name
            embedding_model: Ollama embedding model
            llm_model: Ollama LLM model
            config_override: Override specific config values
        """
        self.domain = domain
        self.collection_name = collection_name
        
        # Get domain configuration
        self.config = get_domain_config(domain)
        logger.info(f"Initialized RAGPipeline for domain: {domain}")
        
        # Initialize components
        self._init_embeddings(embedding_model)
        self._init_vectorstore()
        self._init_retriever()
        self._init_llm(llm_model)
        self._init_expander()
        
        # Track indexed documents
        self._indexed_count = 0
    
    def _init_embeddings(self, model: str):
        """Initialize embeddings."""
        self.embeddings = EmbeddingsFactory.create("ollama", model=model)
        logger.info(f"Embeddings: {model}")
    
    def _init_vectorstore(self):
        """Initialize vector store based on config."""
        # Use hybrid store for sparse+dense
        self.vectorstore = QdrantHybridStore(
            collection_name=self.collection_name,
            dense_dimensions=768,
            recreate_collection=True,
        )
        logger.info(f"Vector store: {self.collection_name}")
    
    def _init_retriever(self):
        """Initialize retriever with domain settings."""
        sparse_config = {
            "type": self.config.retrieval.sparse_encoder,
            "k1": self.config.retrieval.bm25_k1,
            "b": self.config.retrieval.bm25_b,
        }
        
        self.retriever = HybridRetriever(
            embeddings=self.embeddings,
            vectorstore=self.vectorstore,
            sparse_encoder=sparse_config,
            top_k=self.config.retrieval.top_k,
        )
        logger.info(f"Retriever: hybrid with {self.config.retrieval.sparse_encoder}")
    
    def _init_llm(self, model: str):
        """Initialize LLM."""
        self.llm = LLMFactory.create("ollama", model=model)
        logger.info(f"LLM: {model}")
    
    def _init_expander(self):
        """Initialize query expander if configured."""
        self.expander = self.config.create_query_expander(llm=self.llm)
        if self.expander:
            logger.info(f"Query expansion: {self.config.expansion.method}")
    
    def index_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> int:
        """
        Index documents into the pipeline.
        
        Args:
            documents: List of Document objects
            show_progress: Whether to show progress
            
        Returns:
            Number of chunks indexed
        """
        # Create chunker with domain settings
        chunker = self.config.create_chunker()
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        if show_progress:
            print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Clean chunks
        texts = []
        metadatas = []
        for chunk in all_chunks:
            content = ''.join(c for c in chunk.content if c.isprintable() or c in '\n\t ')
            if len(content) > 50:
                texts.append(content)
                metadatas.append(chunk.metadata)
        
        # Index
        self.retriever.add_documents(texts, metadatas)
        self._indexed_count = len(texts)
        
        if show_progress:
            print(f"Indexed {len(texts)} chunks")
        
        return len(texts)
    
    def index_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> int:
        """
        Index raw texts directly.
        
        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text
            
        Returns:
            Number of texts indexed
        """
        self.retriever.add_documents(texts, metadatas)
        self._indexed_count = len(texts)
        return len(texts)
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
            top_k: Override retrieval count
            return_sources: Include source chunks in response
            
        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        k = top_k or self.config.retrieval.top_k
        
        # Expand query if configured
        search_query = question
        if self.expander:
            expanded = self.expander.expand(question)
            search_query = expanded.expanded
            logger.debug(f"Expanded query: {search_query[:100]}...")
        
        # Retrieve
        results = self.retriever.retrieve(search_query, top_k=k)
        
        if not results:
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "sources": [] if return_sources else None,
            }
        
        # Generate answer
        context = [r.content for r in results]
        
        # Use domain-specific system prompt
        answer = self.llm.generate_with_context(
            question,
            context,
            system_prompt=self.config.generation.system_prompt,
        )
        
        response = {"answer": answer}
        
        if return_sources:
            response["sources"] = [
                {
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results[:3]
            ]
        
        return response
    
    def get_config_summary(self) -> str:
        """Get configuration summary."""
        return self.config.summary()
    
    @property
    def indexed_count(self) -> int:
        """Number of indexed chunks."""
        return self._indexed_count
    
    def cleanup(self):
        """Delete collection and cleanup."""
        self.vectorstore._client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
