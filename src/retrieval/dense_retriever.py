"""Dense retriever using embeddings and vector store."""

import logging
from typing import List, Optional, Dict, Any

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.factory import register_retriever
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


@register_retriever("dense")
class DenseRetriever(BaseRetriever):
    """
    Dense retriever using embedding similarity.
    
    Usage:
        retriever = DenseRetriever(
            embeddings=embeddings,
            vectorstore=vectorstore,
            top_k=5
        )
        results = retriever.retrieve("What is RAG?")
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """
        Args:
            embeddings: Embedding provider
            vectorstore: Vector store for search
            top_k: Default number of results
            score_threshold: Minimum score filter
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        logger.info(
            f"Initialized DenseRetriever: "
            f"top_k={top_k}, score_threshold={score_threshold}"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using dense embeddings.
        
        Args:
            query: Search query text
            top_k: Number of results (overrides default)
            
        Returns:
            List of RetrievalResult objects
        """
        k = top_k or self.top_k
        
        # Embed the query
        query_embedding = self.embeddings.embed_text(query)
        
        # Search vector store
        search_results = self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=k,
            score_threshold=self.score_threshold,
        )
        
        # Convert to RetrievalResult
        results = [
            RetrievalResult(
                content=sr.content,
                metadata=sr.metadata,
                score=sr.score
            )
            for sr in search_results
        ]
        
        logger.debug(f"Retrieved {len(results)} results for query: {query[:50]}...")
        return results

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add documents to the index.
        
        Args:
            texts: List of text content
            metadatas: Optional metadata for each text
            
        Returns:
            List of document IDs
        """
        # Generate embeddings
        embeddings = self.embeddings.embed_batch(texts)
        
        # Add to vector store
        ids = self.vectorstore.add(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(ids)} documents to index")
        return ids

    def health_check(self) -> bool:
        """Check if retriever is operational."""
        return self.vectorstore.health_check()
