"""Dense retriever using embeddings and vector store."""
import logging
from typing import List, Optional, Dict, Any, Tuple

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.factory import register_retriever
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


@register_retriever("dense")
class DenseRetriever(BaseRetriever):
    """
    Dense retriever using embedding similarity.
    
    Includes optional pre-embedding quality analysis to filter
    or flag problematic chunks before embedding.

    Usage:
        retriever = DenseRetriever(
            embeddings=embeddings,
            vectorstore=vectorstore,
            top_k=5,
            quality_gate=True,  # Enable quality filtering
        )
        results = retriever.retrieve("What is RAG?")
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        quality_gate: bool = False,
        quality_gate_mode: str = "filter",  # "filter" or "warn"
        quality_config: Optional[str] = "default",
    ):
        """
        Args:
            embeddings: Embedding provider
            vectorstore: Vector store for search
            top_k: Default number of results
            score_threshold: Minimum score filter
            quality_gate: Enable pre-embedding quality analysis
            quality_gate_mode: "filter" to skip bad chunks, "warn" to log only
            quality_config: Analyzer config name ("default", "financial", etc.)
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.quality_gate = quality_gate
        self.quality_gate_mode = quality_gate_mode
        self.quality_config = quality_config
        
        # Lazy-load analyzer
        self._analyzer = None

        logger.info(
            f"Initialized DenseRetriever: "
            f"top_k={top_k}, score_threshold={score_threshold}, "
            f"quality_gate={quality_gate}"
        )

    @property
    def analyzer(self):
        """Lazy-load embedding analyzer."""
        if self._analyzer is None and self.quality_gate:
            try:
                from src.embedding_analyzer import EmbeddingAnalyzer
                from src.embedding_analyzer.config_loader import load_config
                
                config = load_config(self.quality_config)
                self._analyzer = EmbeddingAnalyzer.from_config(config)
                logger.info(f"Loaded embedding analyzer with '{self.quality_config}' config")
            except ImportError as e:
                logger.warning(f"Could not load embedding analyzer: {e}")
                self.quality_gate = False
        return self._analyzer

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
        
        If quality_gate is enabled, analyzes chunks before embedding
        and either filters or warns about problematic content.

        Args:
            texts: List of text content
            metadatas: Optional metadata for each text

        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Quality gate: analyze before embedding
        if self.quality_gate and self.analyzer:
            texts, metadatas = self._apply_quality_gate(texts, metadatas)
        
        if not texts:
            logger.warning("No texts to add after quality gate")
            return []

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

    def _apply_quality_gate(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Apply quality analysis to texts before embedding.
        
        Args:
            texts: List of text content
            metadatas: List of metadata dicts
            
        Returns:
            Filtered (texts, metadatas) tuple
        """
        logger.info(f"Analyzing {len(texts)} chunks with quality gate...")
        
        reports = self.analyzer.analyze_many(texts)
        stats = self.analyzer.get_summary_stats(reports)
        
        # Log stats
        logger.info(
            f"Quality analysis: {stats['passed']}/{stats['total_analyzed']} passed "
            f"({stats['pass_rate']:.1%}), "
            f"{stats['total_critical']}C/{stats['total_warning']}W/{stats['total_info']}I issues"
        )
        
        if stats['top_issues']:
            top_3 = list(stats['top_issues'].items())[:3]
            logger.info(f"Top issues: {top_3}")
        
        # Filter or warn based on mode
        if self.quality_gate_mode == "filter":
            # Filter out failed chunks
            filtered_texts = []
            filtered_metadatas = []
            
            for text, metadata, report in zip(texts, metadatas, reports):
                if report.overall_passed:
                    # Add quality score to metadata
                    metadata['quality_score'] = report.overall_score
                    filtered_texts.append(text)
                    filtered_metadatas.append(metadata)
                else:
                    # Log filtered chunk
                    issues = [f"{i.code}" for i in report.all_issues if i.severity.value == "critical"]
                    logger.debug(f"Filtered chunk: {text[:50]}... Issues: {issues}")
            
            filtered_count = len(texts) - len(filtered_texts)
            if filtered_count > 0:
                logger.warning(f"Filtered {filtered_count} chunks that failed quality gate")
            
            return filtered_texts, filtered_metadatas
        
        else:
            # Warn mode: just add quality metadata
            for metadata, report in zip(metadatas, reports):
                metadata['quality_score'] = report.overall_score
                metadata['quality_passed'] = report.overall_passed
                if not report.overall_passed:
                    metadata['quality_issues'] = [i.code for i in report.all_issues]
            
            failed_count = sum(1 for r in reports if not r.overall_passed)
            if failed_count > 0:
                logger.warning(
                    f"{failed_count} chunks failed quality check but were embedded (warn mode)"
                )
            
            return texts, metadatas

    def health_check(self) -> bool:
        """Check if retriever is operational."""
        return self.vectorstore.health_check()
