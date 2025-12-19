"""Cross-encoder reranker using sentence-transformers."""
import logging
from typing import List, Optional, Dict, Any

from sentence_transformers import CrossEncoder

from src.reranking.base import BaseReranker, RerankResult
from src.reranking.factory import register_reranker

logger = logging.getLogger(__name__)

# Default model - good balance of speed and accuracy
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@register_reranker("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """
    Reranker using cross-encoder architecture.
    
    Cross-encoders encode query and document together,
    allowing deep interaction between them for better
    relevance scoring than bi-encoders.
    
    Attributes:
        model: The CrossEncoder model instance
        model_name: Name/path of the loaded model
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model: Model name or path (HuggingFace hub or local)
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
                   None = auto-detect
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model
        self.max_length = max_length
        self._device = device
        self._model: Optional[CrossEncoder] = None
        
        logger.info(f"Initialized CrossEncoderReranker with model={model}")
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy load model on first use."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self._device,
            )
            logger.info(f"Model loaded on device: {self._model.device}")
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query text
            documents: List of document texts to rerank
            metadatas: Optional metadata for each document
            top_n: Return only top N results (None = return all)
            
        Returns:
            List of RerankResult objects sorted by relevance (best first)
        """
        if not documents:
            return []
        
        # Create query-document pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Convert numpy array to list of floats and normalize with sigmoid
        import math
        scores = [1 / (1 + math.exp(-float(s))) for s in scores]  # Sigmoid normalization to 0-1
        
        # Build results using helper from base class
        results = self._build_results(documents, metadatas, scores, top_n)
        
        logger.debug(f"Reranking complete. Top score: {results[0].score:.3f}")
        return results
    
    def health_check(self) -> bool:
        """
        Check if reranker is operational.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try a minimal prediction
            test_score = self.model.predict([["test query", "test document"]])
            return test_score is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model='{self.model_name}')"
