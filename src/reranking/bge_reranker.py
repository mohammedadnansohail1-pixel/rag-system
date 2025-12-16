"""BGE reranker using sentence-transformers."""
import logging
from typing import List, Optional, Dict, Any

from sentence_transformers import CrossEncoder

from src.reranking.base import BaseReranker, RerankResult
from src.reranking.factory import register_reranker

logger = logging.getLogger(__name__)

# Available BGE reranker models
BGE_MODELS = {
    "base": "BAAI/bge-reranker-base",        # 278M params, English
    "large": "BAAI/bge-reranker-large",      # 560M params, English
    "v2-m3": "BAAI/bge-reranker-v2-m3",      # 568M params, Multilingual
}

DEFAULT_MODEL = "base"


@register_reranker("bge")
class BGEReranker(BaseReranker):
    """
    Reranker using BGE (BAAI General Embedding) models.
    
    BGE rerankers are cross-encoders trained with:
    - Multi-task learning (search, QA, NLI)
    - Knowledge distillation from larger LLMs
    - Harder negative mining for better discrimination
    
    Advantages over standard cross-encoders:
    - Better accuracy on benchmarks (BEIR, MTEB)
    - Multilingual support (v2-m3)
    - Instruction-tuning capability
    
    Attributes:
        model: The CrossEncoder model instance
        model_name: Full HuggingFace model path
        model_variant: Short name (base, large, v2-m3)
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize BGE reranker.
        
        Args:
            model: Model variant ("base", "large", "v2-m3") or full HF path
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
                   None = auto-detect
            max_length: Maximum sequence length for tokenization
        """
        # Resolve model name
        if model in BGE_MODELS:
            self.model_variant = model
            self.model_name = BGE_MODELS[model]
        else:
            self.model_variant = "custom"
            self.model_name = model
        
        self.max_length = max_length
        self._device = device
        self._model: Optional[CrossEncoder] = None
        
        logger.info(
            f"Initialized BGEReranker with model={self.model_name} "
            f"(variant={self.model_variant})"
        )
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy load model on first use."""
        if self._model is None:
            logger.info(f"Loading BGE reranker model: {self.model_name}")
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
        
        # Convert numpy array to list of floats
        scores = [float(s) for s in scores]
        
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
            test_score = self.model.predict([["test query", "test document"]])
            return test_score is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"BGEReranker(model='{self.model_name}', variant='{self.model_variant}')"
