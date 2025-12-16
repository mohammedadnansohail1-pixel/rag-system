"""Sparse encoders for hybrid retrieval."""
import logging
from abc import ABC, abstractmethod

from typing import Dict, List, Optional, Tuple
from src.core.types import SparseVector

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)


class BaseSparseEncoder(ABC):
    """Abstract base class for sparse encoders."""
    
    @abstractmethod
    def encode(self, text: str) -> SparseVector:
        """
        Encode text to sparse vector.
        
        Args:
            text: Input text
            
        Returns:
            SparseVector with token indices and weights
        """
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[SparseVector]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of SparseVectors
        """
        pass


class SpladeEncoder(BaseSparseEncoder):
    """
    SPLADE sparse encoder for learned sparse representations.
    
    SPLADE (SParse Lexical AnD Expansion) learns which terms are
    important and expands queries/documents with related terms.
    
    Advantages over BM25:
    - Learns term importance (not just frequency)
    - Expands with semantically related terms
    - Better zero-shot performance
    
    Attributes:
        model: The SPLADE transformer model
        tokenizer: Tokenizer for the model
    """
    
    # Default model - good balance of speed and accuracy
    DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        """
        Initialize SPLADE encoder.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on (None = auto-detect)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        
        self._model: Optional[AutoModelForMaskedLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        
        logger.info(f"Initialized SpladeEncoder: model={model_name}, device={self._device}")
    
    @property
    def model(self) -> AutoModelForMaskedLM:
        """Lazy load model."""
        if self._model is None:
            logger.info(f"Loading SPLADE model: {self.model_name}")
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"Model loaded on {self._device}")
        return self._model
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    def encode(self, text: str) -> SparseVector:
        """
        Encode text to sparse vector using SPLADE.
        
        Args:
            text: Input text
            
        Returns:
            SparseVector with token indices and weights
        """
        return self.encode_batch([text])[0]
    
    def encode_batch(self, texts: List[str]) -> List[SparseVector]:
        """
        Encode multiple texts to sparse vectors.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of SparseVectors
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self._device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # SPLADE aggregation: max over sequence, then ReLU + log
        # Shape: (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
        logits = outputs.logits
        
        # Apply attention mask to zero out padding
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        logits = logits * attention_mask
        
        # Max pooling over sequence length
        max_logits, _ = torch.max(logits, dim=1)
        
        # SPLADE activation: log(1 + ReLU(x))
        sparse_vecs = torch.log1p(torch.relu(max_logits))
        
        # Convert to SparseVector objects
        results = []
        for vec in sparse_vecs:
            # Get non-zero indices and values
            non_zero_mask = vec > 0
            indices = torch.where(non_zero_mask)[0].cpu().tolist()
            values = vec[non_zero_mask].cpu().tolist()
            
            results.append(SparseVector(indices=indices, values=values))
        
        logger.debug(f"Encoded {len(texts)} texts, avg nnz: {sum(len(r.indices) for r in results) / len(results):.0f}")
        return results
    
    def decode_tokens(self, sparse_vec: SparseVector, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Decode sparse vector to human-readable tokens.
        
        Useful for debugging/understanding what SPLADE learned.
        
        Args:
            sparse_vec: Sparse vector to decode
            top_k: Number of top tokens to return
            
        Returns:
            List of (token, weight) tuples sorted by weight
        """
        pairs = list(zip(sparse_vec.indices, sparse_vec.values))
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        decoded = []
        for idx, weight in pairs[:top_k]:
            token = self.tokenizer.decode([idx])
            decoded.append((token.strip(), weight))
        
        return decoded
    
    def health_check(self) -> bool:
        """Check if encoder is operational."""
        try:
            _ = self.encode("test")
            return True
        except Exception as e:
            logger.error(f"SPLADE health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"SpladeEncoder(model='{self.model_name}')"
