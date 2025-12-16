"""Sparse encoders for hybrid retrieval - pluggable implementations."""
import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.core.types import SparseVector

logger = logging.getLogger(__name__)


class BaseSparseEncoder(ABC):
    """Abstract base class for sparse encoders."""

    @abstractmethod
    def encode(self, text: str) -> SparseVector:
        """Encode text to sparse vector."""
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[SparseVector]:
        """Encode multiple texts."""
        pass
    
    def fit(self, corpus: List[str]) -> None:
        """
        Fit encoder on corpus (optional, some encoders need this).
        
        Args:
            corpus: List of documents to fit on
        """
        pass
    
    def health_check(self) -> bool:
        """Check if encoder is operational."""
        try:
            _ = self.encode("test")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class BM25Encoder(BaseSparseEncoder):
    """
    BM25 sparse encoder - industry standard for lexical search.
    
    CPU-only, scales infinitely, no GPU required.
    Used by Elasticsearch, Pinecone, Weaviate in production.
    
    BM25 scoring: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter (0=no normalization, 1=full)
            epsilon: Floor for IDF to handle unseen terms
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Vocabulary mapping: token -> index
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._doc_count: int = 0
        self._fitted: bool = False
        
        logger.info(f"Initialized BM25Encoder: k1={k1}, b={b}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        # Basic preprocessing - can be extended
        text = text.lower()
        # Remove punctuation, keep alphanumeric
        tokens = []
        current_token = []
        for char in text:
            if char.isalnum():
                current_token.append(char)
            elif current_token:
                tokens.append(''.join(current_token))
                current_token = []
        if current_token:
            tokens.append(''.join(current_token))
        return tokens
    
    def fit(self, corpus: List[str]) -> None:
        """
        Fit BM25 on corpus to compute IDF scores.
        
        Args:
            corpus: List of documents
        """
        if not corpus:
            return
        
        self._doc_count = len(corpus)
        doc_freqs: Counter = Counter()
        total_len = 0
        
        # Build vocabulary and document frequencies
        for doc in corpus:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            
            # Unique tokens in this document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)
        
        self._avg_doc_len = total_len / self._doc_count if self._doc_count else 0
        
        # Compute IDF: log((N - df + 0.5) / (df + 0.5))
        for token, df in doc_freqs.items():
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)
            self._idf[token] = max(idf, self.epsilon)
        
        self._fitted = True
        logger.info(f"BM25 fitted: {len(self._vocab)} terms, {self._doc_count} docs, avg_len={self._avg_doc_len:.1f}")
    
    def encode(self, text: str) -> SparseVector:
        """Encode text to BM25 sparse vector."""
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        tf_counter = Counter(tokens)
        
        indices = []
        values = []
        
        for token, tf in tf_counter.items():
            # Get or create vocab index
            if token not in self._vocab:
                if self._fitted:
                    # Unknown token after fitting - skip or use default
                    continue
                self._vocab[token] = len(self._vocab)
            
            idx = self._vocab[token]
            
            # BM25 score
            idf = self._idf.get(token, self.epsilon)
            
            if self._avg_doc_len > 0:
                # Full BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                score = idf * (numerator / denominator)
            else:
                # Fallback: simple TF-IDF style
                score = idf * (1 + math.log(1 + tf))
            
            if score > 0:
                indices.append(idx)
                values.append(score)
        
        return SparseVector(indices=indices, values=values)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[SparseVector]:
        """Encode multiple texts (CPU, no batching needed)."""
        return [self.encode(text) for text in texts]
    
    def __repr__(self) -> str:
        return f"BM25Encoder(k1={self.k1}, b={self.b}, vocab_size={len(self._vocab)})"


class TFIDFEncoder(BaseSparseEncoder):
    """
    TF-IDF sparse encoder - lightweight baseline.
    
    CPU-only, fast, good for smaller corpora.
    """
    
    def __init__(self, max_features: Optional[int] = None):
        """
        Args:
            max_features: Maximum vocabulary size (None = unlimited)
        """
        self.max_features = max_features
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._fitted: bool = False
        
        logger.info(f"Initialized TFIDFEncoder: max_features={max_features}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = []
        current = []
        for char in text:
            if char.isalnum():
                current.append(char)
            elif current:
                tokens.append(''.join(current))
                current = []
        if current:
            tokens.append(''.join(current))
        return tokens
    
    def fit(self, corpus: List[str]) -> None:
        """Fit TF-IDF on corpus."""
        if not corpus:
            return
        
        doc_count = len(corpus)
        doc_freqs: Counter = Counter()
        term_freqs: Counter = Counter()
        
        for doc in corpus:
            tokens = self._tokenize(doc)
            term_freqs.update(tokens)
            unique = set(tokens)
            doc_freqs.update(unique)
        
        # Select top features if limited
        if self.max_features:
            top_terms = [t for t, _ in term_freqs.most_common(self.max_features)]
        else:
            top_terms = list(doc_freqs.keys())
        
        # Build vocab and IDF
        for i, term in enumerate(top_terms):
            self._vocab[term] = i
            df = doc_freqs[term]
            self._idf[term] = math.log(doc_count / (1 + df)) + 1
        
        self._fitted = True
        logger.info(f"TF-IDF fitted: {len(self._vocab)} terms")
    
    def encode(self, text: str) -> SparseVector:
        """Encode text to TF-IDF sparse vector."""
        tokens = self._tokenize(text)
        tf_counter = Counter(tokens)
        doc_len = len(tokens) or 1
        
        indices = []
        values = []
        
        for token, tf in tf_counter.items():
            if token not in self._vocab:
                continue
            
            idx = self._vocab[token]
            idf = self._idf.get(token, 1.0)
            
            # TF-IDF: normalized TF * IDF
            tfidf = (tf / doc_len) * idf
            
            if tfidf > 0:
                indices.append(idx)
                values.append(tfidf)
        
        return SparseVector(indices=indices, values=values)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[SparseVector]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def __repr__(self) -> str:
        return f"TFIDFEncoder(vocab_size={len(self._vocab)})"


class SpladeEncoder(BaseSparseEncoder):
    """
    SPLADE sparse encoder - learned sparse representations.
    
    GPU recommended, best quality, higher resource requirements.
    Expands queries with semantically related terms.
    """
    
    DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 4,
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device (None = auto-detect)
            max_length: Maximum sequence length
            batch_size: Batch size for encoding (memory management)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Lazy imports to avoid loading torch if not needed
        self._torch = None
        self._model = None
        self._tokenizer = None
        
        if device is None:
            import torch
            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        
        logger.info(f"Initialized SpladeEncoder: model={model_name}, device={self._device}")
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            import torch
            from transformers import AutoModelForMaskedLM
            
            self._torch = torch
            logger.info(f"Loading SPLADE model: {self.model_name}")
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"SPLADE model loaded on {self._device}")
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    def encode(self, text: str) -> SparseVector:
        """Encode single text."""
        return self.encode_batch([text])[0]
    
    def encode_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[SparseVector]:
        """
        Encode multiple texts with memory-safe batching.
        
        Args:
            texts: List of texts
            batch_size: Override default batch size
        """
        if not texts:
            return []
        
        import torch
        
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            
            # SPLADE: log(1 + ReLU(x)) with attention mask
            logits = logits * attention_mask
            activated = torch.log1p(torch.relu(logits))
            sparse_vecs = torch.max(activated, dim=1).values
            
            for vec in sparse_vecs:
                nonzero = torch.nonzero(vec, as_tuple=True)[0]
                indices = nonzero.cpu().tolist()
                values = vec[nonzero].cpu().tolist()
                results.append(SparseVector(indices=indices, values=values))
            
            # Clear GPU memory
            del inputs, outputs, logits, activated, sparse_vecs
            torch.cuda.empty_cache()
        
        return results
    
    def decode_tokens(self, sparse_vec: SparseVector, top_k: int = 10) -> List[Tuple[str, float]]:
        """Decode sparse vector to human-readable tokens."""
        pairs = sorted(zip(sparse_vec.indices, sparse_vec.values), key=lambda x: -x[1])
        return [(self.tokenizer.decode([idx]).strip(), w) for idx, w in pairs[:top_k]]
    
    def __repr__(self) -> str:
        return f"SpladeEncoder(model='{self.model_name}', device='{self._device}')"


class SparseEncoderFactory:
    """Factory for creating sparse encoders based on configuration."""
    
    _encoders = {
        "bm25": BM25Encoder,
        "tfidf": TFIDFEncoder,
        "splade": SpladeEncoder,
    }
    
    @classmethod
    def create(
        cls,
        encoder_type: str = "bm25",
        **kwargs
    ) -> BaseSparseEncoder:
        """
        Create sparse encoder by type.
        
        Args:
            encoder_type: One of 'bm25', 'tfidf', 'splade'
            **kwargs: Encoder-specific arguments
            
        Returns:
            Configured sparse encoder
        """
        encoder_type = encoder_type.lower()
        
        if encoder_type not in cls._encoders:
            available = ", ".join(cls._encoders.keys())
            raise ValueError(f"Unknown encoder: {encoder_type}. Available: {available}")
        
        encoder_class = cls._encoders[encoder_type]
        logger.info(f"Creating sparse encoder: {encoder_type}")
        
        return encoder_class(**kwargs)
    
    @classmethod
    def from_config(cls, config: Dict) -> BaseSparseEncoder:
        """
        Create encoder from config dict.
        
        Example config:
            {"type": "bm25", "k1": 1.5, "b": 0.75}
            {"type": "splade", "device": "cuda", "batch_size": 4}
        """
        config = config.copy()
        encoder_type = config.pop("type", "bm25")
        return cls.create(encoder_type, **config)
    
    @classmethod
    def list_encoders(cls) -> List[str]:
        """List available encoder types."""
        return list(cls._encoders.keys())
    
    @classmethod
    def register(cls, name: str, encoder_class: type):
        """Register custom encoder."""
        cls._encoders[name.lower()] = encoder_class
        logger.info(f"Registered sparse encoder: {name}")
