"""MedCPT - Medical domain-specific embeddings from NCBI."""

import logging
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel

from src.embeddings.base import BaseEmbeddings

logger = logging.getLogger(__name__)


class MedCPTEmbeddings(BaseEmbeddings):
    """
    MedCPT embeddings - medical domain-specific.
    
    Uses separate encoders for queries and documents:
    - ncbi/MedCPT-Query-Encoder (for questions)
    - ncbi/MedCPT-Article-Encoder (for documents)
    
    768 dimensions, optimized for medical text.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize MedCPT embeddings.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model_name = "ncbi/MedCPT"
        self._dimensions = 768
        
        logger.info(f"Loading MedCPT models on {device}...")
        
        # Query encoder (for search queries)
        self.query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
        self.query_model.eval()
        
        # Article encoder (for documents)
        self.article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.article_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
        self.article_model.eval()
        
        logger.info("MedCPT models loaded successfully")
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name
    
    def get_dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def _encode(self, texts: List[str], model, tokenizer) -> List[List[float]]:
        """Encode texts using specified model."""
        with torch.no_grad():
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = model(**encoded)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            return embeddings.cpu().numpy().tolist()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self._encode([text], self.query_model, self.query_tokenizer)[0]
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed batch of document texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._encode(batch, self.article_model, self.article_tokenizer)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query (uses query encoder)."""
        return self._encode([query], self.query_model, self.query_tokenizer)[0]
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents (uses article encoder)."""
        return self.embed_batch(documents)
