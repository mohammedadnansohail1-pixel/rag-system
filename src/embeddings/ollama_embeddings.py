"""Ollama embedding provider."""

import logging
from src.utils.text_utils import is_garbage_text
import time
import requests
from typing import List, Optional

from src.embeddings.base import BaseEmbeddings
from src.embeddings.factory import register_embeddings

logger = logging.getLogger(__name__)


@register_embeddings("ollama")
class OllamaEmbeddings(BaseEmbeddings):
    """
    Generates embeddings using Ollama.

    Usage:
        embeddings = OllamaEmbeddings(
            host="http://localhost:11434",
            model="nomic-embed-text"
        )
        vector = embeddings.embed_text("Hello world")
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        max_retries: int = 1,
    ):
        """
        Args:
            host: Ollama server URL
            model: Embedding model name
            dimensions: Expected embedding dimensions
            max_retries: Number of retries on failure
        """
        self.host = host.rstrip("/")
        self.model = model
        self._dimensions = dimensions
        self.max_retries = max_retries

        logger.info(f"Initialized OllamaEmbeddings: model={model}, host={host}")

    def _truncate_text(self, text: str, max_chars: int = 4000) -> str:
        """Truncate text to avoid token limits."""
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def _clean_text(self, text: str) -> str:
        """Clean text of problematic characters."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Fix common UTF-8 encoding issues
        replacements = [
            ('Â¶', ''),      # Garbled pilcrow
            ('Â', ''),       # Garbled A-circumflex
            ('â€"', '-'),    # Garbled em-dash
            ('â€™', "'"),    # Garbled apostrophe
            ('â€œ', '"'),    # Garbled left quote
            ('â€', '"'),     # Garbled right quote
            ('\u200b', ''), # Zero-width space
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        
        # Remove any remaining non-ASCII that might cause issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        text = self._clean_text(text)
        text = self._truncate_text(text)
        url = f"{self.host}/api/embeddings"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json={"model": self.model, "prompt": text},
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # Longer wait
                else:
                    raise

    def embed_batch(
        self,
        texts: List[str],
        skip_errors: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            skip_errors: If True, use zero vector for failed embeddings

        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        failed = 0
        garbage_filtered = 0
        for i, text in enumerate(texts):


            # Skip garbage text (base64, binary data from PDFs/SEC filings)
            if is_garbage_text(text):
                embeddings.append([0.0] * self._dimensions)
                garbage_filtered += 1
                continue

            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                failed += 1
                if skip_errors:
                    # Use zero vector as placeholder
                    embeddings.append([0.0] * self._dimensions)
                    logger.warning(f"Skipped chunk {i}: {e}")
                else:
                    raise

        if garbage_filtered > 0:
            print(f"  Filtered: {garbage_filtered} garbage chunks (base64/binary)")
        if failed > 0:
            print(f"  Warning: {failed} chunks failed embedding")

        logger.debug(f"Generated {len(embeddings)} embeddings ({failed} failed, {garbage_filtered} filtered)")
        return embeddings

    def get_dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.model
