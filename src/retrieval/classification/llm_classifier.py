"""LLM-based query complexity classification for ambiguous cases."""
import logging
from typing import Optional
import requests

from .base import QueryComplexity

logger = logging.getLogger(__name__)


class LLMClassifier:
    """
    Classify query complexity using LLM for ambiguous cases.
    Adds ~500-2000ms latency depending on model.
    """
    
    PROMPT_TEMPLATE = """Classify this search query's complexity for a RAG system.

Query: "{query}"

Categories:
- SIMPLE: Single fact lookup, definitions, "what is X" questions
- MEDIUM: Entity + attribute queries, "how does X work", most business questions  
- COMPLEX: Comparisons, multi-hop reasoning, conditional questions, analysis

Respond with exactly one word: SIMPLE, MEDIUM, or COMPLEX"""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 5.0,
    ):
        self.host = host
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        if self._available is not None:
            return self._available
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2.0)
            self._available = response.status_code == 200
        except Exception:
            self._available = False
        
        return self._available
    
    def classify(self, query: str) -> Optional[QueryComplexity]:
        """Classify query using LLM."""
        prompt = self.PROMPT_TEMPLATE.format(query=query)
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            result = response.json().get("response", "").strip().upper()
            return self._parse_response(result)
            
        except requests.exceptions.Timeout:
            logger.warning("LLM classification timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("LLM not available")
            self._available = False
            return None
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None
    
    def _parse_response(self, response: str) -> Optional[QueryComplexity]:
        """Parse LLM response to QueryComplexity."""
        response = response.split()[0] if response else ""
        
        if "SIMPLE" in response:
            return QueryComplexity.SIMPLE
        elif "COMPLEX" in response:
            return QueryComplexity.COMPLEX
        elif "MEDIUM" in response:
            return QueryComplexity.MEDIUM
        
        logger.warning(f"LLM returned unexpected response: {response}")
        return None
