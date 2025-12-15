"""Environment variable secret backend."""

import os
import logging

from src.core.secrets.base import SecretBackend
from src.core.secrets.exceptions import SecretNotFoundError

logger = logging.getLogger(__name__)


class EnvSecretBackend(SecretBackend):
    """
    Reads secrets from environment variables.
    
    Uses exact key match - no conversion.
    
    Examples:
        ${secret:OLLAMA_HOST} -> env var OLLAMA_HOST
        ${secret:QDRANT_PORT} -> env var QDRANT_PORT
    """

    def __init__(self, prefix: str = ""):
        """
        Args:
            prefix: Optional prefix for env vars (e.g., 'APP_')
        """
        self.prefix = prefix

    def get_secret(self, key: str) -> str:
        """
        Get secret from environment variable.
        
        Args:
            key: Secret key (exact match to env var name)
            
        Returns:
            Secret value
            
        Raises:
            SecretNotFoundError: If env var not set
        """
        env_key = f"{self.prefix}{key}"
        value = os.environ.get(env_key)

        if value is None:
            raise SecretNotFoundError(
                f"Secret '{key}' not found. Set environment variable: {env_key}"
            )

        logger.debug(f"Resolved secret '{key}' from env var '{env_key}'")
        return value

    def health_check(self) -> bool:
        """Environment backend is always available."""
        return True
