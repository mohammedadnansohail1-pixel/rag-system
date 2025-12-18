"""Environment variable secret backend."""
import os
import logging
from pathlib import Path
from src.core.secrets.base import SecretBackend
from src.core.secrets.exceptions import SecretNotFoundError

logger = logging.getLogger(__name__)

# Auto-load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parents[3] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


class EnvSecretBackend(SecretBackend):
    """Reads secrets from environment variables with sensible defaults."""
    
    DEFAULTS = {
        "OLLAMA_HOST": "http://localhost:11434",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "OPENAI_API_KEY": "",  # Empty = not configured
        "LLM_MODEL": "llama3.2",
    }
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
    
    def get_secret(self, key: str) -> str:
        env_key = f"{self.prefix}{key}" if self.prefix else key
        value = os.environ.get(env_key)
        
        if value is None:
            if key in self.DEFAULTS:
                return self.DEFAULTS[key]
            raise SecretNotFoundError(
                f"Secret '{key}' not found. Set environment variable: {env_key}"
            )
        return value
    
    def set_secret(self, key: str, value: str) -> None:
        env_key = f"{self.prefix}{key}" if self.prefix else key
        os.environ[env_key] = value
    
    def health_check(self) -> bool:
        return True
