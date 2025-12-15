"""Secret resolver - resolves ${secret:KEY} patterns in config."""

import re
import logging
from typing import Any, Dict

from src.core.secrets.base import SecretBackend
from src.core.secrets.env_backend import EnvSecretBackend
from src.core.secrets.exceptions import SecretBackendError

logger = logging.getLogger(__name__)

# Pattern to match ${secret:KEY_NAME}
SECRET_PATTERN = re.compile(r'\$\{secret:([^}]+)\}')


class SecretResolver:
    """
    Resolves ${secret:KEY} patterns in config values.
    
    Usage:
        resolver = SecretResolver(backend='env')
        resolved = resolver.resolve_value('${secret:DB_PASSWORD}')
        
        # Or resolve entire config dict
        config = resolver.resolve_config(raw_config)
    """

    def __init__(self, backend: str = "env", **backend_config):
        """
        Args:
            backend: Backend type ('env', 'file', 'vault')
            **backend_config: Backend-specific configuration
        """
        self.backend = self._create_backend(backend, backend_config)
        logger.info(f"Initialized SecretResolver with backend: {backend}")

    def _create_backend(self, backend: str, config: dict) -> SecretBackend:
        """Factory method to create the appropriate backend."""
        
        if backend == "env":
            return EnvSecretBackend(prefix=config.get("prefix", ""))
        
        elif backend == "file":
            raise NotImplementedError("File backend not yet implemented")
        
        elif backend == "vault":
            raise NotImplementedError("Vault backend not yet implemented")
        
        else:
            raise SecretBackendError(f"Unknown backend: {backend}")

    def resolve_value(self, value: str) -> str:
        """
        Resolve a single value, replacing ${secret:KEY} patterns.
        
        Args:
            value: String that may contain secret references
            
        Returns:
            String with secrets resolved
        """
        if not isinstance(value, str):
            return value

        def replace_secret(match):
            key = match.group(1)
            return self.backend.get_secret(key)

        return SECRET_PATTERN.sub(replace_secret, value)

    def resolve_config(self, config: Any) -> Any:
        """
        Recursively resolve all secrets in a config structure.
        
        Args:
            config: Dict, list, or value that may contain secret references
            
        Returns:
            Config with all secrets resolved
        """
        if isinstance(config, dict):
            return {k: self.resolve_config(v) for k, v in config.items()}
        
        elif isinstance(config, list):
            return [self.resolve_config(item) for item in config]
        
        elif isinstance(config, str):
            return self.resolve_value(config)
        
        else:
            return config

    def health_check(self) -> bool:
        """Check if the secret backend is healthy."""
        return self.backend.health_check()
