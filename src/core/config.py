"""
Configuration loader for RAG system.
Loads YAML config and resolves ${secret:KEY} patterns.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.secrets import SecretResolver

logger = logging.getLogger(__name__)


class Config:
    """
    Singleton config loader.
    
    Usage:
        config = Config.load("config/rag.yaml")
        chunk_size = config.get("chunking.chunk_size")
        embeddings = config.get_section("embeddings")
    """
    
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: str = "config/rag.yaml", env: str = None) -> "Config":
        """
        Load config from YAML file.
        
        Args:
            config_path: Path to main config file
            env: Environment name (loads config/environments/{env}.yaml as override)
            
        Returns:
            Config instance
        """
        instance = cls()
        
        # Load main config
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, "r") as f:
            instance._config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        
        # Load environment override if specified
        if env:
            env_path = Path(f"config/environments/{env}.yaml")
            if env_path.exists():
                with open(env_path, "r") as f:
                    env_config = yaml.safe_load(f)
                instance._config = instance._merge_configs(instance._config, env_config)
                logger.info(f"Applied environment override: {env}")
        
        # Resolve secrets
        backend = instance._config.get("secrets", {}).get("backend", "env")
        resolver = SecretResolver(backend=backend)
        instance._config = resolver.resolve_config(instance._config)
        
        return instance
    
    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge override into base config."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Example:
            config.get("chunking.chunk_size")  # Returns 512
            config.get("chunking.missing", 100)  # Returns 100
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section as dict."""
        return self.get(section, {})
    
    @property
    def raw(self) -> Dict[str, Any]:
        """Get raw config dict."""
        return self._config
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._config = {}
