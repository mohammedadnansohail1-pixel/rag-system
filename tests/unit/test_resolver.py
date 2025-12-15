"""Tests for secret resolver."""

import pytest

from src.core.secrets.resolver import SecretResolver
from src.core.secrets.exceptions import SecretNotFoundError, SecretBackendError


class TestSecretResolver:
    """Tests for SecretResolver."""

    def test_resolve_single_secret(self, monkeypatch):
        """Should resolve ${secret:KEY} pattern."""
        # Arrange
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        resolver = SecretResolver(backend="env")

        # Act
        result = resolver.resolve_value("${secret:DB_PASSWORD}")

        # Assert
        assert result == "secret123"

    def test_resolve_embedded_secret(self, monkeypatch):
        """Should resolve secret embedded in string."""
        # Arrange
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        resolver = SecretResolver(backend="env")

        # Act
        result = resolver.resolve_value("postgresql://${secret:HOST}:${secret:PORT}")

        # Assert
        assert result == "postgresql://localhost:5432"

    def test_resolve_no_secret(self):
        """Should return string unchanged if no secret pattern."""
        # Arrange
        resolver = SecretResolver(backend="env")

        # Act
        result = resolver.resolve_value("plain string")

        # Assert
        assert result == "plain string"

    def test_resolve_config_dict(self, monkeypatch):
        """Should recursively resolve secrets in dict."""
        # Arrange
        monkeypatch.setenv("QDRANT_HOST", "localhost")
        monkeypatch.setenv("QDRANT_PORT", "6333")
        resolver = SecretResolver(backend="env")
        
        config = {
            "vectorstore": {
                "host": "${secret:QDRANT_HOST}",
                "port": "${secret:QDRANT_PORT}",
                "name": "documents"
            }
        }

        # Act
        result = resolver.resolve_config(config)

        # Assert
        assert result["vectorstore"]["host"] == "localhost"
        assert result["vectorstore"]["port"] == "6333"
        assert result["vectorstore"]["name"] == "documents"

    def test_resolve_config_list(self, monkeypatch):
        """Should resolve secrets in lists."""
        # Arrange
        monkeypatch.setenv("KEY1", "value1")
        resolver = SecretResolver(backend="env")
        
        config = ["${secret:KEY1}", "plain"]

        # Act
        result = resolver.resolve_config(config)

        # Assert
        assert result == ["value1", "plain"]

    def test_unknown_backend_raises(self):
        """Should raise error for unknown backend."""
        # Act & Assert
        with pytest.raises(SecretBackendError) as exc_info:
            SecretResolver(backend="unknown")

        assert "Unknown backend" in str(exc_info.value)

    def test_health_check(self):
        """Health check should return True for env backend."""
        # Arrange
        resolver = SecretResolver(backend="env")

        # Act
        result = resolver.health_check()

        # Assert
        assert result is True
