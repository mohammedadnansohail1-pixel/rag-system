"""Tests for environment variable secret backend."""

import os
import pytest

from src.core.secrets.env_backend import EnvSecretBackend
from src.core.secrets.exceptions import SecretNotFoundError


class TestEnvSecretBackend:
    """Tests for EnvSecretBackend."""

    def test_get_secret_success(self, monkeypatch):
        """Should return value when env var exists."""
        # Arrange
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        backend = EnvSecretBackend()

        # Act
        result = backend.get_secret("DB_PASSWORD")

        # Assert
        assert result == "secret123"

    def test_get_secret_with_prefix(self, monkeypatch):
        """Should use prefix when provided."""
        # Arrange
        monkeypatch.setenv("APP_DB_PASSWORD", "secret456")
        backend = EnvSecretBackend(prefix="APP_")

        # Act
        result = backend.get_secret("DB_PASSWORD")

        # Assert
        assert result == "secret456"

    def test_get_secret_not_found(self):
        """Should raise SecretNotFoundError when env var missing."""
        # Arrange
        backend = EnvSecretBackend()

        # Act & Assert
        with pytest.raises(SecretNotFoundError) as exc_info:
            backend.get_secret("NONEXISTENT_KEY")

        assert "NONEXISTENT_KEY" in str(exc_info.value)

    def test_health_check(self):
        """Health check should always return True."""
        # Arrange
        backend = EnvSecretBackend()

        # Act
        result = backend.health_check()

        # Assert
        assert result is True
