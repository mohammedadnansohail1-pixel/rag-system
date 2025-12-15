"""Tests for config loader."""

import pytest
from pathlib import Path

from src.core.config import Config


class TestConfig:
    """Tests for Config loader."""

    def setup_method(self):
        """Reset singleton before each test."""
        Config.reset()

    def test_load_config(self, tmp_path, monkeypatch):
        """Should load YAML config file."""
        # Arrange
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
project:
  name: test-project
  version: "1.0"
chunking:
  chunk_size: 512
  chunk_overlap: 50
""")

        # Act
        config = Config.load(str(config_file))

        # Assert
        assert config.get("project.name") == "test-project"
        assert config.get("chunking.chunk_size") == 512

    def test_get_with_dot_notation(self, tmp_path):
        """Should access nested values with dot notation."""
        # Arrange
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
level1:
  level2:
    level3: "deep_value"
""")
        config = Config.load(str(config_file))

        # Act
        result = config.get("level1.level2.level3")

        # Assert
        assert result == "deep_value"

    def test_get_default_value(self, tmp_path):
        """Should return default when key not found."""
        # Arrange
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")
        config = Config.load(str(config_file))

        # Act
        result = config.get("nonexistent.key", "default")

        # Assert
        assert result == "default"

    def test_get_section(self, tmp_path):
        """Should return entire section as dict."""
        # Arrange
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
chunking:
  strategy: recursive
  chunk_size: 512
""")
        config = Config.load(str(config_file))

        # Act
        result = config.get_section("chunking")

        # Assert
        assert result == {"strategy": "recursive", "chunk_size": 512}

    def test_secret_resolution(self, tmp_path, monkeypatch):
        """Should resolve ${secret:KEY} patterns."""
        # Arrange
        monkeypatch.setenv("DB_HOST", "localhost")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
secrets:
  backend: env
database:
  host: ${secret:DB_HOST}
""")

        # Act
        config = Config.load(str(config_file))

        # Assert
        assert config.get("database.host") == "localhost"

    def test_environment_override(self, tmp_path, monkeypatch):
        """Should merge environment config over base."""
        # Arrange
        base_config = tmp_path / "config.yaml"
        base_config.write_text("""
chunking:
  chunk_size: 512
  chunk_overlap: 50
""")
        
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        dev_config = env_dir / "dev.yaml"
        dev_config.write_text("""
chunking:
  chunk_size: 256
""")
        
        # Change to tmp_path so relative env path works
        monkeypatch.chdir(tmp_path)
        
        # Create config directory structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "rag.yaml").write_text(base_config.read_text())
        (config_dir / "environments").mkdir()
        (config_dir / "environments" / "dev.yaml").write_text(dev_config.read_text())

        # Act
        config = Config.load("config/rag.yaml", env="dev")

        # Assert
        assert config.get("chunking.chunk_size") == 256  # Overridden
        assert config.get("chunking.chunk_overlap") == 50  # Preserved

    def test_singleton_pattern(self, tmp_path):
        """Should return same instance on multiple loads."""
        # Arrange
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")

        # Act
        config1 = Config.load(str(config_file))
        config2 = Config.load(str(config_file))

        # Assert
        assert config1 is config2

    def test_file_not_found(self):
        """Should raise error when config file missing."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            Config.load("/nonexistent/path/config.yaml")
