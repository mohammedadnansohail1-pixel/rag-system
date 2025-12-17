"""Configuration loader for embedding analyzer."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config directory
DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(
    config_name: str = "default",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load analyzer configuration from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)
        config_path: Full path to config file (overrides config_name)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        path = DEFAULT_CONFIG_DIR / f"{config_name}.yaml"

    if not path.exists():
        available = [f.stem for f in DEFAULT_CONFIG_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Config not found: {path}. Available: {available}"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded embedding analyzer config: {path.name}")
    return config


def list_available_configs() -> list:
    """
    List available configuration files.

    Returns:
        List of config names (without .yaml extension)
    """
    return [f.stem for f in DEFAULT_CONFIG_DIR.glob("*.yaml")]
