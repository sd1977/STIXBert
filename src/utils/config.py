"""Load and access config.yaml settings."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_config: dict | None = None

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and cache it.

    Args:
        path: Path to config file. Defaults to configs/config.yaml.

    Returns:
        Parsed config dict.
    """
    global _config

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        _config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return _config


def get_config() -> dict:
    """Return the cached config, loading if needed."""
    if _config is None:
        return load_config()
    return _config


def cfg(*keys: str, default: Any = None) -> Any:
    """Access a nested config value using dot-path keys.

    Examples:
        cfg('model', 'encoder', 'hidden_dim')   -> 128
        cfg('training', 'pretrain', 'lr')        -> 0.001
        cfg('grid_search', 'enabled')            -> False

    Args:
        *keys: Sequence of nested keys.
        default: Fallback if key path doesn't exist.

    Returns:
        Config value or default.
    """
    config = get_config()
    node = config
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node
