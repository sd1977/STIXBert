"""Centralized path management for STIXBert.

All data, checkpoints, and results are stored on Google Drive when running
on Colab, or locally otherwise. This module provides a single place to
configure and resolve paths.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Google Drive base path (set after mounting)
_DRIVE_BASE: Path | None = None

# Default Drive subdirectory
DRIVE_PROJECT_DIR = "stixbert"


def mount_drive() -> Path:
    """Mount Google Drive and return the project root on Drive.

    Creates the full directory tree under MyDrive/stixbert/ if it doesn't exist.

    Returns:
        Path to the project root on Drive.
    """
    global _DRIVE_BASE

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        _DRIVE_BASE = Path("/content/drive/MyDrive") / DRIVE_PROJECT_DIR
    except ImportError:
        logger.info("Not running on Colab — using local paths")
        _DRIVE_BASE = Path(".")

    # Create directory tree
    for subdir in DRIVE_DIRS.values():
        (_DRIVE_BASE / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Project root: {_DRIVE_BASE}")
    _log_tree()
    return _DRIVE_BASE


# All subdirectories under the project root on Drive
DRIVE_DIRS = {
    # Raw data from external sources (immutable after download)
    "raw_mitre":       "data/raw/mitre_attack",
    "raw_threatfox":   "data/raw/threatfox",
    "raw_digitalside": "data/raw/digitalside",
    "raw_taxii":       "data/raw/taxii",
    "raw_misp":        "data/raw/misp",

    # Processed graph data (PyG HeteroData, features)
    "processed":       "data/processed",

    # Train / test / eval splits
    "train":           "data/train",
    "test":            "data/test",
    "eval":            "data/eval",

    # Model checkpoints
    "checkpoints":     "checkpoints",

    # Results (plots, metrics, embeddings)
    "results":         "results",
}


def get_path(key: str) -> Path:
    """Get the absolute path for a named directory.

    Args:
        key: One of the keys in DRIVE_DIRS (e.g., 'raw_mitre', 'train', 'checkpoints').

    Returns:
        Absolute Path to the directory.
    """
    base = _DRIVE_BASE or Path(".")
    if key not in DRIVE_DIRS:
        raise KeyError(f"Unknown path key '{key}'. Valid keys: {list(DRIVE_DIRS.keys())}")
    path = base / DRIVE_DIRS[key]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Return the project root (Drive or local)."""
    return _DRIVE_BASE or Path(".")


def _log_tree():
    """Log the directory tree."""
    base = _DRIVE_BASE or Path(".")
    logger.info("Directory tree:")
    logger.info(f"  {base}/")
    for name, subdir in sorted(DRIVE_DIRS.items()):
        p = base / subdir
        status = "✓" if p.exists() else "○"
        logger.info(f"    {status} {subdir}/")
