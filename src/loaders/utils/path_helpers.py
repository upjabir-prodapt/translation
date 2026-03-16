"""Path helpers for cache file resolution."""

import os
from pathlib import Path

from config.constants import _PROJECT_ROOT


def get_cache_root() -> Path:
    """Get the root cache directory.

    Respects BABELDOC_CACHE_DIR environment variable,
    otherwise defaults to project_root/assets.
    """
    cache_dir = os.environ.get("BABELDOC_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    return _PROJECT_ROOT / "assests"


def get_cache_file_path(filename: str, subdir: str = "") -> Path:
    """Get cache file path with optional subdirectory.

    Creates parent directories if they don't exist.

    Args:
        filename: Name of the file
        subdir: Optional subdirectory (e.g., "fonts", "models")

    Returns:
        Path to the cache file (parent directories created)
    """
    cache_root = get_cache_root()

    if subdir:
        path = cache_root / subdir / filename
    else:
        path = cache_root / filename

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_subdir_path(subdir: str) -> Path:
    """Get a subdirectory path within the cache.

    Args:
        subdir: Subdirectory name (e.g., "fonts", "cmap", "models")

    Returns:
        Path to the subdirectory (created if it doesn't exist)
    """
    path = get_cache_root() / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path
