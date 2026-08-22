"""Path helpers for cache file resolution."""

from pathlib import Path

from src.config.constants import settings


def get_cache_root() -> Path:
    """Get the root cache directory.

    Returns the canonical asset cache root from settings.
    """
    return settings.assets_root_path


def _assert_under_cache_root(path: Path) -> None:
    """Ensure path remains inside configured cache root."""
    root = get_cache_root().resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Resolved path '{target}' escapes cache root '{root}'") from e


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

    _assert_under_cache_root(path)
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
    _assert_under_cache_root(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
