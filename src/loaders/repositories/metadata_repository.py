"""Metadata repository for loading and caching metadata."""

import json
import threading
from pathlib import Path
from typing import Any

from config.logging import logger
from loaders.constants import CMAP_METADATA_FILENAME
from loaders.constants import FONT_METADATA_FILENAME
from loaders.constants import METADATA_DIR
from loaders.exceptions import MetadataNotFoundError
from loaders.models import CMapMetadata
from loaders.models import FontMetadata
from loaders.utils.path_helpers import get_cache_file_path

# Thread-safe cache storage with locks
_font_cache: dict[str, Any] | None = None
_cmap_cache: dict[str, Any] | None = None
_cache_lock = threading.RLock()


def _load_json_file(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        MetadataNotFoundError: If file doesn't exist or is invalid
    """
    if not path.exists():
        raise MetadataNotFoundError(
            f"Metadata file not found: {path}",
            metadata_file=str(path),
        )

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise MetadataNotFoundError(
            f"Invalid JSON in metadata file: {e}",
            metadata_file=str(path),
        )
    except OSError as e:
        raise MetadataNotFoundError(
            f"Cannot read metadata file: {e}",
            metadata_file=str(path),
        )


def get_font_metadata() -> dict[str, FontMetadata]:
    """Load font metadata with caching.

    Thread-safe. Loads from JSON file only on first call or after reload.

    Returns:
        Dictionary mapping font filenames to FontMetadata
    """
    global _font_cache

    with _cache_lock:
        if _font_cache is not None:
            return _font_cache

        path = get_cache_file_path(FONT_METADATA_FILENAME, METADATA_DIR)

        try:
            raw_data = _load_json_file(path)
            _font_cache = {
                name: FontMetadata.from_dict(name, data)
                for name, data in raw_data.items()
            }
            logger.debug(f"Loaded {_font_cache.__len__()} font metadata entries")
            return _font_cache
        except MetadataNotFoundError:
            logger.warning(f"Font metadata not available at {path}")
            _font_cache = {}
            return _font_cache


def get_cmap_metadata() -> dict[str, CMapMetadata]:
    """Load CMap metadata with caching.

    Thread-safe. Loads from JSON file only on first call or after reload.

    Returns:
        Dictionary mapping CMap filenames to CMapMetadata
    """
    global _cmap_cache

    with _cache_lock:
        if _cmap_cache is not None:
            return _cmap_cache

        path = get_cache_file_path(CMAP_METADATA_FILENAME, METADATA_DIR)

        try:
            raw_data = _load_json_file(path)
            _cmap_cache = {
                name: CMapMetadata.from_dict(name, data)
                for name, data in raw_data.items()
            }
            logger.debug(f"Loaded {_cmap_cache.__len__()} CMap metadata entries")
            return _cmap_cache
        except MetadataNotFoundError:
            logger.warning(f"CMap metadata not available at {path}")
            _cmap_cache = {}
            return _cmap_cache


def clear_metadata_cache() -> None:
    """Clear the metadata cache to force reload on next access.

    Useful after downloading new metadata files.
    """
    global _font_cache, _cmap_cache

    with _cache_lock:
        _font_cache = None
        _cmap_cache = None
        logger.debug("Metadata cache cleared")


def get_font_names() -> set[str]:
    """Get set of all available font names.

    Returns:
        Set of font display names
    """
    metadata = get_font_metadata()
    return {m.font_name for m in metadata.values() if m.font_name}


def get_font_metadata_by_name(name: str) -> FontMetadata | None:
    """Get metadata for a specific font.

    Args:
        name: Font filename (e.g., "SourceHanSansCN-Regular.ttf")

    Returns:
        FontMetadata if found, None otherwise
    """
    metadata = get_font_metadata()
    return metadata.get(name)


def get_cmap_metadata_by_name(name: str) -> CMapMetadata | None:
    """Get metadata for a specific CMap.

    Args:
        name: CMap filename (e.g., "Adobe-Japan1-UCS2.json")

    Returns:
        CMapMetadata if found, None otherwise
    """
    metadata = get_cmap_metadata()
    return metadata.get(name)


# Backward-compatible dynamic dict-like access for EMBEDDING_FONT_METADATA
class _FontMetadataProxy:
    """Proxy object that behaves like a dict but loads from cache on demand."""

    def __getitem__(self, key: str) -> dict[str, Any]:
        meta = get_font_metadata_by_name(key)
        if meta is None:
            raise KeyError(key)
        # Return as dict for backward compatibility
        return {
            "sha3_256": meta.sha3_256,
            "size": meta.size,
            "url": meta.url,
            "font_name": meta.font_name,
            "subset_font_path": meta.subset_font_path,
        }

    def __contains__(self, key: str) -> bool:
        return key in get_font_metadata()

    def __iter__(self):
        return iter(get_font_metadata())

    def keys(self):
        return get_font_metadata().keys()

    def values(self):
        for name, meta in get_font_metadata().items():
            yield {
                "sha3_256": meta.sha3_256,
                "size": meta.size,
                "url": meta.url,
                "font_name": meta.font_name,
                "subset_font_path": meta.subset_font_path,
            }

    def items(self):
        for name, meta in get_font_metadata().items():
            yield (
                name,
                {
                    "sha3_256": meta.sha3_256,
                    "size": meta.size,
                    "url": meta.url,
                    "font_name": meta.font_name,
                    "subset_font_path": meta.subset_font_path,
                },
            )

    def __len__(self) -> int:
        return len(get_font_metadata())


class _CMapMetadataProxy:
    """Proxy object that behaves like a dict but loads from cache on demand."""

    def __getitem__(self, key: str) -> dict[str, Any]:
        meta = get_cmap_metadata_by_name(key)
        if meta is None:
            raise KeyError(key)
        return {
            "sha3_256": meta.sha3_256,
            "size": meta.size,
            "url": meta.url,
            "cmap_name": meta.cmap_name,
        }

    def __contains__(self, key: str) -> bool:
        return key in get_cmap_metadata()

    def __iter__(self):
        return iter(get_cmap_metadata())

    def keys(self):
        return get_cmap_metadata().keys()

    def values(self):
        for name, meta in get_cmap_metadata().items():
            yield {
                "sha3_256": meta.sha3_256,
                "size": meta.size,
                "url": meta.url,
                "cmap_name": meta.cmap_name,
            }

    def items(self):
        for name, meta in get_cmap_metadata().items():
            yield (
                name,
                {
                    "sha3_256": meta.sha3_256,
                    "size": meta.size,
                    "url": meta.url,
                    "cmap_name": meta.cmap_name,
                },
            )

    def __len__(self) -> int:
        return len(get_cmap_metadata())


# Public proxy instances for backward compatibility
EMBEDDING_FONT_METADATA = _FontMetadataProxy()
CMAP_METADATA = _CMapMetadataProxy()
