"""Integrity service for file verification."""

import logging
from pathlib import Path

from src.worker.loaders.exceptions import AssetIntegrityError
from src.worker.loaders.repositories.cache_repository import verify_or_delete

logger = logging.getLogger(__name__)


def verify_and_raise(
    path: Path,
    expected_hash: str,
    asset_name: str | None = None,
) -> None:
    """Verify file integrity, raise exception if invalid.

    Args:
        path: Path to file
        expected_hash: Expected SHA3-256 hash
        asset_name: Name of asset for error messages

    Raises:
        AssetIntegrityError: If file is missing or corrupted
    """
    if not path.exists():
        raise AssetIntegrityError(
            f"Asset file not found: {path}",
            asset_name=asset_name or path.name,
        )

    is_valid = verify_or_delete(path, expected_hash)

    if not is_valid:
        raise AssetIntegrityError(
            f"Asset integrity check failed: {path}",
            asset_name=asset_name or path.name,
            expected_hash=expected_hash,
        )


def is_valid(path: Path, expected_hash: str) -> bool:
    """Quick check if file exists and is valid.

    Args:
        path: Path to file
        expected_hash: Expected SHA3-256 hash

    Returns:
        True if valid, False otherwise
    """
    from src.worker.loaders.repositories.cache_repository import verify_file_integrity

    return verify_file_integrity(path, expected_hash)
