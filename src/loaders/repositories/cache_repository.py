"""Cache repository for local file operations."""

import hashlib
import logging
from pathlib import Path

from loaders.utils.path_helpers import get_cache_file_path

logger = logging.getLogger(__name__)


def get_file_hash(path: Path, algorithm: str = "sha3_256") -> str:
    """Calculate file hash using specified algorithm.

    Args:
        path: Path to the file
        algorithm: Hash algorithm to use (sha3_256, sha256, md5)

    Returns:
        Hex digest of the file hash

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    hash_obj = hashlib.new(algorithm)

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):  # Read 1MB chunks
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def verify_file_integrity(
    path: Path,
    expected_hash: str,
    algorithm: str = "sha3_256",
) -> bool:
    """Verify file integrity by comparing hash.

    Args:
        path: Path to the file to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if file exists and hash matches, False otherwise
    """
    if not path.exists():
        logger.debug(f"File does not exist: {path}")
        return False

    try:
        actual_hash = get_file_hash(path, algorithm)
        is_valid = actual_hash == expected_hash

        if not is_valid:
            logger.warning(
                f"Hash mismatch for {path.name}: "
                f"expected {expected_hash[:8]}..., got {actual_hash[:8]}..."
            )

        return is_valid
    except OSError as e:
        logger.error(f"Error reading file {path}: {e}")
        return False


def verify_or_delete(
    path: Path,
    expected_hash: str,
    algorithm: str = "sha3_256",
) -> bool:
    """Verify file integrity, delete if corrupted.

    Args:
        path: Path to the file to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if file is valid, False if deleted or doesn't exist
    """
    if not path.exists():
        return False

    is_valid = verify_file_integrity(path, expected_hash, algorithm)
    if not is_valid:
        logger.warning(f"Deleting corrupted file: {path}")
        try:
            path.unlink()
        except OSError as e:
            logger.error(f"Failed to delete corrupted file {path}: {e}")
        return False

    return True


def ensure_cached(
    filename: str,
    expected_hash: str,
    subdir: str = "",
    algorithm: str = "sha3_256",
) -> Path | None:
    """Get path to cached file if it exists and is valid.

    Args:
        filename: Name of the cached file
        expected_hash: Expected hash for verification
        subdir: Optional subdirectory
        algorithm: Hash algorithm used

    Returns:
        Path to cached file if valid, None otherwise
    """
    path = get_cache_file_path(filename, subdir)

    if verify_or_delete(path, expected_hash, algorithm):
        return path

    return None


def get_file_size(path: Path) -> int:
    """Get file size in bytes.

    Args:
        path: Path to file

    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return path.stat().st_size
    except (FileNotFoundError, OSError):
        return 0
