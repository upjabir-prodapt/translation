"""Download service for asset downloads with retry logic."""

import asyncio
import logging
from pathlib import Path

from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from src.config.constants import settings
from src.worker.loaders.exceptions import AssetDownloadError
from src.worker.loaders.exceptions import AssetIntegrityError
from src.worker.loaders.repositories.cache_repository import verify_or_delete
from src.worker.loaders.services.integrity_service import verify_and_raise
from src.worker.loaders.utils.path_helpers import get_cache_file_path
from src.repository.translation_storage_repository import TranslationStorageRepository

logger = logging.getLogger(__name__)


def _get_storage_repo(
    storage_repo: TranslationStorageRepository | None = None,
) -> TranslationStorageRepository:
    """Get or create storage repository instance."""
    return storage_repo or TranslationStorageRepository()


@retry(
    stop=stop_after_attempt(settings.DOWNLOAD_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=settings.DOWNLOAD_RETRY_MULTIPLIER,
        min=settings.DOWNLOAD_RETRY_MIN_SECONDS,
        max=settings.DOWNLOAD_RETRY_MAX_SECONDS,
    ),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def download_with_retry(
    blob_path: str,
    local_path: Path,
    storage_repo: TranslationStorageRepository | None = None,
) -> None:
    """Download file from GCS with automatic retry logic.

    Args:
        blob_path: Blob path in GCS (relative to assets prefix)
        local_path: Local destination path
        storage_repo: Optional repository instance

    Raises:
        AssetDownloadError: If download fails after all retries
    """
    repo = _get_storage_repo(storage_repo)

    try:
        repo.download_asset_sync(blob_path, local_path)
        logger.debug(f"Downloaded {blob_path}")
    except Exception as e:
        logger.error(f"Failed to download {blob_path}: {e}")
        raise AssetDownloadError(
            f"Download failed: {e}",
            blob_path=blob_path,
            attempts=settings.DOWNLOAD_MAX_ATTEMPTS,
        ) from e


async def download_async(
    blob_path: str,
    local_path: Path,
    storage_repo: TranslationStorageRepository | None = None,
) -> None:
    """Download file asynchronously with retry logic.

    Args:
        blob_path: Blob path in GCS (relative to assets prefix)
        local_path: Local destination path
        storage_repo: Optional repository instance

    Raises:
        AssetDownloadError: If download fails after all retries
    """
    # Run sync download in thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        download_with_retry,
        blob_path,
        local_path,
        storage_repo,
    )


def download_and_verify(
    blob_path: str,
    local_path: Path,
    expected_hash: str,
    asset_name: str | None = None,
    storage_repo: TranslationStorageRepository | None = None,
) -> Path:
    """Download asset and verify its integrity.

    Args:
        blob_path: GCS blob path (relative to assets prefix)
        local_path: Local destination path
        expected_hash: Expected SHA3-256 hash
        asset_name: Human-readable name for logging
        storage_repo: Optional repository instance

    Returns:
        Path to verified file

    Raises:
        AssetDownloadError: If download fails
        AssetIntegrityError: If downloaded file is corrupted
    """
    name = asset_name or local_path.name
    logger.info(f"Downloading {name}...")

    download_with_retry(blob_path, local_path, storage_repo)

    try:
        verify_and_raise(local_path, expected_hash, name)
    except AssetIntegrityError:
        # Delete corrupted file and re-raise
        verify_or_delete(local_path, expected_hash)
        raise

    logger.info(f"Verified {name}")
    return local_path


async def download_and_verify_async(
    blob_path: str,
    local_path: Path,
    expected_hash: str,
    asset_name: str | None = None,
    storage_repo: TranslationStorageRepository | None = None,
) -> Path:
    """Download asset async and verify its integrity.

    Args:
        blob_path: GCS blob path (relative to assets prefix)
        local_path: Local destination path
        expected_hash: Expected SHA3-256 hash
        asset_name: Human-readable name for logging
        storage_repo: Optional repository instance

    Returns:
        Path to verified file

    Raises:
        AssetDownloadError: If download fails
        AssetIntegrityError: If downloaded file is corrupted
    """
    name = asset_name or local_path.name
    logger.info(f"Downloading {name} (async)...")

    await download_async(blob_path, local_path, storage_repo)

    try:
        verify_and_raise(local_path, expected_hash, name)
    except AssetIntegrityError:
        verify_or_delete(local_path, expected_hash)
        raise

    logger.info(f"Verified {name}")
    return local_path


def get_or_download_model(
    filename: str,
    expected_hash: str,
    model_name: str,
    subdir: str = "models",
) -> Path:
    """Get model path, downloading if necessary.

    Args:
        filename: Model filename (from settings)
        expected_hash: Expected SHA3-256 hash
        model_name: Human-readable model name
        subdir: Subdirectory for the model

    Returns:
        Path to verified model file

    Raises:
        AssetDownloadError: If download fails
        AssetIntegrityError: If file is corrupted
    """
    model_path = get_cache_file_path(filename, subdir)

    # Check if already cached and valid
    if verify_or_delete(model_path, expected_hash):
        logger.debug(f"Model {model_name} already cached")
        return model_path

    return download_and_verify(
        f"{subdir}/{filename}",
        model_path,
        expected_hash,
        asset_name=model_name,
    )


async def get_or_download_model_async(
    filename: str,
    expected_hash: str,
    model_name: str,
    subdir: str = "models",
) -> Path:
    """Get model path async, downloading if necessary.

    Args:
        filename: Model filename (from settings)
        expected_hash: Expected SHA3-256 hash
        model_name: Human-readable model name
        subdir: Subdirectory for the model

    Returns:
        Path to verified model file
    """
    model_path = get_cache_file_path(filename, subdir)

    if verify_or_delete(model_path, expected_hash):
        logger.debug(f"Model {model_name} already cached")
        return model_path

    return await download_and_verify_async(
        f"{subdir}/{filename}",
        model_path,
        expected_hash,
        asset_name=model_name,
    )
