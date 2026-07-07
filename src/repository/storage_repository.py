"""Generic GCS Storage Repository - Handles all Google Cloud Storage operations."""

import asyncio
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage

from config.constants import settings
from config.logging_config import logger
from repository.repository_exception import StorageError


class FileType(Enum):
    """Supported file types for storage operations."""

    PDF = "application/pdf"
    CSV = "text/csv"
    JSON = "application/json"
    TEXT = "text/plain"
    IMAGE = "image/png"
    JPEG = "image/jpeg"
    ZIP = "application/zip"
    BINARY = "application/octet-stream"


@dataclass
class StoragePath:
    """Configuration for constructing GCS paths."""

    prefix: str
    job_id: str | None = None
    folder: str | None = None
    filename: str | None = None

    def build(self) -> str:
        """Build the complete blob path."""
        parts = [self.prefix]
        if self.job_id:
            parts.append(self.job_id)
        if self.folder:
            parts.append(self.folder)
        if self.filename:
            parts.append(self.filename)
        return "/".join(parts)


class StorageRepository:
    """Generic repository for Google Cloud Storage operations."""

    def __init__(
        self, client: storage.Client | None = None, bucket_name: str | None = None
    ):
        """Initialize GCS repository.

        Args:
            client: Optional GCS client (creates new one if not provided)
            bucket_name: Optional bucket name (defaults to settings.GCS_BUCKET_NAME)
        """
        self.client = client or storage.Client()
        self.bucket_name = bucket_name or settings.GCS_BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)

    # ========================================================================
    # Generic File Operations
    # ========================================================================

    async def upload_file(
        self,
        source: Path | bytes,
        blob_path: str,
        file_type: FileType | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Generic file upload to GCS.

        Args:
            source: Local file path or bytes content
            blob_path: Destination blob path in GCS
            file_type: File type enum for content type (optional)
            metadata: Optional custom metadata

        Returns:
            GCS URI (gs://bucket/path)

        Raises:
            StorageError: If upload fails
        """
        try:
            blob = self.bucket.blob(blob_path)

            if metadata:
                blob.metadata = metadata

            if isinstance(source, bytes):
                if file_type:
                    await asyncio.to_thread(
                        blob.upload_from_string, source, content_type=file_type.value
                    )
                else:
                    await asyncio.to_thread(blob.upload_from_string, source)
            elif isinstance(source, Path):
                if not source.exists():
                    raise FileNotFoundError(f"Source file not found: {source}")
                if file_type:
                    await asyncio.to_thread(
                        blob.upload_from_filename,
                        str(source),
                        content_type=file_type.value,
                    )
                else:
                    await asyncio.to_thread(blob.upload_from_filename, str(source))
            else:
                raise ValueError("Source must be Path or bytes")

            gcs_uri = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Uploaded file to {gcs_uri}")
            return gcs_uri

        except GoogleAPIError as e:
            logger.error(f"Failed to upload file: {e}")
            raise StorageError(
                f"Failed to upload file: {e}", operation="upload", path=blob_path
            ) from e

    async def download_file(
        self, blob_path: str, local_path: Path, create_dirs: bool = True
    ) -> Path:
        """
        Generic file download from GCS.

        Args:
            blob_path: Source blob path in GCS
            local_path: Local destination path
            create_dirs: Whether to create parent directories

        Returns:
            Path to downloaded file

        Raises:
            StorageError: If download fails
        """
        try:
            blob = self.bucket.blob(blob_path)

            if create_dirs:
                local_path.parent.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(blob.download_to_filename, str(local_path))

            logger.info(
                f"Downloaded file from gs://{self.bucket_name}/{blob_path} to {local_path}"
            )
            return local_path

        except GoogleAPIError as e:
            logger.error(f"Failed to download file: {e}")
            raise StorageError(
                f"Failed to download file: {e}", operation="download", path=blob_path
            ) from e

    async def list_files(
        self,
        prefix: str | None = None,
        delimiter: str | None = None,
        max_results: int | None = None,
    ) -> list[storage.Blob]:
        """
        Generic file listing in GCS.

        Args:
            prefix: Filter by prefix path
            delimiter: Delimiter for directory-like listing
            max_results: Maximum number of results

        Returns:
            List of blob objects

        Raises:
            StorageError: If listing fails
        """
        try:
            blobs = await asyncio.to_thread(
                lambda: list(
                    self.bucket.list_blobs(
                        prefix=prefix, delimiter=delimiter, max_results=max_results
                    )
                )
            )
            logger.debug(f"Listed {len(blobs)} files with prefix '{prefix}'")
            return blobs

        except GoogleAPIError as e:
            logger.error(f"Failed to list files: {e}")
            raise StorageError(
                f"Failed to list files: {e}", operation="list", path=prefix or ""
            ) from e

    async def delete_file(self, blob_path: str) -> None:
        """
        Delete a single file from GCS.

        Args:
            blob_path: Blob path to delete

        Raises:
            StorageError: If deletion fails
        """
        try:
            blob = self.bucket.blob(blob_path)
            await asyncio.to_thread(blob.delete)
            logger.info(f"Deleted file: gs://{self.bucket_name}/{blob_path}")

        except GoogleAPIError as e:
            logger.error(f"Failed to delete file: {e}")
            raise StorageError(
                f"Failed to delete file: {e}", operation="delete", path=blob_path
            ) from e

    async def delete_files(self, prefix: str) -> int:
        """
        Delete all files matching a prefix.

        Args:
            prefix: Prefix path to match files for deletion

        Returns:
            Number of files deleted

        Raises:
            StorageError: If deletion fails
        """
        try:
            blobs = await self.list_files(prefix=prefix)

            if not blobs:
                logger.info(f"No files found with prefix '{prefix}'")
                return 0

            batch_size = 100
            deleted_count = 0

            for i in range(0, len(blobs), batch_size):
                batch = blobs[i : i + batch_size]
                await asyncio.to_thread(self.bucket.delete_blobs, batch)
                deleted_count += len(batch)

            logger.info(f"Deleted {deleted_count} files with prefix '{prefix}'")
            return deleted_count

        except GoogleAPIError as e:
            logger.error(f"Failed to delete files: {e}")
            raise StorageError(
                f"Failed to delete files: {e}", operation="delete", path=prefix
            ) from e

    async def file_exists(self, blob_path: str) -> bool:
        """
        Check if a file exists in GCS.

        Args:
            blob_path: Blob path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_path)
            exists = await asyncio.to_thread(blob.exists)
            return exists
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False

    # ========================================================================
    # Path Builder Helpers
    # ========================================================================

    def build_path(self, storage_path: StoragePath) -> str:
        """
        Build a complete blob path from StoragePath configuration.

        Args:
            storage_path: StoragePath configuration object

        Returns:
            Complete blob path string
        """
        return storage_path.build()

    def build_job_path(
        self,
        job_id: str,
        folder: str | None = None,
        filename: str | None = None,
        prefix: str | None = None,
    ) -> str:
        """
        Build a job-specific path.

        Args:
            job_id: Job identifier
            folder: Optional subfolder (e.g., 'input', 'output')
            filename: Optional filename
            prefix: Optional prefix (defaults to GCS_TRANSLATION_PREFIX)

        Returns:
            Complete blob path
        """
        path = StoragePath(
            prefix=prefix or settings.GCS_TRANSLATION_PREFIX,
            job_id=job_id,
            folder=folder,
            filename=filename,
        )
        return path.build()

    def build_asset_path(self, filename: str) -> str:
        """
        Build an asset-specific path.

        Args:
            filename: Asset filename

        Returns:
            Complete blob path
        """
        path = StoragePath(prefix=settings.GCS_ASSETS_PREFIX, filename=filename)
        return path.build()

    # ========================================================================
    # URL & Metadata Operations
    # ========================================================================

    async def generate_signed_url(
        self, blob_path: str, expires_in: int = 3600, method: str = "GET"
    ) -> str:
        """
        Generate signed URL for file access.

        Args:
            blob_path: Blob path or GCS URI (gs://bucket/path)
            expires_in: Expiration time in seconds
            method: HTTP method for the URL

        Returns:
            Signed URL string

        Raises:
            StorageError: If URL generation fails
        """
        try:
            blob_path = self._extract_blob_path(blob_path)
            blob = self.bucket.blob(blob_path)
            expiration = datetime.now(UTC) + timedelta(seconds=expires_in)

            url = await asyncio.to_thread(
                blob.generate_signed_url,
                expiration=expiration,
                method=method,
                version="v4",
            )

            logger.debug(f"Generated signed URL for {blob_path}")
            return url

        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            raise StorageError(
                f"Failed to generate signed URL: {e}",
                operation="sign_url",
                path=blob_path,
            ) from e

    async def get_file_metadata(self, blob_path: str) -> dict[str, Any]:
        """
        Get file metadata from GCS.

        Args:
            blob_path: Blob path or GCS URI

        Returns:
            Dict with file metadata

        Raises:
            StorageError: If metadata retrieval fails
        """
        try:
            blob_path = self._extract_blob_path(blob_path)
            blob = self.bucket.blob(blob_path)
            await asyncio.to_thread(blob.reload)

            return {
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "updated": blob.updated,
                "created": blob.time_created,
                "md5_hash": blob.md5_hash,
                "metadata": blob.metadata or {},
            }

        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            raise StorageError(
                f"Failed to get file metadata: {e}",
                operation="metadata",
                path=blob_path,
            ) from e

    def _extract_blob_path(self, path: str) -> str:
        """
        Extract blob path from GCS URI or return as-is.

        Args:
            path: Blob path or GCS URI (gs://bucket/path)

        Returns:
            Clean blob path
        """
        if path.startswith("gs://"):
            path = path[5:]

        if path.startswith(self.bucket_name + "/"):
            return path[len(self.bucket_name) + 1 :]

        return path


# ============================================================================
# Generic GCS Helper Functions (usable anywhere)
# ============================================================================


def download_blob_sync(
    blob_path: str,
    local_path: Path,
    bucket_name: str | None = None,
    prefix: str | None = None,
    client: storage.Client | None = None,
) -> Path:
    """
    Generic synchronous download from GCS.

    Args:
        blob_path: Blob path (relative if prefix provided, absolute otherwise)
        local_path: Local destination path
        bucket_name: GCS bucket name (defaults to settings.GCS_BUCKET_NAME)
        prefix: Optional prefix to prepend to blob_path
        client: Optional GCS client (creates new one if not provided)

    Returns:
        Path to downloaded file

    Raises:
        Exception: If download fails
    """
    bucket_name = bucket_name or settings.GCS_BUCKET_NAME
    full_blob_path = f"{prefix}/{blob_path}" if prefix else blob_path

    logger.info(f"Downloading gs://{bucket_name}/{full_blob_path} to {local_path}")

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gcs_client = client or storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(full_blob_path)
        blob.download_to_filename(str(local_path))
        logger.info(f"Successfully downloaded {local_path.name}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {full_blob_path} from GCS: {e}")
        raise


async def download_blob_async(
    blob_path: str,
    local_path: Path,
    bucket_name: str | None = None,
    prefix: str | None = None,
    client: storage.Client | None = None,
) -> Path:
    """
    Generic asynchronous download from GCS.

    Args:
        blob_path: Blob path (relative if prefix provided, absolute otherwise)
        local_path: Local destination path
        bucket_name: GCS bucket name (defaults to settings.GCS_BUCKET_NAME)
        prefix: Optional prefix to prepend to blob_path
        client: Optional GCS client (creates new one if not provided)

    Returns:
        Path to downloaded file
    """
    return await asyncio.to_thread(
        download_blob_sync, blob_path, local_path, bucket_name, prefix, client
    )


def upload_blob_sync(
    local_path: Path,
    blob_path: str,
    bucket_name: str | None = None,
    prefix: str | None = None,
    content_type: str | None = None,
    client: storage.Client | None = None,
) -> str:
    """
    Generic synchronous upload to GCS.

    Args:
        local_path: Local file path to upload
        blob_path: Destination blob path (relative if prefix provided)
        bucket_name: GCS bucket name (defaults to settings.GCS_BUCKET_NAME)
        prefix: Optional prefix to prepend to blob_path
        content_type: Optional content type
        client: Optional GCS client (creates new one if not provided)

    Returns:
        GCS URI (gs://bucket/path)

    Raises:
        Exception: If upload fails
    """
    bucket_name = bucket_name or settings.GCS_BUCKET_NAME
    full_blob_path = f"{prefix}/{blob_path}" if prefix else blob_path

    logger.info(f"Uploading {local_path} to gs://{bucket_name}/{full_blob_path}")

    try:
        gcs_client = client or storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(full_blob_path)

        if content_type:
            blob.upload_from_filename(str(local_path), content_type=content_type)
        else:
            blob.upload_from_filename(str(local_path))

        logger.info(f"Successfully uploaded {local_path.name}")
        return f"gs://{bucket_name}/{full_blob_path}"
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCS: {e}")
        raise


async def upload_blob_async(
    local_path: Path,
    blob_path: str,
    bucket_name: str | None = None,
    prefix: str | None = None,
    content_type: str | None = None,
    client: storage.Client | None = None,
) -> str:
    """
    Generic asynchronous upload to GCS.

    Args:
        local_path: Local file path to upload
        blob_path: Destination blob path (relative if prefix provided)
        bucket_name: GCS bucket name (defaults to settings.GCS_BUCKET_NAME)
        prefix: Optional prefix to prepend to blob_path
        content_type: Optional content type
        client: Optional GCS client (creates new one if not provided)

    Returns:
        GCS URI (gs://bucket/path)
    """
    return await asyncio.to_thread(
        upload_blob_sync,
        local_path,
        blob_path,
        bucket_name,
        prefix,
        content_type,
        client,
    )


def list_blobs(
    bucket_name: str | None = None,
    prefix: str | None = None,
    client: storage.Client | None = None,
) -> list[storage.Blob]:
    """
    Generic function to list blobs in GCS bucket.

    Args:
        bucket_name: GCS bucket name (defaults to settings.GCS_BUCKET_NAME)
        prefix: Optional prefix to filter blobs
        client: Optional GCS client (creates new one if not provided)

    Returns:
        List of blob objects
    """
    bucket_name = bucket_name or settings.GCS_BUCKET_NAME

    try:
        gcs_client = client or storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        return list(bucket.list_blobs(prefix=prefix))
    except Exception as e:
        logger.error(f"Failed to list blobs from GCS: {e}")
        raise


async def list_blobs_async(
    bucket_name: str | None = None,
    prefix: str | None = None,
    client: storage.Client | None = None,
) -> list[storage.Blob]:
    """
    Generic async function to list blobs in GCS bucket.

    Args:
        bucket_name: GCS bucket name (defaults to settings.GCS_BUCKET_NAME)
        prefix: Optional prefix to filter blobs
        client: Optional GCS client (creates new one if not provided)

    Returns:
        List of blob objects
    """
    return await asyncio.to_thread(list_blobs, bucket_name, prefix, client)


# ============================================================================
# Backward Compatibility Functions (for loaders/assets.py)
# ============================================================================


def download_from_gcs_sync(blob_path: str, local_path: Path) -> None:
    """
    Download asset file from GCS (backward compatible).

    Args:
        blob_path: Relative blob path (without assets prefix)
        local_path: Local destination path
    """
    download_blob_sync(
        blob_path=blob_path, local_path=local_path, prefix=settings.GCS_ASSETS_PREFIX
    )


async def download_from_gcs_async(blob_path: str, local_path: Path) -> None:
    """
    Async download asset file from GCS (backward compatible).

    Args:
        blob_path: Relative blob path (without assets prefix)
        local_path: Local destination path
    """
    await download_blob_async(
        blob_path=blob_path, local_path=local_path, prefix=settings.GCS_ASSETS_PREFIX
    )


def list_gcs_blobs() -> list[storage.Blob]:
    """
    List asset blobs from GCS (backward compatible).

    Returns:
        List of blob objects
    """
    return list_blobs(prefix=settings.GCS_ASSETS_PREFIX)


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def get_storage_repository(
    client: storage.Client | None = None, bucket_name: str | None = None
) -> StorageRepository:
    """
    Get a StorageRepository instance.

    Args:
        client: Optional GCS client
        bucket_name: Optional bucket name

    Returns:
        StorageRepository instance
    """
    return StorageRepository(client=client, bucket_name=bucket_name)
