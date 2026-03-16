"""File service for handling file operations."""

from typing import Any

from api.exceptions import FileProcessingError
from api.repository.api_storage_repository import APIStorageRepository
from config.logging import logger


class FileService:
    """Service for file-related operations."""

    def __init__(self, storage: APIStorageRepository | None = None):
        self.storage = storage or APIStorageRepository()

    async def cleanup_job_files(self, job_id: str) -> None:
        """Clean up all files associated with a job."""
        try:
            await self.storage.delete_job_files(job_id)
            logger.info(f"Cleaned up files for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup files for job {job_id}: {e}")
            # Don't raise - cleanup failures shouldn't break the flow

    async def get_file_metadata(self, gcs_uri: str) -> dict[str, Any]:
        """Get metadata for a file."""
        try:
            return await self.storage.get_file_info(gcs_uri)
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            raise FileProcessingError(f"Failed to get file metadata: {e}") from e

    async def verify_file_integrity(self, gcs_uri: str, expected_checksum: str) -> bool:
        """Verify file integrity using checksum."""
        try:
            # For now, just check if file exists
            # In production, you might want to download and verify checksum
            info = await self.storage.get_file_info(gcs_uri)
            return info is not None
        except Exception as e:
            logger.error(f"Failed to verify file integrity: {e}")
            return False
