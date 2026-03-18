"""API Storage Repository - API-specific storage operations."""

from typing import Any

from google.cloud import storage

from config.constants import settings
from config.logging import logger
from repository.storage_repository import FileType
from repository.storage_repository import StorageRepository


class APIStorageRepository(StorageRepository):
    """API-specific storage repository with job and file management methods."""

    def __init__(
        self, client: storage.Client | None = None, bucket_name: str | None = None
    ):
        """Initialize API storage repository.

        Args:
            client: Optional GCS client
            bucket_name: Optional bucket name
        """
        super().__init__(client=client, bucket_name=bucket_name)

    async def upload_input_pdf(
        self, file_content: bytes, filename: str, job_id: str
    ) -> str:
        """
        Upload input PDF to GCS for a translation job.

        Args:
            file_content: PDF file content as bytes
            filename: Name of the file
            job_id: Job identifier

        Returns:
            GCS URI of uploaded file

        Raises:
            StorageError: If upload fails
        """
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        return await self.upload_file(
            source=file_content,
            blob_path=blob_path,
            file_type=FileType.PDF,
            metadata={"job_id": job_id, "file_type": "input"},
        )

    async def get_job_files(self, job_id: str) -> list[storage.Blob]:
        """
        List all files for a specific job.

        Args:
            job_id: Job identifier

        Returns:
            List of blob objects for the job

        Raises:
            StorageError: If listing fails
        """
        prefix = f"{settings.GCS_TRANSLATION_PREFIX}/{job_id}/"
        return await self.list_files(prefix=prefix)

    async def get_job_input_files(self, job_id: str) -> list[storage.Blob]:
        """
        List input files for a specific job.

        Args:
            job_id: Job identifier

        Returns:
            List of input blob objects
        """
        prefix = (
            self.build_job_path(job_id=job_id, folder=settings.GCS_INPUT_FOLDER) + "/"
        )
        return await self.list_files(prefix=prefix)

    async def get_job_output_files(self, job_id: str) -> list[storage.Blob]:
        """
        List output files for a specific job.

        Args:
            job_id: Job identifier

        Returns:
            List of output blob objects
        """
        prefix = (
            self.build_job_path(job_id=job_id, folder=settings.GCS_OUTPUT_FOLDER) + "/"
        )
        return await self.list_files(prefix=prefix)

    async def delete_job_files(self, job_id: str) -> int:
        """
        Delete all files associated with a job.

        Args:
            job_id: Job identifier

        Returns:
            Number of files deleted

        Raises:
            StorageError: If deletion fails
        """
        job_prefix = f"{settings.GCS_TRANSLATION_PREFIX}/{job_id}/"
        deleted_count = await self.delete_files(job_prefix)
        logger.info(f"Deleted {deleted_count} files for job {job_id}")
        return deleted_count

    async def generate_download_url(
        self, job_id: str, filename: str, folder: str = "output", expires_in: int = 3600
    ) -> str:
        """
        Generate signed download URL for a job file.

        Args:
            job_id: Job identifier
            filename: Name of the file
            folder: Folder ('input' or 'output')
            expires_in: Expiration time in seconds

        Returns:
            Signed URL string

        Raises:
            StorageError: If URL generation fails
        """
        blob_path = self.build_job_path(job_id=job_id, folder=folder, filename=filename)
        return await self.generate_signed_url(
            blob_path=blob_path, expires_in=expires_in, method="GET"
        )

    async def get_file_info(self, gcs_uri: str) -> dict[str, Any]:
        """
        Get file information from GCS (backward compatible alias).

        Args:
            gcs_uri: GCS URI or blob path

        Returns:
            Dict with file metadata
        """
        return await self.get_file_metadata(gcs_uri)


def get_api_storage_repository(
    client: storage.Client | None = None, bucket_name: str | None = None
) -> APIStorageRepository:
    """
    Factory function to get an API storage repository instance.

    Args:
        client: Optional GCS client (uses cached client if not provided)
        bucket_name: Optional bucket name

    Returns:
        APIStorageRepository instance
    """
    if client is None:
        from repository import get_storage_client

        client = get_storage_client()
    return APIStorageRepository(client=client, bucket_name=bucket_name)
