"""Worker Storage Repository - Worker-specific storage operations."""

from pathlib import Path

from google.cloud import storage

from config.constants import settings
from config.logging import logger
from repository.storage_repository import FileType
from repository.storage_repository import StorageError
from repository.storage_repository import StorageRepository


class WorkerStorageRepository(StorageRepository):
    """Worker-specific storage repository for processing translation jobs."""

    def __init__(
        self, client: storage.Client | None = None, bucket_name: str | None = None
    ):
        """Initialize Worker storage repository.

        Args:
            client: Optional GCS client
            bucket_name: Optional bucket name
        """
        super().__init__(client=client, bucket_name=bucket_name)

    async def download_input_pdf(
        self, job_id: str, local_path: Path, filename: str = "input.pdf"
    ) -> Path:
        """
        Download input PDF from GCS for processing.

        Args:
            job_id: Job identifier
            local_path: Local destination path
            filename: Name of the file in GCS

        Returns:
            Path to downloaded file

        Raises:
            StorageError: If download fails
        """
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        logger.info(f"Downloading input PDF for job {job_id}")
        return await self.download_file(blob_path, local_path)

    async def upload_output_files(
        self, job_id: str, output_files: dict[str, Path]
    ) -> dict[str, str]:
        """
        Upload processed output PDFs to GCS.

        Args:
            job_id: Job identifier
            output_files: Dict mapping file type to local path

        Returns:
            Dict mapping file type to GCS URI

        Raises:
            StorageError: If upload fails
        """
        output_uris = {}

        for file_type, file_path in output_files.items():
            if not file_path.exists():
                logger.warning(f"Output file {file_path} does not exist")
                continue

            blob_path = self.build_job_path(
                job_id=job_id,
                folder=settings.GCS_OUTPUT_FOLDER,
                filename=f"{file_type}.pdf",
            )

            try:
                gcs_uri = await self.upload_file(
                    source=file_path,
                    blob_path=blob_path,
                    file_type=FileType.PDF,
                    metadata={"job_id": job_id, "output_type": file_type},
                )
                output_uris[file_type] = gcs_uri
                logger.info(f"Uploaded {file_type} PDF for job {job_id}")
            except StorageError as e:
                logger.error(f"Failed to upload {file_type}: {e}")
                raise

        return output_uris

    async def download_asset(self, blob_path: str, local_path: Path) -> Path:
        """
        Download asset file from GCS for worker processing (async).

        Args:
            blob_path: Relative blob path (without assets prefix)
            local_path: Local destination path

        Returns:
            Path to downloaded file

        Raises:
            StorageError: If download fails
        """
        full_blob_path = f"{settings.GCS_ASSETS_PREFIX}/{blob_path}"
        logger.info(f"Downloading asset: {blob_path}")
        return await self.download_file(full_blob_path, local_path)

    def download_asset_sync(self, blob_path: str, local_path: Path) -> Path:
        """
        Download asset file from GCS for worker processing (synchronous).

        Args:
            blob_path: Relative blob path (without assets prefix)
            local_path: Local destination path

        Returns:
            Path to downloaded file

        Raises:
            StorageError: If download fails
        """
        from repository.storage_repository import download_blob_sync

        full_blob_path = f"{settings.GCS_ASSETS_PREFIX}/{blob_path}"
        logger.info(f"Downloading asset (sync): {blob_path}")
        return download_blob_sync(
            blob_path=full_blob_path,
            local_path=local_path,
            bucket_name=self.bucket_name,
            client=self.client,
        )

    async def list_assets(self) -> list[storage.Blob]:
        """
        List all asset blobs for worker processing.

        Returns:
            List of blob objects

        Raises:
            StorageError: If listing fails
        """
        return await self.list_files(prefix=settings.GCS_ASSETS_PREFIX)

    async def download_glossary(
        self, job_id: str, local_path: Path, filename: str
    ) -> Path:
        """
        Download glossary CSV from GCS for processing.

        Args:
            job_id: Job identifier
            local_path: Local destination path
            filename: Name of the glossary file

        Returns:
            Path to downloaded file

        Raises:
            StorageError: If download fails
        """
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        logger.info(f"Downloading glossary for job {job_id}")
        return await self.download_file(blob_path, local_path)

    async def cleanup_job_files(self, job_id: str, keep_output: bool = True) -> int:
        """
        Clean up job files after processing.

        Args:
            job_id: Job identifier
            keep_output: Whether to keep output files

        Returns:
            Number of files deleted
        """
        if keep_output:
            input_prefix = (
                self.build_job_path(job_id=job_id, folder=settings.GCS_INPUT_FOLDER)
                + "/"
            )
            deleted_count = await self.delete_files(input_prefix)
            logger.info(f"Cleaned up {deleted_count} input files for job {job_id}")
        else:
            job_prefix = f"{settings.GCS_TRANSLATION_PREFIX}/{job_id}/"
            deleted_count = await self.delete_files(job_prefix)
            logger.info(f"Cleaned up all {deleted_count} files for job {job_id}")

        return deleted_count


def get_worker_storage_repository(
    client: storage.Client | None = None, bucket_name: str | None = None
) -> WorkerStorageRepository:
    """
    Factory function to get a Worker storage repository instance.

    Args:
        client: Optional GCS client (uses cached client if not provided)
        bucket_name: Optional bucket name

    Returns:
        WorkerStorageRepository instance
    """
    if client is None:
        from repository import get_storage_client

        client = get_storage_client()
    return WorkerStorageRepository(client=client, bucket_name=bucket_name)
