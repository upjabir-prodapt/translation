"""API Storage Repository - API-specific storage operations."""

from typing import Any

from google.cloud import storage

from src.config.constants import settings
import logging
logger = logging.getLogger(__name__)
from src.repository.storage_repository import FileType
from src.repository.storage_repository import StorageRepository


class APIStorageRepository(StorageRepository):
    """API-specific storage repository with job and file management methods."""

    def __init__(
        self, client: storage.Client | None = None, bucket_name: str | None = None
    ):
        super().__init__(client=client, bucket_name=bucket_name)

    async def upload_input_pdf(
        self, file_content: bytes, filename: str, job_id: str
    ) -> str:
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        return await self.upload_file(
            source=file_content,
            blob_path=blob_path,
            file_type=FileType.PDF,
            metadata={"job_id": job_id, "file_type": "input"},
        )

    async def upload_glossary(
        self, file_content: bytes, filename: str, job_id: str
    ) -> str:
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        return await self.upload_file(
            source=file_content,
            blob_path=blob_path,
            file_type=FileType.CSV,
            metadata={"job_id": job_id, "file_type": "glossary"},
        )

    async def get_job_files(self, job_id: str) -> list[storage.Blob]:
        prefix = f"{settings.GCS_TRANSLATION_PREFIX}/{job_id}/"
        return await self.list_files(prefix=prefix)

    async def get_job_input_files(self, job_id: str) -> list[storage.Blob]:
        prefix = (
            self.build_job_path(job_id=job_id, folder=settings.GCS_INPUT_FOLDER) + "/"
        )
        return await self.list_files(prefix=prefix)

    async def get_job_output_files(self, job_id: str) -> list[storage.Blob]:
        prefix = (
            self.build_job_path(job_id=job_id, folder=settings.GCS_OUTPUT_FOLDER) + "/"
        )
        return await self.list_files(prefix=prefix)

    async def delete_job_files(self, job_id: str) -> int:
        job_prefix = f"{settings.GCS_TRANSLATION_PREFIX}/{job_id}/"
        deleted_count = await self.delete_files(job_prefix)
        logger.info(f"Deleted {deleted_count} files for job {job_id}")
        return deleted_count

    async def generate_download_url(
        self, job_id: str, filename: str, folder: str = "output", expires_in: int = 3600
    ) -> str:
        blob_path = self.build_job_path(job_id=job_id, folder=folder, filename=filename)
        return await self.generate_signed_url(
            blob_path=blob_path, expires_in=expires_in, method="GET"
        )

    async def get_file_info(self, gcs_uri: str) -> dict[str, Any]:
        return await self.get_file_metadata(gcs_uri)


def get_api_storage_repository(
    client: storage.Client | None = None, bucket_name: str | None = None
) -> APIStorageRepository:
    if client is None:
        from src.repository import get_storage_client

        client = get_storage_client()
    return APIStorageRepository(client=client, bucket_name=bucket_name)
