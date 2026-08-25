"""Storage repository utilities for translation processing and assets."""

import logging
from pathlib import Path

from google.cloud import storage

from src.config.constants import settings
from src.repository.storage_repository import FileType
from src.repository.storage_repository import StorageError
from src.repository.storage_repository import StorageRepository

logger = logging.getLogger(__name__)


class TranslationStorageRepository(StorageRepository):
    """Storage repository for processing translation jobs and shared assets."""

    def __init__(
        self, client: storage.Client | None = None, bucket_name: str | None = None
    ):
        super().__init__(client=client, bucket_name=bucket_name)

    async def download_input_pdf(
        self, job_id: str, local_path: Path, filename: str = "input.pdf"
    ) -> Path:
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        logger.info(f"Downloading input PDF for job {job_id}")
        return await self.download_file(blob_path, local_path)

    async def upload_output_files(
        self, job_id: str, output_files: dict[str, Path]
    ) -> dict[str, str]:
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
            except StorageError:
                logger.exception(f"Failed to upload {file_type} for job {job_id}")
                raise

        return output_uris

    async def upload_attempt_artifacts(
        self,
        job_id: str,
        attempt_index: int,
        tracking_path: Path | None = None,
        quality_report_path: Path | None = None,
    ) -> list[str]:
        """Upload attempt tracking and quality report JSON to GCS input folder."""
        uploaded_uris: list[str] = []
        artifacts = [
            (tracking_path, f"iter_{attempt_index}_translate_tracking.json", "tracking"),
            (quality_report_path, f"iter_{attempt_index}_quality_report.json", "quality_report"),
        ]
        for path, filename, artifact_type in artifacts:
            if path is None or not path.exists():
                continue
            blob_path = self.build_job_path(
                job_id=job_id,
                folder=settings.GCS_INPUT_FOLDER,
                filename=filename,
            )
            try:
                uri = await self.upload_file(
                    source=path,
                    blob_path=blob_path,
                    file_type=FileType.JSON,
                    metadata={
                        "job_id": job_id,
                        "attempt_index": str(attempt_index),
                        "artifact_type": artifact_type,
                    },
                )
                uploaded_uris.append(uri)
                logger.info(f"Uploaded {filename} for job {job_id} to {uri}")
            except Exception:
                logger.warning(
                    f"Failed to upload {filename} for job {job_id}",
                    exc_info=True,
                )
        return uploaded_uris

    async def download_asset(self, blob_path: str, local_path: Path) -> Path:
        full_blob_path = f"{settings.GCS_ASSETS_PREFIX}/{blob_path}"
        logger.info(f"Downloading asset: {blob_path}")
        return await self.download_file(full_blob_path, local_path)

    def download_asset_sync(self, blob_path: str, local_path: Path) -> Path:
        from src.repository.storage_repository import download_blob_sync

        full_blob_path = f"{settings.GCS_ASSETS_PREFIX}/{blob_path}"
        logger.info(f"Downloading asset (sync): {blob_path}")
        return download_blob_sync(
            blob_path=full_blob_path,
            local_path=local_path,
            bucket_name=self.bucket_name,
            client=self.client,
        )

    async def list_assets(self) -> list[storage.Blob]:
        return await self.list_files(prefix=settings.GCS_ASSETS_PREFIX)

    async def download_glossary(
        self, job_id: str, local_path: Path, filename: str
    ) -> Path:
        blob_path = self.build_job_path(
            job_id=job_id, folder=settings.GCS_INPUT_FOLDER, filename=filename
        )
        logger.info(f"Downloading glossary for job {job_id}")
        return await self.download_file(blob_path, local_path)

    async def cleanup_job_files(self, job_id: str, keep_output: bool = True) -> int:
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


def get_translation_storage_repository(
    client: storage.Client | None = None, bucket_name: str | None = None
) -> TranslationStorageRepository:
    if client is None:
        from src.repository import get_storage_client

        client = get_storage_client()
    return TranslationStorageRepository(client=client, bucket_name=bucket_name)
