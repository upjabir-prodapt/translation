"""Final document assembly service."""

from __future__ import annotations

from pathlib import Path

from src.repository.api_storage_repository import APIStorageRepository


class AssemblyService:
    """Upload translated outputs to object storage."""

    def __init__(self, storage: APIStorageRepository):
        self.storage = storage

    async def upload_outputs(self, job_id: str, output_files: dict[str, Path]) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        for key, local_path in output_files.items():
            if not local_path.exists():
                continue
            blob_filename = local_path.name
            blob_path = self.storage.build_job_path(
                job_id=job_id,
                folder="output",
                filename=blob_filename,
            )
            uploaded[key] = await self.storage.upload_file(
                source=local_path,
                blob_path=blob_path,
            )
        return uploaded

