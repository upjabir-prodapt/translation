"""Final document assembly service."""

from __future__ import annotations

from pathlib import Path

from src.repository.api_storage_repository import APIStorageRepository


class AssemblyService:
    """Upload translated outputs to object storage."""

    def __init__(self, storage: APIStorageRepository):
        self.storage = storage

    async def upload_output(
        self,
        job_id: str,
        local_path: Path,
        preferred_filename: str | None = None,
    ) -> str | None:
        """Upload a single translated output file."""
        if not local_path.exists():
            return None
        blob_filename = preferred_filename or local_path.name
        blob_path = self.storage.build_job_path(
            job_id=job_id,
            folder="output",
            filename=blob_filename,
        )
        return await self.storage.upload_file(
            source=local_path,
            blob_path=blob_path,
        )

    async def upload_outputs(
        self,
        job_id: str,
        output_files: dict[str, Path],
        preferred_filenames: dict[str, str] | None = None,
    ) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        for key, local_path in output_files.items():
            if not local_path.exists():
                continue
            blob_filename = (preferred_filenames or {}).get(key) or local_path.name
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
