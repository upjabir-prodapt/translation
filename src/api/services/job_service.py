"""Job management service."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from fastapi import status

from api.exceptions import JobAlreadyCompletedError
from api.exceptions import JobNotFoundError
from api.repository.api_storage_repository import APIStorageRepository
from api.schemas.requests import JobCancelRequest
from api.schemas.responses import DownloadResponse
from api.schemas.responses import JobDetailResponse
from api.schemas.responses import JobListResponse
from api.schemas.responses import JobStatusResponse
from api.schemas.responses import TranslatedDocumentResult
from api.schemas.responses import TranslationLabels
from api.schemas.responses import TranslationMetadata
from api.schemas.responses import TranslationResult
from config.logging import logger
from repository.firestore_repository import FirestoreRepository


class JobService:
    """Service for managing translation jobs."""

    def __init__(
        self,
        firestore: FirestoreRepository | None = None,
        storage: APIStorageRepository | None = None,
    ):
        self.firestore = firestore or FirestoreRepository()
        self.storage = storage or APIStorageRepository()

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Get the current status of a job."""
        job_data = await self.firestore.get_job(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        cost_attribution = job_data.get("cost_attribution", {})
        return JobStatusResponse(
            job_id=job_data["job_id"],
            status=job_data["status"],
            progress=job_data.get("progress", 0.0),
            current_stage=job_data.get("current_stage"),
            user=cost_attribution.get("user_id", ""),
            department=cost_attribution.get("business_unit", ""),
            created_at=job_data["created_at"],
            updated_at=job_data["updated_at"],
            completed_at=job_data.get("completed_at"),
            error_message=job_data.get("error_message"),
        )

    async def get_translation_status(self, job_id: str) -> JobDetailResponse:
        """Get full translation status and result for GET /translate/{job_id}."""
        job_data = await self.firestore.get_job(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        status = job_data.get("status", "unknown")
        timestamps = job_data.get("timestamps", {})
        submitted_at = timestamps.get("submitted_at") or job_data.get("created_at")
        completed_at = timestamps.get("completed_at") or job_data.get("completed_at")

        result: TranslationResult | None = None

        if status == "completed":
            raw_result = job_data.get("result", {}) or {}
            source_doc = job_data.get("source_document", {}) or {}
            translation_cfg = job_data.get("translation_config", {}) or {}
            processing = job_data.get("processing", {}) or {}

            # Build download URL from GCS URI
            download_url: str | None = None
            output_gcs_uri = raw_result.get("output_gcs_uri")
            if output_gcs_uri:
                try:
                    download_url = await self.storage.generate_signed_url(
                        blob_path=output_gcs_uri, expires_in=3600
                    )
                except Exception:
                    logger.warning(f"Could not generate download URL for job {job_id}")

            # Determine output filename
            output_filename = (
                source_doc.get("output_filename") or f"{job_id}_translated.pdf"
            )

            translated_doc = TranslatedDocumentResult(
                content=None,  # not returned inline; use download_url
                format=source_doc.get("format", "pdf"),
                filename=output_filename,
                download_url=download_url,
            )

            metadata = TranslationMetadata(
                source_language=source_doc.get("source_language")
                or translation_cfg.get("source_language"),
                target_language=translation_cfg.get("target_language"),
                domain=translation_cfg.get("domain"),
                model_used=processing.get("model_used"),
                model_version=processing.get("model_version"),
                quality_score=raw_result.get("confidence_score"),
                ab_test_variant=processing.get("ab_variant"),
                chunks_processed=processing.get("chunks"),
                retry_attempts=processing.get("retry_count", 0),
            )

            labels = TranslationLabels(
                translation_intent=translation_cfg.get("intent"),
                processing_time_seconds=job_data.get("processing_seconds"),
                token_count=raw_result.get("token_count"),
                cost_usd=raw_result.get("cost_usd"),
            )

            result = TranslationResult(
                translated_document=translated_doc,
                metadata=metadata,
                labels=labels,
            )

        return JobDetailResponse(
            job_id=job_data["job_id"],
            status=status,
            submitted_at=submitted_at,
            completed_at=completed_at,
            result=result,
        )

    async def list_jobs(
        self, status: str | None = None, limit: int = 10, offset: int = 0
    ) -> JobListResponse:
        """List jobs with filtering and pagination."""
        # Get jobs from Firestore
        jobs = await self.firestore.list_jobs(status=status, limit=limit, offset=offset)

        # Convert to response format
        job_responses = []
        for job in jobs:
            cost_attribution = job.get("cost_attribution", {})
            job_responses.append(
                JobStatusResponse(
                    job_id=job["job_id"],
                    status=job["status"],
                    progress=job.get("progress", 0.0),
                    current_stage=job.get("current_stage"),
                    user=cost_attribution.get("user_id", ""),
                    department=cost_attribution.get("business_unit", ""),
                    created_at=job["created_at"],
                    updated_at=job["updated_at"],
                    completed_at=job.get("completed_at"),
                    error_message=job.get("error_message"),
                )
            )

        # Get total count (simplified - in production use a separate count query)
        total = len(job_responses)

        return JobListResponse(
            jobs=job_responses, total=total, limit=limit, offset=offset
        )

    async def cancel_job(self, job_id: str, request: JobCancelRequest) -> None:
        """Cancel a translation job."""
        # Get current job status
        job_data = await self.firestore.get_job(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        status = job_data["status"]

        # Check if job can be cancelled
        if status in ["completed", "failed", "cancelled"]:
            raise JobAlreadyCompletedError(job_id)

        # Update job status
        updates = {
            "status": "cancelled",
            "error_message": f"Cancelled by user: {request.reason}"
            if request.reason
            else "Cancelled by user",
            "updated_at": datetime.now(UTC),
        }

        await self.firestore.update_job(job_id, updates)

        # TODO: Send cancellation signal to worker if currently processing
        logger.info(f"Cancelled job {job_id}")

    async def get_download_url(self, job_id: str, file_type: str) -> DownloadResponse:
        """Generate a signed URL for downloading output files."""
        # Get job data
        job_data = await self.firestore.get_job(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        if job_data["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is not completed yet",
            )

        # Get output URIs
        output_uris = job_data.get("output_gs_uris", {})

        if file_type not in output_uris:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Output file {file_type} not found",
            )

        # Generate signed URL
        gcs_uri = output_uris[file_type]
        download_url = await self.storage.generate_signed_url(
            blob_path=gcs_uri,
            expires_in=3600,  # 1 hour
        )

        # Get file info
        file_info = await self.storage.get_file_info(gcs_uri)

        return DownloadResponse(
            download_url=download_url,
            expires_in=3600,
            filename=f"{job_id}_{file_type}.pdf",
            file_size=file_info["size"],
        )

    async def stream_job_progress(
        self, job_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream job progress updates."""
        # Get initial state
        last_update = None

        while True:
            job_data = await self.firestore.get_job(job_id)

            if not job_data:
                yield {"type": "error", "message": "Job not found"}
                break

            # Check if job is finished
            if job_data["status"] in ["completed", "failed", "cancelled"]:
                yield {
                    "type": "complete",
                    "status": job_data["status"],
                    "progress": job_data.get("progress", 1.0),
                    "error_message": job_data.get("error_message"),
                }
                break

            # Check for updates
            current_update = job_data["updated_at"]
            if last_update != current_update:
                yield {
                    "type": "progress",
                    "status": job_data["status"],
                    "progress": job_data.get("progress", 0.0),
                    "current_stage": job_data.get("current_stage"),
                }
                last_update = current_update

            # Wait before next check
            await asyncio.sleep(1)
