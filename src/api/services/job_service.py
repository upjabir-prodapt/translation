"""Job management service."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi import status

from src.api.exceptions import JobAlreadyCompletedError
from src.api.exceptions import JobNotFoundError
from src.api.schemas.requests import JobCancelRequest
from src.api.schemas.responses import DownloadResponse
from src.api.schemas.responses import JobDetailResponse
from src.api.schemas.responses import JobListResponse
from src.api.schemas.responses import JobStatusResponse
from src.api.schemas.responses import MultiJobStatusItemResponse
from src.api.schemas.responses import MultiJobStatusResponse
from src.api.schemas.responses import TranslatedDocumentResult
from src.api.schemas.responses import TranslationLabels
from src.api.schemas.responses import TranslationMetadata
from src.api.schemas.responses import TranslationResult
from src.api.services.cloud_tasks_service import CloudTasksService
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository

logger = logging.getLogger(__name__)


class JobService:
    """Service for managing translation jobs."""

    def __init__(
        self,
        storage: APIStorageRepository | None = None,
        bigquery: BigQueryRepository | None = None,
        cloud_tasks: CloudTasksService | None = None,
    ):
        self.storage = storage or APIStorageRepository()
        self.bigquery = bigquery or BigQueryRepository()
        # Used only to delete a queued task when a job is cancelled.
        self.cloud_tasks = cloud_tasks or CloudTasksService()

    async def _get_job_data(self, job_id: str) -> dict[str, Any] | None:
        return await self.bigquery.get_translation_job(job_id)

    @staticmethod
    def _result_payload(job_data: dict[str, Any]) -> dict[str, Any]:
        result = job_data.get("result")
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _progress_and_stage(status: str) -> tuple[float, str | None]:
        if status == "queued":
            return 0.0, "queued"
        if status == "processing":
            return 0.5, "processing"
        if status in ("completed", "human_review_required", "failed", "cancelled"):
            return 1.0, status
        return 0.0, status

    @staticmethod
    def _job_error_message(job_data: dict[str, Any]) -> str | None:
        msg = job_data.get("error_message")
        if msg:
            return str(msg)
        result = job_data.get("result")
        if isinstance(result, dict) and result.get("error_message"):
            return str(result["error_message"])
        return None

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Get the current status of a job."""
        job_data = await self._get_job_data(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        status = str(job_data.get("status", ""))
        progress, current_stage = self._progress_and_stage(status)
        download_url: str | None = None
        if status in ("completed", "human_review_required"):
            result_payload = self._result_payload(job_data)
            output_gcs_uri = result_payload.get("output_gcs_uri")
            if output_gcs_uri:
                try:
                    download_url = await self.storage.generate_signed_url(
                        blob_path=output_gcs_uri, expires_in=3600
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not generate download URL for job {job_id}: {e}"
                    )

        cost_attribution = job_data.get("cost_attribution", {})
        return JobStatusResponse(
            job_id=job_data["job_id"],
            status=job_data["status"],
            progress=progress,
            current_stage=current_stage,
            user=cost_attribution.get("user_id", ""),
            department=cost_attribution.get("business_unit", ""),
            created_at=job_data.get("submitted_at"),
            updated_at=job_data.get("completed_at") or job_data.get("submitted_at"),
            completed_at=job_data.get("completed_at"),
            download_url=download_url,
            error_message=self._job_error_message(job_data),
        )

    async def get_jobs_status(
        self, job_ids: list[str], user_id: str
    ) -> MultiJobStatusResponse:
        """Get authorized job statuses in the caller's requested order."""
        jobs = await self.bigquery.get_translation_jobs_by_ids(job_ids)
        jobs_by_id = {str(job.get("job_id")): job for job in jobs}
        if any(job_id not in jobs_by_id for job_id in job_ids):
            raise JobNotFoundError(job_ids[0])
        if any(
            (jobs_by_id[job_id].get("cost_attribution") or {}).get("user_id") != user_id
            for job_id in job_ids
        ):
            raise JobNotFoundError(job_ids[0])

        responses: list[MultiJobStatusItemResponse] = []
        for job_id in job_ids:
            job = jobs_by_id[job_id]
            status = str(job.get("status", ""))
            result = self._result_payload(job)
            source_document = job.get("source_document") or {}
            download_url = None
            download_filename = None
            if status in ("completed", "human_review_required"):
                output_gcs_uri = result.get("output_gcs_uri")
                if output_gcs_uri:
                    try:
                        download_url = await self.storage.generate_signed_url(
                            blob_path=output_gcs_uri, expires_in=3600
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not generate download URL for job %s: %s",
                            job_id,
                            e,
                        )
                download_filename = result.get(
                    "download_filename"
                ) or source_document.get("output_filename")
            responses.append(
                MultiJobStatusItemResponse(
                    job_id=job_id,
                    target_language=(job.get("translation_config") or {}).get(
                        "target_language"
                    ),
                    status=status,
                    download_url=download_url,
                    download_filename=download_filename,
                    error_message=self._job_error_message(job),
                )
            )
        return MultiJobStatusResponse(jobs=responses)

    async def get_translation_status(self, job_id: str) -> JobDetailResponse:
        """Get full translation status and result for GET /translate/{job_id}."""
        job_data = await self._get_job_data(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        status = job_data.get("status", "unknown")
        timestamps = (
            job_data.get("timestamps", {})
            if isinstance(job_data.get("timestamps"), dict)
            else {}
        )
        submitted_at = (
            job_data.get("submitted_at")
            or timestamps.get("submitted_at")
            or job_data.get("created_at")
        )
        completed_at = job_data.get("completed_at") or timestamps.get("completed_at")

        result: TranslationResult | None = None

        if status in ("completed", "human_review_required"):
            result = await self._build_translation_result(job_id, job_data)

        return JobDetailResponse(
            job_id=job_data["job_id"],
            status=status,
            submitted_at=submitted_at,
            completed_at=completed_at,
            result=result,
            error_message=self._job_error_message(job_data),
        )

    async def _build_translation_result(
        self, job_id: str, job_data: dict[str, Any]
    ) -> TranslationResult:
        raw_result = job_data.get("result", {}) or {}
        source_doc = job_data.get("source_document", {}) or {}
        translation_cfg = job_data.get("translation_config", {}) or {}

        # Build download URL from GCS URI
        download_url: str | None = None
        output_gcs_uri = raw_result.get("output_gcs_uri")
        if output_gcs_uri:
            try:
                download_url = await self.storage.generate_signed_url(
                    blob_path=output_gcs_uri, expires_in=3600
                )
            except Exception as e:
                logger.warning(f"Could not generate download URL for job {job_id}: {e}")

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
            model_used=raw_result.get("model_used"),
            model_version=raw_result.get("model_version"),
            quality_score=raw_result.get("confidence_score"),
            ab_test_variant=raw_result.get("ab_variant"),
            chunks_processed=raw_result.get("chunks"),
            retry_attempts=int(raw_result.get("retry_count", 0) or 0),
        )

        labels = TranslationLabels(
            translation_intent=raw_result.get("intent")
            or translation_cfg.get("intent"),
            processing_time_seconds=job_data.get("processing_seconds"),
            token_count=raw_result.get("token_count"),
            cost_usd=raw_result.get("cost_usd"),
        )

        return TranslationResult(
            translated_document=translated_doc,
            metadata=metadata,
            labels=labels,
        )

    async def list_jobs(
        self,
        status: str | None = None,
        limit: int = 10,
        offset: int = 0,
        user_id: str | None = None,
    ) -> JobListResponse:
        """List jobs with filtering and pagination, scoped to the requesting user.

        Only jobs submitted within the last 7 days are returned, matching the
        retention window the UI advertises for job history.
        """
        submitted_after = datetime.now(UTC) - timedelta(days=7)
        jobs = await self.bigquery.list_translation_jobs(
            status=status,
            limit=limit,
            offset=offset,
            user_id=user_id,
            submitted_after=submitted_after,
        )

        # Convert to response format
        job_responses = []
        for job in jobs:
            progress, current_stage = self._progress_and_stage(
                str(job.get("status", ""))
            )
            cost_attribution = job.get("cost_attribution", {}) or {}
            source_document = job.get("source_document", {}) or {}
            translation_config = job.get("translation_config", {}) or {}
            result_payload = self._result_payload(job)
            status_value = str(job.get("status", ""))
            download_url: str | None = None
            if status_value in ("completed", "human_review_required"):
                output_gcs_uri = result_payload.get("output_gcs_uri")
                if output_gcs_uri:
                    try:
                        download_url = await self.storage.generate_signed_url(
                            blob_path=output_gcs_uri, expires_in=3600
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not generate download URL for job %s: %s",
                            job.get("job_id"),
                            e,
                        )
            job_responses.append(
                JobStatusResponse(
                    job_id=job["job_id"],
                    status=job["status"],
                    progress=progress,
                    current_stage=current_stage,
                    user=cost_attribution.get("user_id", ""),
                    department=cost_attribution.get("business_unit", ""),
                    created_at=job.get("submitted_at"),
                    updated_at=job.get("completed_at") or job.get("submitted_at"),
                    completed_at=job.get("completed_at"),
                    download_url=download_url,
                    error_message=self._job_error_message(job),
                    filename=source_document.get("original_filename"),
                    source_language=translation_config.get("source_language")
                    or source_document.get("source_language"),
                    target_language=translation_config.get("target_language"),
                )
            )

        # Get total count (simplified - in production use a separate count query)
        total = len(job_responses)

        return JobListResponse(
            jobs=job_responses, total=total, limit=limit, offset=offset
        )

    async def cancel_job(
        self, job_id: str, request: JobCancelRequest, user_id: str
    ) -> None:
        """Cancel a translation job owned by `user_id`.

        Ownership is enforced here, mirroring `get_jobs_status`: a job
        belonging to another user raises `JobNotFoundError` rather than 403,
        so this endpoint cannot be used to probe which job IDs exist.
        """
        # Get current job status
        job_data = await self._get_job_data(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        owner = (job_data.get("cost_attribution") or {}).get("user_id")
        if owner != user_id:
            # Deliberately indistinguishable from "no such job".
            logger.warning(
                "Rejected cancel of job %s: requested by %s, owned by %s",
                job_id,
                user_id,
                owner,
            )
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

        await self.bigquery.patch_translation_job(
            job_id,
            {
                "status": "cancelled",
                "completed_at": datetime.now(UTC),
                "error_message": updates["error_message"],
            },
        )

        # Remove the queued Cloud Task *after* the status patch, so a Cloud
        # Tasks outage can never leave a job un-cancelled. Without this the
        # task still dispatches and the worker no-ops -- wasteful in general,
        # and on the high-priority queue it burns one of only a few reserved
        # dispatch slots. Best-effort by design: never fails the cancel.
        priority = (job_data.get("processing_options") or {}).get("priority")
        try:
            await asyncio.to_thread(
                self.cloud_tasks.delete_translate_task, job_id, priority
            )
        except Exception as exc:  # noqa: BLE001 - cancel must still succeed
            logger.warning(
                "Cloud Task cleanup failed for cancelled job %s: %s", job_id, exc
            )

        logger.info(f"Cancelled job {job_id}")

    async def get_download_url(self, job_id: str, file_type: str) -> DownloadResponse:
        """Generate a signed URL for downloading output files."""
        # Get job data
        job_data = await self._get_job_data(job_id)

        if not job_data:
            raise JobNotFoundError(job_id)

        if job_data["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is not completed yet",
            )

        # Get output URIs
        result_payload = job_data.get("result", {}) or {}
        if file_type != "mono":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Output file {file_type} not found",
            )
        gcs_uri = result_payload.get("output_gcs_uri")
        if not gcs_uri:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Output file mono not found",
            )

        # Generate signed URL
        download_url = await self.storage.generate_signed_url(
            blob_path=gcs_uri,
            expires_in=3600,  # 1 hour
        )

        # Get file info
        file_info = await self.storage.get_file_info(gcs_uri)

        # Preserve the actual output extension (.pdf / .docx / .txt) instead
        # of assuming .pdf -- the DOCX and TXT pipelines both produce
        # non-PDF outputs.
        output_ext = Path(gcs_uri).suffix or ".pdf"

        return DownloadResponse(
            download_url=download_url,
            expires_in=3600,
            filename=f"{job_id}_{file_type}{output_ext}",
            file_size=file_info["size"],
        )

    async def stream_job_progress(
        self, job_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream job progress updates."""
        # Get initial state
        last_update = None

        while True:
            job_data = await self._get_job_data(job_id)

            if not job_data:
                yield {"type": "error", "message": "Job not found"}
                break

            # Check if job is finished
            if job_data["status"] in (
                "completed",
                "human_review_required",
                "failed",
                "cancelled",
            ):
                prog, _ = self._progress_and_stage(str(job_data["status"]))
                yield {
                    "type": "complete",
                    "status": job_data["status"],
                    "progress": prog,
                    "error_message": self._job_error_message(job_data),
                }
                break

            # Check for updates
            progress, current_stage = self._progress_and_stage(str(job_data["status"]))
            current_update = job_data.get("completed_at") or job_data.get(
                "submitted_at"
            )
            if last_update != current_update:
                yield {
                    "type": "progress",
                    "status": job_data["status"],
                    "progress": progress,
                    "current_stage": current_stage,
                }
                last_update = current_update

            # Wait before next check
            await asyncio.sleep(1)
