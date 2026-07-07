"""Output file lifecycle service.

Translated outputs are temporary: they remain downloadable in GCS until a
configured retention window (``OUTPUT_FILE_TTL_SECONDS``) elapses, after which
a scheduled cleanup deletes them. Lifecycle metadata lives on the Firestore
job document under the ``file_lifecycle`` key.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

from api.repository.api_storage_repository import APIStorageRepository
from config.constants import settings
from config.logging_config import logger
from repository.firestore_repository import FirestoreRepository

STATE_AVAILABLE = "available"
STATE_DELETE_PENDING = "delete_pending"
STATE_DELETED = "deleted"


class FileLifecycleService:
    """Manages the retention window and cleanup of translated output files."""

    def __init__(
        self,
        firestore: FirestoreRepository | None = None,
        storage: APIStorageRepository | None = None,
    ):
        self.firestore = firestore or FirestoreRepository()
        self.storage = storage or APIStorageRepository()

    @staticmethod
    def build_lifecycle(available_at: datetime) -> dict[str, Any]:
        """Build a fresh lifecycle record for a newly available output file."""
        return {
            "storage_state": STATE_AVAILABLE,
            "available_at": available_at,
            "expires_at": available_at
            + timedelta(seconds=settings.OUTPUT_FILE_TTL_SECONDS),
            "deleted_at": None,
            "url_issue_count": 0,
            "last_url_issued_at": None,
            "delete_attempts": 0,
            "last_delete_error": None,
        }

    async def ensure_lifecycle(
        self, job_id: str, job_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the job's lifecycle record, backfilling jobs that predate it.

        Jobs completed before lifecycle tracking existed get a record anchored
        to their completion time, so their retention window is not restarted.
        """
        lifecycle = job_data.get("file_lifecycle")
        if lifecycle:
            return lifecycle

        timestamps = job_data.get("timestamps", {}) or {}
        available_at = (
            timestamps.get("completed_at")
            or job_data.get("completed_at")
            or datetime.now(UTC)
        )
        lifecycle = self.build_lifecycle(available_at)
        job_data["file_lifecycle"] = lifecycle
        try:
            await self.firestore.update_job(job_id, {"file_lifecycle": lifecycle})
        except Exception:
            logger.warning(f"Could not persist file lifecycle for job {job_id}")
        return lifecycle

    @staticmethod
    def remaining_ttl_seconds(
        lifecycle: dict[str, Any], now: datetime | None = None
    ) -> int:
        """Seconds until the output file expires (0 if already expired)."""
        expires_at = lifecycle.get("expires_at")
        if expires_at is None:
            return 0
        now = now or datetime.now(UTC)
        return max(0, int((expires_at - now).total_seconds()))

    def is_available(
        self, lifecycle: dict[str, Any], now: datetime | None = None
    ) -> bool:
        """Whether the output file can still be downloaded."""
        return (
            lifecycle.get("storage_state") == STATE_AVAILABLE
            and self.remaining_ttl_seconds(lifecycle, now) > 0
        )

    async def record_url_issued(self, job_id: str, lifecycle: dict[str, Any]) -> None:
        """Track signed URL issuance (the app tracks issuance, not downloads)."""
        now = datetime.now(UTC)
        lifecycle["url_issue_count"] = int(lifecycle.get("url_issue_count") or 0) + 1
        lifecycle["last_url_issued_at"] = now
        try:
            await self.firestore.update_job(
                job_id,
                {
                    "file_lifecycle.url_issue_count": lifecycle["url_issue_count"],
                    "file_lifecycle.last_url_issued_at": now,
                },
            )
        except Exception:
            logger.warning(f"Could not record URL issuance for job {job_id}")

    async def cleanup_expired_outputs(self) -> dict[str, int]:
        """Delete expired output blobs and leftover files of dead jobs.

        Deletion is prefix-based and therefore idempotent: an already-missing
        object simply yields zero deletions and the job is still marked
        deleted. Failed deletions move the record to ``delete_pending`` so the
        next scheduled run retries them.
        """
        now = datetime.now(UTC)
        summary = {
            "expired_outputs_scanned": 0,
            "terminal_jobs_scanned": 0,
            "files_deleted": 0,
            "jobs_cleaned": 0,
            "failures": 0,
        }

        expired = await self.firestore.list_jobs_with_expired_outputs(
            now, limit=settings.CLEANUP_BATCH_SIZE
        )
        summary["expired_outputs_scanned"] = len(expired)
        for job in expired:
            await self._cleanup_job(job, delete_all=False, now=now, summary=summary)

        terminal = await self.firestore.list_expired_terminal_jobs(
            now, limit=settings.CLEANUP_BATCH_SIZE
        )
        summary["terminal_jobs_scanned"] = len(terminal)
        for job in terminal:
            lifecycle = job.get("file_lifecycle") or {}
            if lifecycle.get("storage_state") == STATE_DELETED:
                continue
            await self._cleanup_job(job, delete_all=True, now=now, summary=summary)

        logger.info(f"Output file cleanup finished: {summary}")
        return summary

    async def _cleanup_job(
        self,
        job: dict[str, Any],
        *,
        delete_all: bool,
        now: datetime,
        summary: dict[str, int],
    ) -> None:
        """Delete a single job's blobs and update its lifecycle state."""
        job_id = job["job_id"]
        lifecycle = job.get("file_lifecycle") or {}

        try:
            if delete_all:
                deleted = await self.storage.delete_job_files(job_id)
            else:
                output_prefix = (
                    self.storage.build_job_path(
                        job_id=job_id, folder=settings.GCS_OUTPUT_FOLDER
                    )
                    + "/"
                )
                deleted = await self.storage.delete_files(output_prefix)
        except Exception as e:
            summary["failures"] += 1
            attempts = int(lifecycle.get("delete_attempts") or 0) + 1
            logger.error(f"Cleanup failed for job {job_id} (attempt {attempts}): {e}")
            await self._safe_update(
                job_id,
                {
                    "file_lifecycle.storage_state": STATE_DELETE_PENDING,
                    "file_lifecycle.delete_attempts": attempts,
                    "file_lifecycle.last_delete_error": str(e),
                },
            )
            return

        summary["files_deleted"] += deleted
        summary["jobs_cleaned"] += 1
        await self._safe_update(
            job_id,
            {
                "file_lifecycle.storage_state": STATE_DELETED,
                "file_lifecycle.deleted_at": now,
                "file_lifecycle.last_delete_error": None,
            },
        )

    async def _safe_update(self, job_id: str, updates: dict[str, Any]) -> None:
        try:
            await self.firestore.update_job(job_id, updates)
        except Exception:
            logger.warning(f"Could not update lifecycle state for job {job_id}")
