"""
Unit tests for api/services/file_lifecycle_service.py — FileLifecycleService.

All Firestore and Storage clients are mocked; no real GCP calls are made.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from api.services.file_lifecycle_service import (
    STATE_AVAILABLE,
    STATE_DELETE_PENDING,
    STATE_DELETED,
    FileLifecycleService,
)
from config.constants import settings
from conftest import make_job_doc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(firestore=None, storage=None) -> FileLifecycleService:
    fs = firestore or AsyncMock()
    st = storage or AsyncMock()
    return FileLifecycleService(firestore=fs, storage=st)


def make_lifecycle(
    *,
    state: str = STATE_AVAILABLE,
    expires_in_seconds: int = 3600,
    **overrides,
) -> dict:
    now = datetime.now(UTC)
    lifecycle = {
        "storage_state": state,
        "available_at": now - timedelta(hours=1),
        "expires_at": now + timedelta(seconds=expires_in_seconds),
        "deleted_at": None,
        "url_issue_count": 0,
        "last_url_issued_at": None,
        "delete_attempts": 0,
        "last_delete_error": None,
    }
    lifecycle.update(overrides)
    return lifecycle


# ---------------------------------------------------------------------------
# build_lifecycle / ensure_lifecycle
# ---------------------------------------------------------------------------


class TestEnsureLifecycle:
    def test_build_lifecycle_sets_ttl_window(self):
        available_at = datetime.now(UTC)
        lifecycle = FileLifecycleService.build_lifecycle(available_at)

        assert lifecycle["storage_state"] == STATE_AVAILABLE
        assert lifecycle["expires_at"] == available_at + timedelta(
            seconds=settings.OUTPUT_FILE_TTL_SECONDS
        )
        assert lifecycle["url_issue_count"] == 0
        assert lifecycle["delete_attempts"] == 0

    async def test_returns_existing_record_without_persisting(self):
        fs = AsyncMock()
        service = _make_service(firestore=fs)
        existing = make_lifecycle()
        job_data = make_job_doc(status="completed", file_lifecycle=existing)

        result = await service.ensure_lifecycle(job_data["job_id"], job_data)

        assert result is existing
        fs.update_job.assert_not_called()

    async def test_backfills_anchored_to_completed_at(self):
        """Jobs completed before lifecycle tracking keep their original window."""
        fs = AsyncMock()
        service = _make_service(firestore=fs)
        completed_at = datetime.now(UTC) - timedelta(hours=2)
        job_data = make_job_doc(
            status="completed",
            timestamps={"submitted_at": completed_at, "completed_at": completed_at},
        )

        result = await service.ensure_lifecycle(job_data["job_id"], job_data)

        assert result["available_at"] == completed_at
        assert result["expires_at"] == completed_at + timedelta(
            seconds=settings.OUTPUT_FILE_TTL_SECONDS
        )
        fs.update_job.assert_called_once_with(
            job_data["job_id"], {"file_lifecycle": result}
        )

    async def test_backfill_persist_failure_still_returns_record(self):
        fs = AsyncMock()
        fs.update_job.side_effect = Exception("firestore down")
        service = _make_service(firestore=fs)
        job_data = make_job_doc(status="completed")

        result = await service.ensure_lifecycle(job_data["job_id"], job_data)
        assert result["storage_state"] == STATE_AVAILABLE


# ---------------------------------------------------------------------------
# remaining_ttl_seconds / is_available
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_remaining_ttl_positive_before_expiry(self):
        lifecycle = make_lifecycle(expires_in_seconds=600)
        remaining = FileLifecycleService.remaining_ttl_seconds(lifecycle)
        assert 0 < remaining <= 600

    def test_remaining_ttl_zero_after_expiry(self):
        lifecycle = make_lifecycle(expires_in_seconds=-60)
        assert FileLifecycleService.remaining_ttl_seconds(lifecycle) == 0

    def test_remaining_ttl_zero_when_expiry_missing(self):
        assert FileLifecycleService.remaining_ttl_seconds({}) == 0

    def test_available_within_window(self):
        service = _make_service()
        assert service.is_available(make_lifecycle()) is True

    def test_not_available_after_expiry(self):
        service = _make_service()
        assert service.is_available(make_lifecycle(expires_in_seconds=-1)) is False

    @pytest.mark.parametrize("state", [STATE_DELETE_PENDING, STATE_DELETED])
    def test_not_available_when_not_in_available_state(self, state):
        service = _make_service()
        assert service.is_available(make_lifecycle(state=state)) is False


# ---------------------------------------------------------------------------
# record_url_issued
# ---------------------------------------------------------------------------


class TestRecordUrlIssued:
    async def test_increments_issue_count_and_persists(self):
        fs = AsyncMock()
        service = _make_service(firestore=fs)
        lifecycle = make_lifecycle(url_issue_count=2)

        await service.record_url_issued("job-1", lifecycle)

        assert lifecycle["url_issue_count"] == 3
        assert lifecycle["last_url_issued_at"] is not None
        updates = fs.update_job.call_args[0][1]
        assert updates["file_lifecycle.url_issue_count"] == 3

    async def test_persist_failure_does_not_raise(self):
        fs = AsyncMock()
        fs.update_job.side_effect = Exception("firestore down")
        service = _make_service(firestore=fs)

        await service.record_url_issued("job-1", make_lifecycle())


# ---------------------------------------------------------------------------
# cleanup_expired_outputs
# ---------------------------------------------------------------------------


class TestCleanupExpiredOutputs:
    async def test_deletes_expired_output_and_marks_deleted(self):
        job = make_job_doc(
            status="completed",
            file_lifecycle=make_lifecycle(expires_in_seconds=-60),
        )
        fs = AsyncMock()
        fs.list_jobs_with_expired_outputs.return_value = [job]
        fs.list_expired_terminal_jobs.return_value = []
        storage = AsyncMock()
        storage.build_job_path = (
            lambda job_id, folder: f"translation/{job_id}/{folder}"
        )
        storage.delete_files.return_value = 2
        service = _make_service(firestore=fs, storage=storage)

        summary = await service.cleanup_expired_outputs()

        storage.delete_files.assert_awaited_once_with(
            f"translation/{job['job_id']}/output/"
        )
        assert summary["expired_outputs_scanned"] == 1
        assert summary["files_deleted"] == 2
        assert summary["jobs_cleaned"] == 1
        assert summary["failures"] == 0
        updates = fs.update_job.call_args[0][1]
        assert updates["file_lifecycle.storage_state"] == STATE_DELETED
        assert updates["file_lifecycle.deleted_at"] is not None

    async def test_idempotent_when_objects_already_gone(self):
        """A missing object is not an error: zero deletions still marks deleted."""
        job = make_job_doc(
            status="completed",
            file_lifecycle=make_lifecycle(expires_in_seconds=-60),
        )
        fs = AsyncMock()
        fs.list_jobs_with_expired_outputs.return_value = [job]
        fs.list_expired_terminal_jobs.return_value = []
        storage = AsyncMock()
        storage.build_job_path = (
            lambda job_id, folder: f"translation/{job_id}/{folder}"
        )
        storage.delete_files.return_value = 0  # nothing left in GCS
        service = _make_service(firestore=fs, storage=storage)

        summary = await service.cleanup_expired_outputs()

        assert summary["jobs_cleaned"] == 1
        assert summary["failures"] == 0
        updates = fs.update_job.call_args[0][1]
        assert updates["file_lifecycle.storage_state"] == STATE_DELETED

    async def test_failed_deletion_marks_delete_pending(self):
        job = make_job_doc(
            status="completed",
            file_lifecycle=make_lifecycle(expires_in_seconds=-60, delete_attempts=1),
        )
        fs = AsyncMock()
        fs.list_jobs_with_expired_outputs.return_value = [job]
        fs.list_expired_terminal_jobs.return_value = []
        storage = AsyncMock()
        storage.build_job_path = (
            lambda job_id, folder: f"translation/{job_id}/{folder}"
        )
        storage.delete_files.side_effect = Exception("GCS unavailable")
        service = _make_service(firestore=fs, storage=storage)

        summary = await service.cleanup_expired_outputs()

        assert summary["failures"] == 1
        assert summary["jobs_cleaned"] == 0
        updates = fs.update_job.call_args[0][1]
        assert updates["file_lifecycle.storage_state"] == STATE_DELETE_PENDING
        assert updates["file_lifecycle.delete_attempts"] == 2
        assert "GCS unavailable" in updates["file_lifecycle.last_delete_error"]

    async def test_terminal_jobs_have_all_files_deleted(self):
        job = make_job_doc(status="failed")
        fs = AsyncMock()
        fs.list_jobs_with_expired_outputs.return_value = []
        fs.list_expired_terminal_jobs.return_value = [job]
        storage = AsyncMock()
        storage.delete_job_files.return_value = 3
        service = _make_service(firestore=fs, storage=storage)

        summary = await service.cleanup_expired_outputs()

        storage.delete_job_files.assert_awaited_once_with(job["job_id"])
        assert summary["terminal_jobs_scanned"] == 1
        assert summary["files_deleted"] == 3
        assert summary["jobs_cleaned"] == 1

    async def test_terminal_job_already_deleted_is_skipped(self):
        job = make_job_doc(
            status="cancelled",
            file_lifecycle=make_lifecycle(state=STATE_DELETED),
        )
        fs = AsyncMock()
        fs.list_jobs_with_expired_outputs.return_value = []
        fs.list_expired_terminal_jobs.return_value = [job]
        storage = AsyncMock()
        service = _make_service(firestore=fs, storage=storage)

        summary = await service.cleanup_expired_outputs()

        storage.delete_job_files.assert_not_called()
        assert summary["jobs_cleaned"] == 0

    async def test_empty_run_returns_zero_summary(self):
        fs = AsyncMock()
        fs.list_jobs_with_expired_outputs.return_value = []
        fs.list_expired_terminal_jobs.return_value = []
        service = _make_service(firestore=fs)

        summary = await service.cleanup_expired_outputs()

        assert summary == {
            "expired_outputs_scanned": 0,
            "terminal_jobs_scanned": 0,
            "files_deleted": 0,
            "jobs_cleaned": 0,
            "failures": 0,
        }
