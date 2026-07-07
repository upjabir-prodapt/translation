"""
Unit tests for api/services/job_service.py — JobService.

All Firestore and Storage clients are mocked; no real GCP calls are made.
"""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from conftest import make_job_doc
from fastapi import HTTPException

from api.exceptions import JobAlreadyCompletedError
from api.exceptions import JobNotFoundError
from api.schemas.requests import JobCancelRequest
from api.services.job_service import JobService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(firestore=None, storage=None) -> JobService:
    """Build a JobService with provided mocks (avoids real GCP init)."""
    fs = firestore or AsyncMock()
    st = storage or AsyncMock()
    return JobService(firestore=fs, storage=st)


# ---------------------------------------------------------------------------
# get_job_status
# ---------------------------------------------------------------------------


class TestGetJobStatus:
    async def test_returns_status_response_for_existing_job(self, sample_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        result = await service.get_job_status(sample_job_data["job_id"])

        assert result.job_id == sample_job_data["job_id"]
        assert result.status == "queued"

    async def test_raises_job_not_found_for_missing_job(self):
        fs = AsyncMock()
        fs.get_job.return_value = None
        service = _make_service(firestore=fs)

        with pytest.raises(JobNotFoundError):
            await service.get_job_status("nonexistent-job")

    async def test_progress_included(self, sample_job_data):
        sample_job_data["progress"] = 0.45
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        result = await service.get_job_status(sample_job_data["job_id"])
        assert result.progress == 0.45

    async def test_current_stage_included(self, sample_job_data):
        sample_job_data["current_stage"] = "translating"
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        result = await service.get_job_status(sample_job_data["job_id"])
        assert result.current_stage == "translating"

    async def test_cost_attribution_user_and_department(self, sample_job_data):
        sample_job_data["cost_attribution"] = {
            "user_id": "alice@example.com",
            "business_unit": "engineering",
        }
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        result = await service.get_job_status(sample_job_data["job_id"])
        assert result.user == "alice@example.com"
        assert result.department == "engineering"


# ---------------------------------------------------------------------------
# get_translation_status
# ---------------------------------------------------------------------------


class TestGetTranslationStatus:
    async def test_queued_job_has_no_result(self, sample_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        result = await service.get_translation_status(sample_job_data["job_id"])
        assert result.status == "queued"
        assert result.result is None

    async def test_raises_job_not_found(self):
        fs = AsyncMock()
        fs.get_job.return_value = None
        service = _make_service(firestore=fs)

        with pytest.raises(JobNotFoundError):
            await service.get_translation_status("no-such-job")

    async def test_completed_job_has_result(self, completed_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        storage = AsyncMock()
        storage.generate_signed_url.return_value = "https://signed.url/doc_es.pdf"
        service = _make_service(firestore=fs, storage=storage)

        result = await service.get_translation_status(completed_job_data["job_id"])
        assert result.status == "completed"
        assert result.result is not None
        assert result.result.translated_document is not None

    async def test_completed_job_download_url_generated(self, completed_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        storage = AsyncMock()
        storage.generate_signed_url.return_value = "https://signed.url/output.pdf"
        service = _make_service(firestore=fs, storage=storage)

        result = await service.get_translation_status(completed_job_data["job_id"])
        assert (
            result.result.translated_document.download_url
            == "https://signed.url/output.pdf"
        )

    async def test_signed_url_failure_does_not_raise(self, completed_job_data):
        """If signed URL generation fails, the result still returns (download_url=None)."""
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        storage = AsyncMock()
        storage.generate_signed_url.side_effect = Exception("GCS error")
        service = _make_service(firestore=fs, storage=storage)

        result = await service.get_translation_status(completed_job_data["job_id"])
        assert result.result.translated_document.download_url is None

    async def test_completed_job_metadata(self, completed_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        storage = AsyncMock()
        storage.generate_signed_url.return_value = "https://url"
        service = _make_service(firestore=fs, storage=storage)

        result = await service.get_translation_status(completed_job_data["job_id"])
        meta = result.result.metadata
        assert meta is not None
        assert meta.model_used == "gpt-4o-mini"
        assert meta.quality_score == 0.92

    async def test_output_filename_from_source_doc(self, completed_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        storage = AsyncMock()
        storage.generate_signed_url.return_value = "https://url"
        service = _make_service(firestore=fs, storage=storage)

        result = await service.get_translation_status(completed_job_data["job_id"])
        assert result.result.translated_document.filename == "doc_es.pdf"


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    async def test_returns_job_list_response(self, sample_job_data):
        fs = AsyncMock()
        fs.list_jobs.return_value = [sample_job_data]
        service = _make_service(firestore=fs)

        result = await service.list_jobs()
        assert len(result.jobs) == 1
        assert result.total == 1

    async def test_empty_list(self):
        fs = AsyncMock()
        fs.list_jobs.return_value = []
        service = _make_service(firestore=fs)

        result = await service.list_jobs()
        assert result.jobs == []
        assert result.total == 0

    async def test_pagination_params_passed(self, sample_job_data):
        fs = AsyncMock()
        fs.list_jobs.return_value = [sample_job_data]
        service = _make_service(firestore=fs)

        await service.list_jobs(status="queued", limit=25, offset=10)
        fs.list_jobs.assert_called_once_with(status="queued", limit=25, offset=10)

    async def test_multiple_jobs_returned(self):
        jobs = [make_job_doc(status="queued") for _ in range(5)]
        fs = AsyncMock()
        fs.list_jobs.return_value = jobs
        service = _make_service(firestore=fs)

        result = await service.list_jobs(limit=5)
        assert result.total == 5
        assert result.limit == 5

    async def test_offset_returned_in_response(self):
        fs = AsyncMock()
        fs.list_jobs.return_value = []
        service = _make_service(firestore=fs)

        result = await service.list_jobs(offset=20)
        assert result.offset == 20


# ---------------------------------------------------------------------------
# cancel_job
# ---------------------------------------------------------------------------


class TestCancelJob:
    async def test_cancels_queued_job(self, sample_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        fs.update_job.return_value = True
        service = _make_service(firestore=fs)

        await service.cancel_job(sample_job_data["job_id"], JobCancelRequest())
        fs.update_job.assert_called_once()
        call_args = fs.update_job.call_args
        assert call_args[0][1]["status"] == "cancelled"

    async def test_cancels_processing_job(self, sample_job_data):
        sample_job_data["status"] = "processing"
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        fs.update_job.return_value = True
        service = _make_service(firestore=fs)

        await service.cancel_job(sample_job_data["job_id"], JobCancelRequest())
        fs.update_job.assert_called_once()

    async def test_cancel_with_reason(self, sample_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        await service.cancel_job(
            sample_job_data["job_id"],
            JobCancelRequest(reason="No longer needed"),
        )
        updates = fs.update_job.call_args[0][1]
        assert "No longer needed" in updates["error_message"]

    async def test_cancel_without_reason(self, sample_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        await service.cancel_job(sample_job_data["job_id"], JobCancelRequest())
        updates = fs.update_job.call_args[0][1]
        assert "Cancelled by user" in updates["error_message"]

    async def test_raises_not_found(self):
        fs = AsyncMock()
        fs.get_job.return_value = None
        service = _make_service(firestore=fs)

        with pytest.raises(JobNotFoundError):
            await service.cancel_job("ghost-job", JobCancelRequest())

    @pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
    async def test_raises_already_completed_for_terminal_status(
        self, terminal_status, sample_job_data
    ):
        sample_job_data["status"] = terminal_status
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data
        service = _make_service(firestore=fs)

        with pytest.raises(JobAlreadyCompletedError):
            await service.cancel_job(sample_job_data["job_id"], JobCancelRequest())


# ---------------------------------------------------------------------------
# get_download_url
# ---------------------------------------------------------------------------


class TestGetDownloadUrl:
    async def test_raises_not_found_for_missing_job(self):
        fs = AsyncMock()
        fs.get_job.return_value = None
        service = _make_service(firestore=fs)

        with pytest.raises(JobNotFoundError):
            await service.get_download_url("ghost", "mono")

    async def test_raises_400_if_job_not_completed(self, sample_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = sample_job_data  # status=queued
        service = _make_service(firestore=fs)

        with pytest.raises(HTTPException) as exc_info:
            await service.get_download_url(sample_job_data["job_id"], "mono")
        assert exc_info.value.status_code == 400

    async def test_raises_404_if_file_type_missing(self, completed_job_data):
        completed_job_data["output_gs_uris"] = {}  # no 'mono' key
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        service = _make_service(firestore=fs)

        with pytest.raises(HTTPException) as exc_info:
            await service.get_download_url(completed_job_data["job_id"], "mono")
        assert exc_info.value.status_code == 404

    async def test_returns_download_response(self, completed_job_data):
        jid = completed_job_data["job_id"]
        completed_job_data["output_gs_uris"] = {
            "mono": f"gs://bucket/translation/{jid}/output/mono.pdf"
        }
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        storage = AsyncMock()
        storage.generate_signed_url.return_value = "https://signed.url/mono.pdf"
        storage.get_file_info.return_value = {"size": 204800}
        service = _make_service(firestore=fs, storage=storage)

        result = await service.get_download_url(jid, "mono")
        assert result.download_url == "https://signed.url/mono.pdf"
        assert result.expires_in == 3600
        assert result.file_size == 204800


# ---------------------------------------------------------------------------
# stream_job_progress
# ---------------------------------------------------------------------------


class TestStreamJobProgress:
    async def test_yields_error_for_missing_job(self):
        fs = AsyncMock()
        fs.get_job.return_value = None
        service = _make_service(firestore=fs)

        events = []
        async for event in service.stream_job_progress("ghost-job"):
            events.append(event)
            break  # stream terminates on error

        assert events[0]["type"] == "error"

    async def test_yields_complete_for_finished_job(self, completed_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        service = _make_service(firestore=fs)

        events = []
        async for event in service.stream_job_progress(completed_job_data["job_id"]):
            events.append(event)

        assert any(e["type"] == "complete" for e in events)

    async def test_complete_event_has_status(self, completed_job_data):
        fs = AsyncMock()
        fs.get_job.return_value = completed_job_data
        service = _make_service(firestore=fs)

        async for event in service.stream_job_progress(completed_job_data["job_id"]):
            if event["type"] == "complete":
                assert event["status"] == "completed"
                break

    async def test_yields_progress_for_in_progress_job(self, sample_job_data):
        """Covers the progress update branch (lines 262-273)."""
        in_progress = dict(sample_job_data)
        in_progress["status"] = "processing"
        in_progress["progress"] = 0.5
        in_progress["current_stage"] = "translating"
        in_progress["updated_at"] = datetime.now(UTC)

        completed = dict(sample_job_data)
        completed["status"] = "completed"
        completed["progress"] = 1.0
        completed["updated_at"] = datetime.now(UTC)

        fs = AsyncMock()
        fs.get_job.side_effect = [in_progress, completed]
        service = _make_service(firestore=fs)

        events = []
        with patch("api.services.job_service.asyncio.sleep", new=AsyncMock()):
            async for event in service.stream_job_progress(in_progress["job_id"]):
                events.append(event)

        types = [e["type"] for e in events]
        assert "progress" in types
        assert "complete" in types

    async def test_progress_event_has_stage(self, sample_job_data):
        in_progress = dict(sample_job_data)
        in_progress["status"] = "processing"
        in_progress["progress"] = 0.3
        in_progress["current_stage"] = "layout detection"
        in_progress["updated_at"] = datetime.now(UTC)

        completed = dict(sample_job_data)
        completed["status"] = "failed"
        completed["updated_at"] = datetime.now(UTC)

        fs = AsyncMock()
        fs.get_job.side_effect = [in_progress, completed]
        service = _make_service(firestore=fs)

        with patch("api.services.job_service.asyncio.sleep", new=AsyncMock()):
            async for event in service.stream_job_progress(in_progress["job_id"]):
                if event["type"] == "progress":
                    assert event["current_stage"] == "layout detection"
                    break

    async def test_no_duplicate_progress_when_update_unchanged(self, sample_job_data):
        """When updated_at does not change, no progress event is re-emitted."""
        ts = datetime.now(UTC)
        in_progress = dict(sample_job_data)
        in_progress["status"] = "processing"
        in_progress["updated_at"] = ts

        completed = dict(sample_job_data)
        completed["status"] = "completed"
        completed["updated_at"] = ts

        fs = AsyncMock()
        # First call: in progress with same ts; second call: same ts but completed
        fs.get_job.side_effect = [in_progress, in_progress, completed]
        service = _make_service(firestore=fs)

        events = []
        with patch("api.services.job_service.asyncio.sleep", new=AsyncMock()):
            async for event in service.stream_job_progress(in_progress["job_id"]):
                events.append(event)

        progress_events = [e for e in events if e["type"] == "progress"]
        # Only the first occurrence should emit progress, second loop skips (same ts)
        assert len(progress_events) == 1
