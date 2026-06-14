"""Functional tests: GCS write failure → exponential-backoff retry → failed job status.

Expected behaviour (from spec):
  - Transient GoogleAPIError retried with exponential backoff (min 10s, max 300s, max 5 attempts)
  - Retry succeeds if GCS recovers before 5 attempts are exhausted
  - After 5 failed attempts: StorageError raised, job marked failed in translation_jobs,
    logger.critical alert fired
  - Non-retryable errors (403 Forbidden, ValueError, FileNotFoundError) are not retried
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from google.api_core.exceptions import Aborted
from google.api_core.exceptions import DeadlineExceeded
from google.api_core.exceptions import Forbidden
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import NotFound
from google.api_core.exceptions import ServiceUnavailable
from google.api_core.exceptions import TooManyRequests
from src.config.constants import settings
from src.config.retry import is_retryable_gcs_exception
from src.repository.repository_exception import StorageError
from src.repository.storage_repository import FileType
from src.repository.storage_repository import StorageRepository
from tests.async_test_utils import patch_asyncio_sleep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo() -> tuple[StorageRepository, MagicMock]:
    """Return (repo, mock_blob) wired to a fake GCS client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    repo = StorageRepository(client=mock_client, bucket_name="test-bucket")
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    return repo, mock_blob


def _job_data(job_id: str = "test-job-123") -> dict:
    return {
        "source_document": {
            "gcs_uri": f"gs://bucket/input/{job_id}/file.pdf",
            "original_filename": "input.pdf",
        },
        "translation_config": {
            "source_language": "en",
            "target_language": "es",
            "domain": "general",
        },
        "cost_attribution": {"user_id": "user-1"},
    }


# ---------------------------------------------------------------------------
# Unit: is_retryable_gcs_exception predicate
# ---------------------------------------------------------------------------


class TestIsRetryableGcsException:
    def test_service_unavailable_is_retryable(self):
        assert is_retryable_gcs_exception(ServiceUnavailable("down"))

    def test_too_many_requests_is_retryable(self):
        assert is_retryable_gcs_exception(TooManyRequests("rate limit"))

    def test_internal_server_error_is_retryable(self):
        assert is_retryable_gcs_exception(InternalServerError("500"))

    def test_deadline_exceeded_is_retryable(self):
        assert is_retryable_gcs_exception(DeadlineExceeded("timeout"))

    def test_aborted_is_retryable(self):
        assert is_retryable_gcs_exception(Aborted("aborted"))

    def test_connection_error_is_retryable(self):
        assert is_retryable_gcs_exception(ConnectionError("reset"))

    def test_timeout_error_is_retryable(self):
        assert is_retryable_gcs_exception(TimeoutError("timed out"))

    def test_os_error_is_retryable(self):
        assert is_retryable_gcs_exception(OSError("broken pipe"))

    def test_timeout_message_substring_is_retryable(self):
        assert is_retryable_gcs_exception(Exception("connection timeout occurred"))

    def test_service_unavailable_message_is_retryable(self):
        assert is_retryable_gcs_exception(Exception("service unavailable temporarily"))

    def test_not_found_is_not_retryable(self):
        assert not is_retryable_gcs_exception(NotFound("404"))

    def test_forbidden_is_not_retryable(self):
        assert not is_retryable_gcs_exception(Forbidden("403"))

    def test_value_error_is_not_retryable(self):
        assert not is_retryable_gcs_exception(ValueError("bad value"))

    def test_file_not_found_is_not_retryable(self):
        assert not is_retryable_gcs_exception(FileNotFoundError("no file"))

    def test_generic_exception_is_not_retryable(self):
        assert not is_retryable_gcs_exception(Exception("some random error"))


# ---------------------------------------------------------------------------
# Upload succeeds on first attempt — no retry needed
# ---------------------------------------------------------------------------


class TestGcsUploadFirstAttemptSuccess:
    @pytest.mark.asyncio
    async def test_upload_bytes_succeeds_without_retry(self):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_string.return_value = None

        with patch_asyncio_sleep():
            result = await repo.upload_file(b"content", "path/file.pdf")

        assert result == "gs://test-bucket/path/file.pdf"
        mock_blob.upload_from_string.assert_called_once_with(b"content")

    @pytest.mark.asyncio
    async def test_upload_bytes_with_content_type_succeeds(self):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_string.return_value = None

        with patch_asyncio_sleep():
            result = await repo.upload_file(
                b"content", "path/file.pdf", file_type=FileType.PDF
            )

        assert result == "gs://test-bucket/path/file.pdf"
        mock_blob.upload_from_string.assert_called_once_with(
            b"content", content_type="application/pdf"
        )

    @pytest.mark.asyncio
    async def test_upload_path_succeeds_without_retry(self, tmp_path):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_filename.return_value = None
        src = tmp_path / "file.pdf"
        src.write_bytes(b"pdf data")

        with patch_asyncio_sleep():
            result = await repo.upload_file(src, "path/file.pdf")

        assert result == "gs://test-bucket/path/file.pdf"
        mock_blob.upload_from_filename.assert_called_once_with(str(src))


# ---------------------------------------------------------------------------
# Upload retries on transient failures and eventually succeeds
# ---------------------------------------------------------------------------


class TestGcsUploadRetryOnTransientError:
    @pytest.mark.asyncio
    async def test_retries_twice_then_succeeds(self):
        repo, mock_blob = _make_repo()

        attempt = 0

        def flaky(*_args, **_kwargs):
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ServiceUnavailable("GCS down")

        mock_blob.upload_from_string.side_effect = flaky

        with patch_asyncio_sleep():
            result = await repo.upload_file(b"data", "out/file.pdf")

        assert result == "gs://test-bucket/out/file.pdf"
        assert attempt == 3

    @pytest.mark.asyncio
    async def test_retries_on_too_many_requests(self):
        repo, mock_blob = _make_repo()

        call_count = 0

        def rate_limited(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TooManyRequests("429")

        mock_blob.upload_from_string.side_effect = rate_limited

        with patch_asyncio_sleep():
            await repo.upload_file(b"data", "out/file.pdf")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_deadline_exceeded(self):
        repo, mock_blob = _make_repo()

        call_count = 0

        def deadline_fail(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise DeadlineExceeded("deadline")

        mock_blob.upload_from_string.side_effect = deadline_fail

        with patch_asyncio_sleep():
            await repo.upload_file(b"data", "out/file.pdf")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_sleep_called_between_retries(self):
        repo, mock_blob = _make_repo()

        attempt = 0

        def flaky(*_args, **_kwargs):
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ServiceUnavailable("down")

        mock_blob.upload_from_string.side_effect = flaky
        sleep_calls = []

        async def record_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("asyncio.sleep", side_effect=record_sleep):
            await repo.upload_file(b"data", "out/file.pdf")

        assert len(sleep_calls) == 2, "Sleep called once per retry gap"
        for sleep_duration in sleep_calls:
            assert sleep_duration >= settings.GCS_RETRY_MIN_SECONDS
            assert sleep_duration <= settings.GCS_RETRY_MAX_SECONDS


# ---------------------------------------------------------------------------
# Upload exhausts all attempts — StorageError raised
# ---------------------------------------------------------------------------


class TestGcsUploadExhaustsAllAttempts:
    @pytest.mark.asyncio
    async def test_raises_storage_error_after_max_attempts(self):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_string.side_effect = ServiceUnavailable("GCS down")

        with patch_asyncio_sleep():
            with pytest.raises(StorageError, match="Failed to upload file"):
                await repo.upload_file(b"data", "out/file.pdf")

    @pytest.mark.asyncio
    async def test_upload_attempted_exactly_max_attempts_times(self):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_string.side_effect = ServiceUnavailable("GCS down")

        with patch_asyncio_sleep():
            with pytest.raises(StorageError):
                await repo.upload_file(b"data", "out/file.pdf")

        assert (
            mock_blob.upload_from_string.call_count == settings.GCS_RETRY_MAX_ATTEMPTS
        )

    @pytest.mark.asyncio
    async def test_storage_error_contains_blob_path(self):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_string.side_effect = ServiceUnavailable("GCS down")

        with patch_asyncio_sleep():
            with pytest.raises(StorageError) as exc_info:
                await repo.upload_file(b"data", "jobs/123/output.pdf")

        assert exc_info.value.path == "jobs/123/output.pdf"
        assert exc_info.value.operation == "upload"


# ---------------------------------------------------------------------------
# Non-retryable errors — no retry, immediate failure
# ---------------------------------------------------------------------------


class TestGcsUploadNonRetryableErrors:
    @pytest.mark.asyncio
    async def test_forbidden_not_retried(self):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_string.side_effect = Forbidden("no access")

        with patch_asyncio_sleep():
            with pytest.raises(StorageError):
                await repo.upload_file(b"data", "out/file.pdf")

        mock_blob.upload_from_string.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_found_not_retried(self):
        repo, mock_blob = _make_repo()
        mock_blob.upload_from_string.side_effect = NotFound("bucket missing")

        with patch_asyncio_sleep():
            with pytest.raises(StorageError):
                await repo.upload_file(b"data", "out/file.pdf")

        mock_blob.upload_from_string.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_not_found_not_retried(self, tmp_path):
        repo, mock_blob = _make_repo()
        missing_path = tmp_path / "nonexistent.pdf"

        with patch_asyncio_sleep():
            with pytest.raises(FileNotFoundError):
                await repo.upload_file(missing_path, "out/file.pdf")

        mock_blob.upload_from_filename.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline orchestrator: failed job status on persistent GCS failure
# ---------------------------------------------------------------------------


class TestPipelineGcsFailureOnJobsTable:
    """Persistent StorageError in pipeline → critical alert, failed translation_jobs row."""

    @pytest.fixture
    def mock_bigquery(self):
        bq = AsyncMock()
        bq.patch_translation_job = AsyncMock()
        bq.get_translation_job = AsyncMock(return_value={"job_id": "test-job-123"})
        return bq

    @pytest.fixture
    def mock_storage(self):
        return AsyncMock()

    @pytest.fixture
    def orchestrator(self, mock_bigquery, mock_storage):
        from src.api.services.pipeline_orchestrator import PipelineOrchestrator

        return PipelineOrchestrator(bigquery=mock_bigquery, storage=mock_storage)

    @pytest.mark.asyncio
    async def test_error_message_written_on_storage_error(
        self, orchestrator, mock_bigquery, mock_storage
    ):
        job_id = "test-job-123"
        storage_error = StorageError(
            "GCS write failed after 5 attempts",
            operation="upload",
            path="output/test-job-123/file.pdf",
        )
        mock_storage.download_file.side_effect = storage_error

        pipeline_span = MagicMock()
        pipeline_span.set_status = MagicMock()
        pipeline_span.record_exception = MagicMock()

        await orchestrator._execute_pipeline(job_id, _job_data(job_id), pipeline_span)

        patch_calls = mock_bigquery.patch_translation_job.call_args_list
        failed_calls = [
            c
            for c in patch_calls
            if c[0][1].get("status") == "failed"
            and "GCS write failed after 5 attempts" in c[0][1].get("error_message", "")
        ]
        assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_logger_critical_fired_on_storage_error(
        self, orchestrator, mock_bigquery, mock_storage
    ):
        job_id = "test-job-789"
        storage_error = StorageError(
            "GCS persistent failure", operation="upload", path="output/file.pdf"
        )
        mock_storage.download_file.side_effect = storage_error

        pipeline_span = MagicMock()
        with patch("src.api.services.pipeline_orchestrator.logger") as mock_logger:
            await orchestrator._execute_pipeline(
                job_id, _job_data(job_id), pipeline_span
            )

        mock_logger.critical.assert_called_once()
        critical_message = " ".join(str(a) for a in mock_logger.critical.call_args[0])
        assert job_id in critical_message
        assert "retry attempts" in critical_message.lower()

    @pytest.mark.asyncio
    async def test_job_marked_failed_on_storage_error(
        self, orchestrator, mock_bigquery, mock_storage
    ):
        job_id = "test-job-fail"
        storage_error = StorageError(
            "Upload failed", operation="upload", path="output/file.pdf"
        )
        mock_storage.download_file.side_effect = storage_error

        pipeline_span = MagicMock()
        await orchestrator._execute_pipeline(job_id, _job_data(job_id), pipeline_span)

        patch_calls = mock_bigquery.patch_translation_job.call_args_list
        status_calls = [c for c in patch_calls if c[0][1].get("status") == "failed"]
        assert len(status_calls) >= 1, "Job must be marked failed"

    @pytest.mark.asyncio
    async def test_non_storage_error_logs_error_not_critical(
        self, orchestrator, mock_bigquery, mock_storage
    ):
        job_id = "test-job-err"
        mock_storage.download_file.side_effect = RuntimeError("pipeline crash")

        pipeline_span = MagicMock()
        with patch("src.api.services.pipeline_orchestrator.logger") as mock_logger:
            await orchestrator._execute_pipeline(
                job_id, _job_data(job_id), pipeline_span
            )

        mock_logger.error.assert_called()
        mock_logger.critical.assert_not_called()
