"""
Unit tests for api/services/file_service.py — FileService.

All storage calls are mocked; no real GCP connection is made.
"""

from unittest.mock import AsyncMock

import pytest

from api.exceptions import FileProcessingError
from api.services.file_service import FileService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(storage=None) -> FileService:
    st = storage or AsyncMock()
    svc = FileService.__new__(FileService)
    svc.storage = st
    return svc


# ---------------------------------------------------------------------------
# cleanup_job_files
# ---------------------------------------------------------------------------


class TestCleanupJobFiles:
    async def test_calls_delete_job_files(self):
        storage = AsyncMock()
        svc = _make_service(storage=storage)

        await svc.cleanup_job_files("job-1")
        storage.delete_job_files.assert_called_once_with("job-1")

    async def test_swallows_exception_no_raise(self):
        storage = AsyncMock()
        storage.delete_job_files.side_effect = Exception("GCS down")
        svc = _make_service(storage=storage)

        # Should NOT raise
        await svc.cleanup_job_files("job-1")

    async def test_succeeds_silently_on_success(self):
        storage = AsyncMock()
        storage.delete_job_files.return_value = 3
        svc = _make_service(storage=storage)

        # Should complete without error
        await svc.cleanup_job_files("job-1")


# ---------------------------------------------------------------------------
# get_file_metadata
# ---------------------------------------------------------------------------


class TestGetFileMetadata:
    async def test_returns_metadata_dict(self):
        storage = AsyncMock()
        storage.get_file_info.return_value = {"size": 4096, "name": "file.pdf"}
        svc = _make_service(storage=storage)

        result = await svc.get_file_metadata("gs://bucket/file.pdf")
        assert result["size"] == 4096

    async def test_storage_called_with_uri(self):
        storage = AsyncMock()
        storage.get_file_info.return_value = {"name": "f.pdf"}
        svc = _make_service(storage=storage)

        await svc.get_file_metadata("gs://bucket/f.pdf")
        storage.get_file_info.assert_called_once_with("gs://bucket/f.pdf")

    async def test_raises_file_processing_error_on_failure(self):
        storage = AsyncMock()
        storage.get_file_info.side_effect = Exception("Not found")
        svc = _make_service(storage=storage)

        with pytest.raises(FileProcessingError, match="Failed to get file metadata"):
            await svc.get_file_metadata("gs://bucket/missing.pdf")


# ---------------------------------------------------------------------------
# verify_file_integrity
# ---------------------------------------------------------------------------


class TestVerifyFileIntegrity:
    async def test_returns_true_when_file_exists(self):
        storage = AsyncMock()
        storage.get_file_info.return_value = {"name": "file.pdf", "size": 1024}
        svc = _make_service(storage=storage)

        result = await svc.verify_file_integrity("gs://bucket/file.pdf", "abc123")
        assert result is True

    async def test_returns_false_on_exception(self):
        storage = AsyncMock()
        storage.get_file_info.side_effect = Exception("error")
        svc = _make_service(storage=storage)

        result = await svc.verify_file_integrity("gs://bucket/file.pdf", "abc123")
        assert result is False

    async def test_calls_storage_with_uri(self):
        storage = AsyncMock()
        storage.get_file_info.return_value = {}
        svc = _make_service(storage=storage)

        await svc.verify_file_integrity("gs://bucket/f.pdf", "checksum")
        storage.get_file_info.assert_called_once_with("gs://bucket/f.pdf")
