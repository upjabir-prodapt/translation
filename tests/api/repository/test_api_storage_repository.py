"""
Unit tests for api/repository/api_storage_repository.py — APIStorageRepository.

All GCS calls are mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import GoogleAPIError

from api.repository.api_storage_repository import APIStorageRepository
from repository.storage_repository import StorageError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blob(name: str = "path/file.pdf") -> MagicMock:
    blob = MagicMock()
    blob.name = name
    blob.size = 4096
    blob.content_type = "application/pdf"
    blob.updated = None
    blob.time_created = None
    blob.md5_hash = "hash=="
    blob.metadata = {}
    return blob


def _make_repo(bucket=None, client=None) -> APIStorageRepository:
    mock_client = client or MagicMock()
    mock_bucket = bucket or MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return APIStorageRepository(client=mock_client, bucket_name="test-bucket")


# ---------------------------------------------------------------------------
# upload_input_pdf
# ---------------------------------------------------------------------------


class TestUploadInputPdf:
    async def test_returns_gcs_uri(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        uri = await repo.upload_input_pdf(b"pdf bytes", "doc.pdf", "job-1")
        assert uri.startswith("gs://test-bucket/")

    async def test_blob_path_contains_job_id(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.upload_input_pdf(b"data", "report.pdf", "job-abc")
        blob_path = bucket.blob.call_args[0][0]
        assert "job-abc" in blob_path

    async def test_blob_path_contains_filename(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.upload_input_pdf(b"data", "myfile.pdf", "job-1")
        blob_path = bucket.blob.call_args[0][0]
        assert "myfile.pdf" in blob_path

    async def test_storage_error_propagates(self):
        blob = _make_blob()
        blob.upload_from_string.side_effect = GoogleAPIError("quota exceeded")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError):
            await repo.upload_input_pdf(b"data", "doc.pdf", "job-1")


# ---------------------------------------------------------------------------
# upload_glossary
# ---------------------------------------------------------------------------


class TestUploadGlossary:
    async def test_returns_gcs_uri(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        uri = await repo.upload_glossary(b"term,definition", "glossary.csv", "job-2")
        assert uri.startswith("gs://")

    async def test_blob_path_contains_filename(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.upload_glossary(b"csv data", "terms.csv", "job-2")
        blob_path = bucket.blob.call_args[0][0]
        assert "terms.csv" in blob_path


# ---------------------------------------------------------------------------
# get_job_files
# ---------------------------------------------------------------------------


class TestGetJobFiles:
    async def test_returns_list_of_blobs(self):
        blobs = [_make_blob("f1.pdf"), _make_blob("f2.pdf")]
        bucket = MagicMock()
        bucket.list_blobs.return_value = blobs
        repo = _make_repo(bucket=bucket)

        result = await repo.get_job_files("job-1")
        assert len(result) == 2

    async def test_prefix_contains_job_id(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        await repo.get_job_files("job-xyz")
        call_kwargs = bucket.list_blobs.call_args[1]
        assert "job-xyz" in call_kwargs.get("prefix", "")


# ---------------------------------------------------------------------------
# delete_job_files
# ---------------------------------------------------------------------------


class TestDeleteJobFiles:
    async def test_returns_count(self):
        blobs = [_make_blob(f"f{i}.pdf") for i in range(3)]
        bucket = MagicMock()
        bucket.list_blobs.return_value = blobs
        bucket.delete_blobs.return_value = None
        repo = _make_repo(bucket=bucket)

        count = await repo.delete_job_files("job-1")
        assert count == 3

    async def test_returns_zero_for_empty_job(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        count = await repo.delete_job_files("empty-job")
        assert count == 0


# ---------------------------------------------------------------------------
# generate_download_url
# ---------------------------------------------------------------------------


class TestGenerateDownloadUrl:
    async def test_returns_signed_url(self):
        blob = _make_blob()
        blob.generate_signed_url.return_value = "https://signed.url/output.pdf"
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        url = await repo.generate_download_url("job-1", "output.pdf")
        assert url == "https://signed.url/output.pdf"

    async def test_blob_path_contains_job_id(self):
        blob = _make_blob()
        blob.generate_signed_url.return_value = "https://signed.url"
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.generate_download_url("job-abc", "file.pdf")
        blob_path = bucket.blob.call_args[0][0]
        assert "job-abc" in blob_path

    async def test_failure_raises_storage_error(self):
        blob = _make_blob()
        blob.generate_signed_url.side_effect = Exception("creds error")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError):
            await repo.generate_download_url("job-1", "file.pdf")


# ---------------------------------------------------------------------------
# get_file_info
# ---------------------------------------------------------------------------


class TestGetFileInfo:
    async def test_returns_metadata_dict(self):
        blob = _make_blob("path/file.pdf")
        blob.size = 8192
        blob.content_type = "application/pdf"
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        info = await repo.get_file_info("path/file.pdf")
        assert isinstance(info, dict)
        assert info.get("size") == 8192


# ---------------------------------------------------------------------------
# get_job_input_files / get_job_output_files
# ---------------------------------------------------------------------------


class TestJobInputOutputFiles:
    async def test_input_files_prefix_contains_job_id(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        await repo.get_job_input_files("job-test")
        prefix = bucket.list_blobs.call_args[1].get("prefix", "")
        assert "job-test" in prefix

    async def test_output_files_prefix_contains_job_id(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        await repo.get_job_output_files("job-test")
        prefix = bucket.list_blobs.call_args[1].get("prefix", "")
        assert "job-test" in prefix
