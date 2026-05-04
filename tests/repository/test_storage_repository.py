"""
Unit tests for repository/storage_repository.py — StorageRepository.

All GCS client calls are mocked. asyncio.to_thread is exercised as normal
since the mock methods are synchronous MagicMocks, which run fine in threads.
"""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core.exceptions import GoogleAPIError

from src.repository.repository_exception import StorageError
from src.repository.storage_repository import (
    FileType,
    StoragePath,
    StorageRepository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(bucket=None, client=None) -> StorageRepository:
    """Build StorageRepository with injected mock client and bucket."""
    mock_client = client or MagicMock()
    mock_bucket = bucket or MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return StorageRepository(client=mock_client, bucket_name="test-bucket")


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


# ---------------------------------------------------------------------------
# StoragePath
# ---------------------------------------------------------------------------


class TestStoragePath:
    def test_prefix_only(self):
        assert StoragePath(prefix="translation").build() == "translation"

    def test_prefix_and_job_id(self):
        p = StoragePath(prefix="translation", job_id="j1")
        assert p.build() == "translation/j1"

    def test_full_path(self):
        p = StoragePath(prefix="trans", job_id="j1", folder="input", filename="doc.pdf")
        assert p.build() == "trans/j1/input/doc.pdf"

    def test_missing_parts_skipped(self):
        p = StoragePath(prefix="trans", filename="file.pdf")
        assert p.build() == "trans/file.pdf"


# ---------------------------------------------------------------------------
# upload_file — bytes source
# ---------------------------------------------------------------------------


class TestUploadFileBytes:
    async def test_uploads_bytes_returns_gcs_uri(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        uri = await repo.upload_file(b"pdf content", "path/file.pdf", FileType.PDF)
        assert uri == "gs://test-bucket/path/file.pdf"

    async def test_upload_from_string_called(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.upload_file(b"content", "path/file.pdf", FileType.PDF)
        blob.upload_from_string.assert_called_once()

    async def test_metadata_set_on_blob(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.upload_file(b"data", "path/f.pdf", metadata={"key": "val"})
        assert blob.metadata == {"key": "val"}

    async def test_content_type_passed_with_file_type(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.upload_file(b"data", "path/f.csv", FileType.CSV)
        call_kwargs = blob.upload_from_string.call_args
        assert "text/csv" in str(call_kwargs)

    async def test_google_api_error_raises_storage_error(self):
        blob = _make_blob()
        blob.upload_from_string.side_effect = GoogleAPIError("quota exceeded")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError, match="Failed to upload"):
            await repo.upload_file(b"data", "path/f.pdf", FileType.PDF)

    async def test_invalid_source_type_raises_value_error(self):
        repo = _make_repo()
        with pytest.raises((ValueError, StorageError)):
            await repo.upload_file(12345, "path/f.pdf")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# upload_file — Path source
# ---------------------------------------------------------------------------


class TestUploadFilePath:
    async def test_uploads_from_path(self, tmp_path):
        local = tmp_path / "doc.pdf"
        local.write_bytes(b"pdf content")
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        uri = await repo.upload_file(local, "output/doc.pdf", FileType.PDF)
        assert uri == "gs://test-bucket/output/doc.pdf"
        blob.upload_from_filename.assert_called_once()

    async def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "ghost.pdf"
        repo = _make_repo()
        with pytest.raises((FileNotFoundError, StorageError)):
            await repo.upload_file(missing, "output/ghost.pdf")


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    async def test_downloads_to_local_path(self, tmp_path):
        dest = tmp_path / "downloaded.pdf"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        result = await repo.download_file("path/file.pdf", dest)
        blob.download_to_filename.assert_called_once_with(str(dest))
        assert result == dest

    async def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "subdir" / "nested" / "file.pdf"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.download_file("path/file.pdf", dest, create_dirs=True)
        assert dest.parent.exists()

    async def test_google_api_error_raises_storage_error(self, tmp_path):
        dest = tmp_path / "file.pdf"
        blob = _make_blob()
        blob.download_to_filename.side_effect = GoogleAPIError("not found")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError, match="Failed to download"):
            await repo.download_file("path/file.pdf", dest)


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


class TestListFiles:
    async def test_returns_list_of_blobs(self):
        blobs = [_make_blob("a.pdf"), _make_blob("b.pdf")]
        bucket = MagicMock()
        bucket.list_blobs.return_value = blobs
        repo = _make_repo(bucket=bucket)

        result = await repo.list_files(prefix="translation/job1/")
        assert len(result) == 2

    async def test_empty_prefix_returns_all(self):
        blobs = [_make_blob("x.pdf")]
        bucket = MagicMock()
        bucket.list_blobs.return_value = blobs
        repo = _make_repo(bucket=bucket)

        result = await repo.list_files()
        assert len(result) == 1

    async def test_max_results_passed(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        await repo.list_files(max_results=10)
        call_kwargs = bucket.list_blobs.call_args[1]
        assert call_kwargs.get("max_results") == 10

    async def test_google_api_error_raises_storage_error(self):
        bucket = MagicMock()
        bucket.list_blobs.side_effect = GoogleAPIError("error")
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError, match="Failed to list"):
            await repo.list_files()


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------


class TestDeleteFile:
    async def test_deletes_blob(self):
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.delete_file("path/file.pdf")
        blob.delete.assert_called_once()

    async def test_google_api_error_raises_storage_error(self):
        blob = _make_blob()
        blob.delete.side_effect = GoogleAPIError("permission denied")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError, match="Failed to delete"):
            await repo.delete_file("path/file.pdf")


# ---------------------------------------------------------------------------
# delete_files (batch)
# ---------------------------------------------------------------------------


class TestDeleteFiles:
    async def test_deletes_multiple_blobs(self):
        blobs = [_make_blob(f"path/{i}.pdf") for i in range(3)]
        bucket = MagicMock()
        bucket.list_blobs.return_value = blobs
        bucket.delete_blobs.return_value = None
        repo = _make_repo(bucket=bucket)

        count = await repo.delete_files("path/")
        assert count == 3

    async def test_returns_zero_for_empty_prefix(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        count = await repo.delete_files("empty/prefix/")
        assert count == 0
        bucket.delete_blobs.assert_not_called()


# ---------------------------------------------------------------------------
# file_exists
# ---------------------------------------------------------------------------


class TestFileExists:
    async def test_returns_true_when_exists(self):
        blob = _make_blob()
        blob.exists.return_value = True
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        assert await repo.file_exists("path/file.pdf") is True

    async def test_returns_false_when_not_exists(self):
        blob = _make_blob()
        blob.exists.return_value = False
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        assert await repo.file_exists("path/missing.pdf") is False

    async def test_returns_false_on_exception(self):
        blob = _make_blob()
        blob.exists.side_effect = Exception("error")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        assert await repo.file_exists("path/file.pdf") is False


# ---------------------------------------------------------------------------
# build_job_path / build_asset_path
# ---------------------------------------------------------------------------


class TestPathBuilders:
    def test_build_job_path_with_folder_and_file(self):
        repo = _make_repo()
        path = repo.build_job_path("job-1", folder="input", filename="doc.pdf")
        assert "job-1" in path
        assert "input" in path
        assert "doc.pdf" in path

    def test_build_job_path_without_filename(self):
        repo = _make_repo()
        path = repo.build_job_path("job-1", folder="output")
        assert "job-1" in path
        assert "output" in path

    def test_build_asset_path(self):
        repo = _make_repo()
        path = repo.build_asset_path("model.onnx")
        assert "model.onnx" in path


# ---------------------------------------------------------------------------
# generate_signed_url
# ---------------------------------------------------------------------------


class TestGenerateSignedUrl:
    async def test_returns_url_string(self):
        blob = _make_blob()
        blob.generate_signed_url.return_value = "https://storage.googleapis.com/signed"
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        url = await repo.generate_signed_url("path/file.pdf", expires_in=3600)
        assert url == "https://storage.googleapis.com/signed"

    async def test_parses_gs_uri(self):
        blob = _make_blob()
        blob.generate_signed_url.return_value = "https://signed.url"
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        # Should not raise - gs:// URI is parsed before calling blob
        await repo.generate_signed_url("gs://test-bucket/path/file.pdf")
        bucket.blob.assert_called_with("path/file.pdf")

    async def test_failure_raises_storage_error(self):
        blob = _make_blob()
        blob.generate_signed_url.side_effect = Exception("credentials error")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError, match="Failed to generate"):
            await repo.generate_signed_url("path/file.pdf")


# ---------------------------------------------------------------------------
# get_file_metadata
# ---------------------------------------------------------------------------


class TestGetFileMetadata:
    async def test_returns_metadata_dict(self):
        blob = _make_blob("path/file.pdf")
        blob.size = 8192
        blob.content_type = "application/pdf"
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        meta = await repo.get_file_metadata("path/file.pdf")
        assert meta["name"] == "path/file.pdf"
        assert meta["size"] == 8192
        assert meta["content_type"] == "application/pdf"

    async def test_failure_raises_storage_error(self):
        blob = _make_blob()
        blob.reload.side_effect = Exception("not found")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError, match="Failed to get file metadata"):
            await repo.get_file_metadata("path/file.pdf")


# ---------------------------------------------------------------------------
# _extract_blob_path (private helper — tested via public interface)
# ---------------------------------------------------------------------------


class TestExtractBlobPath:
    def test_plain_path_unchanged(self):
        repo = _make_repo()
        assert repo._extract_blob_path("path/file.pdf") == "path/file.pdf"

    def test_strips_gs_scheme(self):
        repo = _make_repo()
        result = repo._extract_blob_path("gs://test-bucket/path/file.pdf")
        assert result == "path/file.pdf"

    def test_strips_bucket_prefix(self):
        repo = _make_repo()
        result = repo._extract_blob_path("test-bucket/path/file.pdf")
        assert result == "path/file.pdf"

    def test_gs_uri_different_bucket_returns_raw_path(self):
        """When gs:// URI has a different bucket, the path after bucket name is returned."""
        repo = _make_repo()
        result = repo._extract_blob_path("gs://other-bucket/path/file.pdf")
        # gs:// stripped → "other-bucket/path/file.pdf" — doesn't start with "test-bucket/"
        assert "path/file.pdf" in result


# ---------------------------------------------------------------------------
# FileType enum
# ---------------------------------------------------------------------------


class TestFileTypeEnum:
    @pytest.mark.parametrize(
        "ft, expected_ct",
        [
            (FileType.PDF, "application/pdf"),
            (FileType.CSV, "text/csv"),
            (FileType.JSON, "application/json"),
            (FileType.TEXT, "text/plain"),
        ],
    )
    def test_content_type_values(self, ft, expected_ct):
        assert ft.value == expected_ct
