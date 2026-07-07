"""
Unit tests for worker/repository/worker_storage_repository.py.

All GCS calls are mocked — no real GCP connection.
"""

from unittest.mock import MagicMock

import pytest

from repository.repository_exception import StorageError
from worker.repository.worker_storage_repository import WorkerStorageRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blob(name="path/file.pdf"):
    blob = MagicMock()
    blob.name = name
    blob.size = 4096
    blob.content_type = "application/pdf"
    blob.updated = None
    blob.time_created = None
    blob.md5_hash = "hash=="
    blob.metadata = {}
    return blob


def _make_repo(bucket=None, client=None):
    mock_client = client or MagicMock()
    mock_bucket = bucket or MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return WorkerStorageRepository(client=mock_client, bucket_name="test-bucket")


# ---------------------------------------------------------------------------
# download_input_pdf
# ---------------------------------------------------------------------------


class TestDownloadInputPdf:
    async def test_calls_download_file(self, tmp_path):
        dest = tmp_path / "input.pdf"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        result = await repo.download_input_pdf("job-1", dest)
        blob.download_to_filename.assert_called_once_with(str(dest))
        assert result == dest

    async def test_builds_correct_blob_path(self, tmp_path):
        dest = tmp_path / "file.pdf"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.download_input_pdf("job-abc", dest)
        blob_path_used = bucket.blob.call_args[0][0]
        assert "job-abc" in blob_path_used

    async def test_uses_provided_filename(self, tmp_path):
        dest = tmp_path / "custom.pdf"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.download_input_pdf("job-1", dest, filename="custom.pdf")
        blob_path_used = bucket.blob.call_args[0][0]
        assert "custom.pdf" in blob_path_used


# ---------------------------------------------------------------------------
# upload_output_files
# ---------------------------------------------------------------------------


class TestUploadOutputFiles:
    async def test_uploads_existing_files(self, tmp_path):
        mono = tmp_path / "mono.pdf"
        mono.write_bytes(b"mono content")

        blob = _make_blob("output/mono.pdf")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        result = await repo.upload_output_files("job-1", {"mono": mono})
        assert "mono" in result
        assert result["mono"].startswith("gs://")

    async def test_skips_nonexistent_files(self, tmp_path):
        missing = tmp_path / "nonexistent.pdf"
        repo = _make_repo()

        result = await repo.upload_output_files("job-1", {"dual": missing})
        assert "dual" not in result

    async def test_returns_gcs_uris(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"content")
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        uris = await repo.upload_output_files("job-1", {"mono": f})
        assert uris["mono"].startswith("gs://test-bucket/")

    async def test_upload_error_raises_storage_error(self, tmp_path):
        from google.api_core.exceptions import GoogleAPIError

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"content")
        blob = _make_blob()
        blob.upload_from_filename.side_effect = GoogleAPIError("upload failed")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        with pytest.raises(StorageError):
            await repo.upload_output_files("job-1", {"mono": f})


# ---------------------------------------------------------------------------
# download_asset (async)
# ---------------------------------------------------------------------------


class TestDownloadAsset:
    async def test_calls_download_file(self, tmp_path):
        dest = tmp_path / "model.onnx"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        result = await repo.download_asset("models/model.onnx", dest)
        blob.download_to_filename.assert_called_once_with(str(dest))

    async def test_prefixes_assets_path(self, tmp_path):
        dest = tmp_path / "file"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.download_asset("model.onnx", dest)
        blob_path = bucket.blob.call_args[0][0]
        assert "model.onnx" in blob_path


# ---------------------------------------------------------------------------
# cleanup_job_files
# ---------------------------------------------------------------------------


class TestCleanupJobFiles:
    async def test_keep_output_deletes_input_only(self):
        blobs = [_make_blob(f"input/{i}.pdf") for i in range(2)]
        bucket = MagicMock()
        bucket.list_blobs.return_value = blobs
        bucket.delete_blobs.return_value = None
        repo = _make_repo(bucket=bucket)

        count = await repo.cleanup_job_files("job-1", keep_output=True)
        assert count == 2

    async def test_no_keep_output_deletes_all(self):
        blobs = [_make_blob(f"path/{i}.pdf") for i in range(5)]
        bucket = MagicMock()
        bucket.list_blobs.return_value = blobs
        bucket.delete_blobs.return_value = None
        repo = _make_repo(bucket=bucket)

        count = await repo.cleanup_job_files("job-1", keep_output=False)
        assert count == 5

    async def test_returns_zero_when_no_files(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        count = await repo.cleanup_job_files("job-1")
        assert count == 0


# ---------------------------------------------------------------------------
# download_asset_sync
# ---------------------------------------------------------------------------


class TestDownloadAssetSync:
    def test_calls_download_blob_sync(self, tmp_path):
        dest = tmp_path / "model.onnx"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        result = repo.download_asset_sync("models/model.onnx", dest)
        blob.download_to_filename.assert_called_once_with(str(dest))
        assert result == dest

    def test_prefixes_assets_path(self, tmp_path):
        dest = tmp_path / "file"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        repo.download_asset_sync("model.onnx", dest)
        blob_path = bucket.blob.call_args[0][0]
        assert "model.onnx" in blob_path


# ---------------------------------------------------------------------------
# list_assets
# ---------------------------------------------------------------------------


class TestListAssets:
    async def test_returns_blobs_list(self):
        b1, b2 = _make_blob("assets/f1"), _make_blob("assets/f2")
        bucket = MagicMock()
        bucket.list_blobs.return_value = [b1, b2]
        repo = _make_repo(bucket=bucket)

        result = await repo.list_assets()
        assert len(result) == 2

    async def test_empty_assets(self):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []
        repo = _make_repo(bucket=bucket)

        result = await repo.list_assets()
        assert result == []


# ---------------------------------------------------------------------------
# download_glossary
# ---------------------------------------------------------------------------


class TestDownloadGlossary:
    async def test_downloads_glossary_file(self, tmp_path):
        dest = tmp_path / "glossary.csv"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        result = await repo.download_glossary("job-1", dest, "glossary.csv")
        blob.download_to_filename.assert_called_once_with(str(dest))
        assert result == dest

    async def test_builds_path_with_job_id(self, tmp_path):
        dest = tmp_path / "glossary.csv"
        blob = _make_blob()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        repo = _make_repo(bucket=bucket)

        await repo.download_glossary("job-xyz", dest, "glossary.csv")
        blob_path = bucket.blob.call_args[0][0]
        assert "job-xyz" in blob_path
        assert "glossary.csv" in blob_path


# ---------------------------------------------------------------------------
# get_worker_storage_repository factory
# ---------------------------------------------------------------------------


class TestGetWorkerStorageRepository:
    def test_returns_instance_with_client(self):
        from worker.repository.worker_storage_repository import WorkerStorageRepository
        from worker.repository.worker_storage_repository import (
            get_worker_storage_repository,
        )

        client = MagicMock()
        client.bucket.return_value = MagicMock()
        repo = get_worker_storage_repository(client=client, bucket_name="b")
        assert isinstance(repo, WorkerStorageRepository)

    def test_uses_provided_bucket_name(self):
        from worker.repository.worker_storage_repository import (
            get_worker_storage_repository,
        )

        client = MagicMock()
        client.bucket.return_value = MagicMock()
        repo = get_worker_storage_repository(client=client, bucket_name="my-bucket")
        assert repo.bucket_name == "my-bucket"
