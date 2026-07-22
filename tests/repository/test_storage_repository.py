from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.repository.repository_exception import StorageError
from src.repository.storage_repository import FileType
from src.repository.storage_repository import StoragePath
from src.repository.storage_repository import StorageRepository


@pytest.fixture
def mock_storage_client():
    client = MagicMock()
    bucket = MagicMock()
    blob = MagicMock()
    client.bucket.return_value = bucket
    bucket.blob.return_value = blob
    return client


@pytest.fixture
def repo(mock_storage_client):
    with patch(
        "src.repository.storage_repository.storage.Client",
        return_value=mock_storage_client,
    ):
        with patch("src.repository.storage_repository.settings") as mock_settings:
            mock_settings.GCS_BUCKET_NAME = "test-bucket"
            return StorageRepository()


class TestStorageRepository:
    def test_init_defaults(self, mock_storage_client):
        with patch("src.repository.storage_repository.settings") as mock_settings:
            mock_settings.GCS_BUCKET_NAME = "test-bucket"
            repo = StorageRepository(client=mock_storage_client)
            assert repo.bucket_name == "test-bucket"

    async def test_upload_file_bytes(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        uri = await repo.upload_file(b"data", "path/to/blob", file_type=FileType.PDF)
        assert uri == "gs://test-bucket/path/to/blob"
        mock_blob.upload_from_string.assert_called_once()

    async def test_download_file_bytes(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        mock_blob.download_to_filename.return_value = None
        local_path = Path("local/path")
        with patch("src.repository.storage_repository.Path.mkdir"):
            await repo.download_file("gs://bucket/path", local_path)
        mock_blob.download_to_filename.assert_called_once_with(str(local_path))

    async def test_delete_file(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        await repo.delete_file("gs://bucket/path")
        mock_blob.delete.assert_called_once()

    async def test_file_exists(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        mock_blob.exists.return_value = True
        assert await repo.file_exists("gs://bucket/path") is True

    def test_storage_path_build(self):
        path = StoragePath(prefix="p", job_id="j", folder="f", filename="n")
        assert path.build() == "p/j/f/n"

    async def test_get_file_metadata(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        mock_blob.size = 100
        mock_blob.updated = datetime.now()
        mock_blob.content_type = "application/pdf"
        mock_blob.md5_hash = "abc"

        meta = await repo.get_file_metadata("gs://bucket/path")
        assert meta["size"] == 100

    async def test_list_files(self, repo, mock_storage_client):
        mock_bucket = mock_storage_client.bucket.return_value
        mock_bucket.list_blobs.return_value = [MagicMock(), MagicMock()]
        res = await repo.list_files(prefix="test")
        assert len(res) == 2

    async def test_delete_file_failure(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        mock_blob.delete.side_effect = Exception("error")
        # In reality, delete_file wraps generic Exceptions too or just GoogleAPIError?
        # The code showed catch GoogleAPIError.
        from google.api_core.exceptions import GoogleAPIError

        mock_blob.delete.side_effect = GoogleAPIError("fail")
        with pytest.raises(StorageError):
            await repo.delete_file("gs://bucket/path")
