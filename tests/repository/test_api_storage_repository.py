from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.repository.api_storage_repository import APIStorageRepository


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
        # We need to ensure settings are mocked during initialization if we don't pass bucket_name
        with patch("src.repository.storage_repository.settings") as mock_settings:
            mock_settings.GCS_BUCKET_NAME = "test-bucket"
            mock_settings.GCS_INPUT_FOLDER = "input"
            return APIStorageRepository()


class TestAPIStorageRepository:
    async def test_upload_input_pdf(self, repo, mock_storage_client):
        mock_bucket = mock_storage_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value

        uri = await repo.upload_input_pdf(b"content", "test.pdf", "job1")

        assert "job1/input/test.pdf" in uri
        mock_blob.upload_from_string.assert_called_once()

    async def test_get_file_info(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        mock_blob.size = 1024
        mock_blob.updated = "2026-05-08"
        mock_blob.content_type = "application/pdf"
        mock_blob.md5_hash = "abc"

        info = await repo.get_file_info("gs://bucket/blob")
        assert info["size"] == 1024
