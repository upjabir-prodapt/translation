import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.repository.translation_storage_repository import TranslationStorageRepository
from pathlib import Path

@pytest.fixture
def mock_storage_client():
    return MagicMock()

@pytest.fixture
def repo(mock_storage_client):
    with patch("src.repository.storage_repository.storage.Client", return_value=mock_storage_client):
        with patch("src.repository.storage_repository.settings") as mock_settings:
            mock_settings.GCS_BUCKET_NAME = "test-bucket"
            return TranslationStorageRepository()

class TestTranslationStorageRepository:
    async def test_download_input_pdf(self, repo, mock_storage_client):
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        await repo.download_input_pdf("job1", Path("/tmp/input.pdf"))
        mock_blob.download_to_filename.assert_called_once()

    async def test_upload_output_files(self, repo, mock_storage_client, tmp_path):
        p = tmp_path / "mono.pdf"
        p.write_text("content")
        
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        uris = await repo.upload_output_files("job1", {"mono": p})
        assert "mono" in uris
        assert mock_blob.upload_from_filename.called

    async def test_list_assets(self, repo, mock_storage_client):
        mock_storage_client.list_blobs.return_value = []
        res = await repo.list_assets()
        assert res == []

    def test_download_asset_sync(self, repo, mock_storage_client):
        with patch("src.repository.storage_repository.download_blob_sync") as mock_sync:
            repo.download_asset_sync("path", Path("/tmp"))
            mock_sync.assert_called_once()
