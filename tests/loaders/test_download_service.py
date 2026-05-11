import pytest
from pathlib import Path
from src.loaders.services.download_service import download_with_retry, download_async, download_and_verify
from src.loaders.exceptions import AssetDownloadError
from unittest.mock import patch, MagicMock, AsyncMock

class TestDownloadService:
    @patch("src.loaders.services.download_service._get_storage_repo")
    def test_download_with_retry_success(self, mock_get_repo):
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        
        download_with_retry("blob", Path("/tmp/dest"))
        mock_repo.download_asset_sync.assert_called_once()

    @patch("src.loaders.services.download_service._get_storage_repo")
    def test_download_with_retry_failure(self, mock_get_repo):
        mock_repo = MagicMock()
        mock_repo.download_asset_sync.side_effect = Exception("fail")
        mock_get_repo.return_value = mock_repo
        
        with pytest.raises(AssetDownloadError):
            download_with_retry("blob", Path("/tmp/dest"))

    @patch("src.loaders.services.download_service.download_with_retry")
    @pytest.mark.asyncio
    async def test_download_async(self, mock_download):
        await download_async("blob", Path("/tmp/dest"))
        # Called via executor, but basically we check it was triggered
        mock_download.assert_called_once()

    @patch("src.loaders.services.download_service.download_with_retry")
    @patch("src.loaders.services.download_service.verify_and_raise")
    def test_download_and_verify(self, mock_verify, mock_down):
        download_and_verify("blob", Path("/tmp/dest"), "abc")
        mock_down.assert_called_once()
        mock_verify.assert_called_once()
