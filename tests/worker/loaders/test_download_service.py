import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.loaders.exceptions import AssetDownloadError
from src.worker.loaders.exceptions import AssetIntegrityError
from src.worker.loaders.services.download_service import download_and_verify
from src.worker.loaders.services.download_service import download_and_verify_async
from src.worker.loaders.services.download_service import download_async
from src.worker.loaders.services.download_service import download_with_retry
from src.worker.loaders.services.download_service import get_or_download_model
from src.worker.loaders.services.download_service import get_or_download_model_async


class TestDownloadService:
    @patch("src.worker.loaders.services.download_service._get_storage_repo")
    def test_download_with_retry_success(self, mock_get_repo):
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        download_with_retry("blob", Path(tempfile.gettempdir()) / "dest")
        mock_repo.download_asset_sync.assert_called_once()

    @patch("src.worker.loaders.services.download_service._get_storage_repo")
    def test_download_with_retry_failure(self, mock_get_repo):
        mock_repo = MagicMock()
        mock_repo.download_asset_sync.side_effect = Exception("fail")
        mock_get_repo.return_value = mock_repo
        # Set DOWNLOAD_MAX_ATTEMPTS=1 for fast test
        with patch("src.worker.loaders.services.download_service.settings") as mock_settings:
            mock_settings.DOWNLOAD_MAX_ATTEMPTS = 1
            with pytest.raises(AssetDownloadError):
                download_with_retry("blob", Path(tempfile.gettempdir()) / "dest")

    @patch("src.worker.loaders.services.download_service.download_with_retry")
    @pytest.mark.asyncio
    async def test_download_async(self, mock_download):
        await download_async("blob", Path(tempfile.gettempdir()) / "dest")
        mock_download.assert_called_once()

    @patch("src.worker.loaders.services.download_service.download_with_retry")
    @patch("src.worker.loaders.services.download_service.verify_and_raise")
    def test_download_and_verify(self, mock_verify, mock_down):
        download_and_verify("blob", Path(tempfile.gettempdir()) / "dest", "abc")
        mock_down.assert_called_once()
        mock_verify.assert_called_once()

    @patch("src.worker.loaders.services.download_service.download_with_retry")
    @patch(
        "src.worker.loaders.services.download_service.verify_and_raise",
        side_effect=AssetIntegrityError("bad", "abc", "def"),
    )
    @patch("src.worker.loaders.services.download_service.verify_or_delete")
    def test_download_and_verify_integrity_error(
        self, mock_del, mock_verify, mock_down
    ):
        with pytest.raises(AssetIntegrityError):
            download_and_verify("blob", Path(tempfile.gettempdir()) / "dest", "abc")
        mock_del.assert_called_once()

    @patch("src.worker.loaders.services.download_service.download_async")
    @patch("src.worker.loaders.services.download_service.verify_and_raise")
    @pytest.mark.asyncio
    async def test_download_and_verify_async(self, mock_verify, mock_down):
        await download_and_verify_async(
            "blob", Path(tempfile.gettempdir()) / "dest", "abc"
        )
        mock_down.assert_called_once()
        mock_verify.assert_called_once()

    @patch(
        "src.worker.loaders.services.download_service.get_cache_file_path",
        return_value=Path(tempfile.gettempdir()) / "m",
    )
    @patch("src.worker.loaders.services.download_service.verify_or_delete", return_value=True)
    def test_get_or_download_model_cached(self, mock_verify, mock_path):
        res = get_or_download_model("f", "abc", "name")
        assert res == Path(tempfile.gettempdir()) / "m"

    @patch(
        "src.worker.loaders.services.download_service.get_cache_file_path",
        return_value=Path(tempfile.gettempdir()) / "m",
    )
    @patch("src.worker.loaders.services.download_service.verify_or_delete", return_value=False)
    @patch("src.worker.loaders.services.download_service.download_and_verify")
    def test_get_or_download_model_download(self, mock_down, mock_verify, mock_path):
        get_or_download_model("f", "abc", "name")
        mock_down.assert_called_once()

    @patch(
        "src.worker.loaders.services.download_service.get_cache_file_path",
        return_value=Path(tempfile.gettempdir()) / "m",
    )
    @patch("src.worker.loaders.services.download_service.verify_or_delete", return_value=True)
    @pytest.mark.asyncio
    async def test_get_or_download_model_async_cached(self, mock_verify, mock_path):
        res = await get_or_download_model_async("f", "abc", "name")
        assert res == Path(tempfile.gettempdir()) / "m"

    @patch(
        "src.worker.loaders.services.download_service.get_cache_file_path",
        return_value=Path(tempfile.gettempdir()) / "m",
    )
    @patch("src.worker.loaders.services.download_service.verify_or_delete", return_value=False)
    @patch("src.worker.loaders.services.download_service.download_and_verify_async")
    @pytest.mark.asyncio
    async def test_get_or_download_model_async_download(
        self, mock_down, mock_verify, mock_path
    ):
        await get_or_download_model_async("f", "abc", "name")
        mock_down.assert_called_once()
