import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.loaders.exceptions import WarmupError
from src.worker.loaders.services.warmup_service import WarmupService
from src.worker.loaders.services.warmup_service import async_warmup
from src.worker.loaders.services.warmup_service import warmup


@pytest.fixture
def service():
    return WarmupService()


class TestWarmupService:
    @patch.object(WarmupService, "_bulk_sync_phased", new_callable=AsyncMock)
    @patch.object(WarmupService, "_download_metadata_files", new_callable=AsyncMock)
    @patch.object(WarmupService, "_warmup_models", new_callable=AsyncMock)
    @patch.object(WarmupService, "_warmup_fonts", new_callable=AsyncMock)
    @patch.object(WarmupService, "_warmup_cmaps", new_callable=AsyncMock)
    @patch.object(WarmupService, "_warmup_tiktoken", new_callable=AsyncMock)
    @patch("src.worker.loaders.services.warmup_service.clear_metadata_cache")
    async def test_warmup_all_success(
        self,
        mock_clear,
        mock_tik,
        mock_cmap,
        mock_font,
        mock_models,
        mock_meta,
        mock_sync,
        service,
    ):
        res = await service.warmup_all()
        assert res.success is True
        mock_sync.assert_called_once()

    @patch.object(WarmupService, "_bulk_sync_phased", side_effect=Exception("Crash"))
    async def test_warmup_all_failure(self, mock_sync, service):
        with pytest.raises(WarmupError, match="Asset warmup failed"):
            await service.warmup_all()

    async def test_warmup_models(self, service):
        with patch(
            "src.worker.loaders.services.warmup_service.get_or_download_model_async",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = [Path("/m1"), Exception("m2 fail")]
            await service._warmup_models()
            assert service._download_stats["verified"] == 1
            assert len(service._download_stats["failed"]) == 1

    @patch("src.worker.loaders.services.warmup_service.get_font_metadata")
    @patch("src.worker.loaders.services.warmup_service.get_cache_file_path")
    @patch("src.worker.loaders.services.warmup_service.verify_or_delete")
    @patch("src.worker.loaders.services.warmup_service.download_async", new_callable=AsyncMock)
    async def test_warmup_fonts(
        self, mock_down, mock_verify, mock_path, mock_get_meta, service
    ):
        mock_font = MagicMock()
        mock_font.sha3_256 = "abc"
        mock_get_meta.return_value = {"arial.ttf": mock_font}
        mock_verify.return_value = False  # Force download

        await service._warmup_fonts()
        mock_down.assert_called_once()

    @patch("src.worker.loaders.services.warmup_service.get_cmap_metadata")
    @patch("src.worker.loaders.services.warmup_service.verify_or_delete")
    @patch("src.worker.loaders.services.warmup_service.download_async", new_callable=AsyncMock)
    async def test_warmup_cmaps(self, mock_down, mock_verify, mock_get_meta, service):
        mock_cmap = MagicMock()
        mock_cmap.sha3_256 = "abc"
        mock_get_meta.return_value = {"UniGB": mock_cmap}
        mock_verify.return_value = True  # No download

        await service._warmup_cmaps()
        mock_down.assert_not_called()

    async def test_download_metadata_files_failure(self, service):
        with patch(
            "src.worker.loaders.services.warmup_service.download_async",
            side_effect=Exception("meta fail"),
        ):
            # Should log and continue
            await service._download_metadata_files()

    async def test_bulk_sync_phased_with_repo(self, service):
        mock_repo = AsyncMock()
        prefix = "assets"
        m1 = MagicMock()
        m1.name = f"{prefix}/fonts/file1"
        m2 = MagicMock()
        m2.name = f"{prefix}/cmaps/file2"
        mock_repo.list_assets.return_value = [m1, m2]
        service.storage_repo = mock_repo

        with patch("src.worker.loaders.services.warmup_service.settings") as mock_settings:
            mock_settings.GCS_ASSETS_PREFIX = prefix
            mock_settings.WARMUP_SYNC_CONCURRENCY = 1
            mock_settings.WARMUP_SYNC_PHASE_PREFIXES = ["fonts", "cmaps"]

            with patch.object(
                service, "_bulk_sync_group", new_callable=AsyncMock, return_value=1
            ) as mock_sync:
                await service._bulk_sync_phased()
                assert mock_sync.call_count == 2

    async def test_bulk_sync_phased_empty(self, service):
        mock_repo = AsyncMock()
        mock_repo.list_assets.return_value = []
        service.storage_repo = mock_repo
        res = await service._bulk_sync_phased()
        assert res == 0

    async def test_bulk_sync_group_success(self, service):
        mock_blob = MagicMock()
        mock_blob.name = "assets/fonts/f1"
        mock_blob.size = 100

        with (
            patch(
                "src.worker.loaders.services.warmup_service.get_cache_file_path",
                return_value=Path(tempfile.gettempdir()) / "f1",
            ),
            patch("src.worker.loaders.services.warmup_service.get_file_size", return_value=50),
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_repo = AsyncMock()
            res = await service._bulk_sync_group(
                group_name="fonts",
                rel_paths=["fonts/f1"],
                blob_map={"fonts/f1": mock_blob},
                repo=mock_repo,
                semaphore=asyncio.Semaphore(1),
            )
            assert res == 1
            mock_repo.download_asset.assert_called_once()

    async def test_download_blob_if_needed_error(self, service):
        mock_repo = AsyncMock()
        mock_repo.download_asset.side_effect = Exception("Download Error")

        with (
            patch(
                "src.worker.loaders.services.warmup_service.get_cache_file_path",
                return_value=Path(tempfile.gettempdir()) / "f2",
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):
            res = await service._download_blob_if_needed(
                rel_path="p",
                blob=MagicMock(),
                repo=mock_repo,
                semaphore=asyncio.Semaphore(1),
            )
            assert res is False
            assert "p" in service._download_stats["failed"]

    async def test_warmup_tiktoken_error(self, service):
        with patch(
            "src.worker.loaders.services.warmup_service.get_subdir_path",
            side_effect=Exception("IO Error"),
        ):
            # Non-critical, should catch
            await service._warmup_tiktoken()

    def test_init_tiktoken(self, service):
        with patch("tiktoken.encoding_for_model") as mock_enc:
            service._init_tiktoken()
            mock_enc.assert_called_once_with("gpt-4o")

    @patch(
        "src.worker.loaders.services.warmup_service.WarmupService.warmup_all",
        new_callable=AsyncMock,
    )
    async def test_async_warmup(self, mock_warm):
        await async_warmup()
        mock_warm.assert_called_once()

    @patch("src.worker.loaders.services.warmup_service.async_warmup")
    def test_warmup_sync(self, mock_async_warm):
        from tests.async_test_utils import mock_asyncio_run

        # When no loop is running, warmup() delegates to asyncio.run(async_warmup(...))
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", side_effect=mock_asyncio_run) as mock_run:
                warmup()
                mock_async_warm.assert_called_once_with(None)
                mock_run.assert_called_once()
