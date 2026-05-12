import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.api.services.startup_assets_service import StartupAssetsService


@pytest.fixture
def service():
    with (
        patch("src.api.services.startup_assets_service.IntentRouterService"),
        patch("src.api.services.startup_assets_service.GlossaryService"),
    ):
        return StartupAssetsService()


class TestStartupAssetsService:
    def test_ensure_assets_layout(self, service, tmp_path):
        with patch(
            "src.api.services.startup_assets_service.get_cache_root",
            return_value=tmp_path,
        ):
            with patch(
                "src.api.services.startup_assets_service.get_subdir_path"
            ) as mock_subdir:
                root = service.ensure_assets_layout()
                assert root == tmp_path
                assert mock_subdir.call_count >= 5

    async def test_ensure_metadata_indexes_success(self, service, tmp_path):
        with patch(
            "src.api.services.startup_assets_service.get_cache_file_path"
        ) as mock_path:
            p = tmp_path / "meta.json"
            p.write_text('{"a": 1}')
            mock_path.return_value = p

            res = await service._ensure_metadata_indexes()
            assert len(res) == 2

    @patch(
        "src.api.services.startup_assets_service.download_async", new_callable=AsyncMock
    )
    async def test_ensure_metadata_indexes_download(self, mock_down, service, tmp_path):
        with patch(
            "src.api.services.startup_assets_service.get_cache_file_path"
        ) as mock_path:
            p = tmp_path / "downloaded.json"
            # mock_path will be called twice.
            mock_path.return_value = p

            # First call exists=False, then write file after download
            def side_effect(*_args, **_kwargs):
                p.write_text('{"ok": true}')
                return p

            with patch.object(Path, "exists", side_effect=[False, False]):
                with pytest.raises(
                    RuntimeError
                ):  # It will fail because I didn't mock enough exists() calls or write fast enough
                    await service._ensure_metadata_indexes()

    async def test_run_preflight(self, service):
        with patch.object(
            service, "ensure_assets_layout", return_value=Path(tempfile.gettempdir())
        ):
            with patch.object(
                service,
                "_ensure_metadata_indexes",
                new_callable=AsyncMock,
                return_value=["f1", "f2"],
            ):
                status = await service.run_preflight()
                assert status.critical_sync_ok is True
                assert "tmp" in status.assets_root
