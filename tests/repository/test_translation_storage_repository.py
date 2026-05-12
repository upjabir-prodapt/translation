import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.repository.storage_repository import StorageError
from src.repository.translation_storage_repository import TranslationStorageRepository


@pytest.fixture
def mock_storage_client():
    return MagicMock()


@pytest.fixture
def repo(mock_storage_client):
    with patch(
        "src.repository.storage_repository.storage.Client",
        return_value=mock_storage_client,
    ):
        with patch("src.repository.storage_repository.settings") as mock_settings:
            mock_settings.GCS_BUCKET_NAME = "test-bucket"
            mock_settings.GCS_TRANSLATION_PREFIX = "translation"
            mock_settings.GCS_INPUT_FOLDER = "input"
            mock_settings.GCS_OUTPUT_FOLDER = "output"
            mock_settings.GCS_ASSETS_PREFIX = "assets"
            return TranslationStorageRepository()


class TestTranslationStorageRepository:
    @pytest.mark.asyncio
    async def test_download_input_pdf(self, repo, mock_storage_client):
        with patch.object(
            repo, "download_file", return_value=Path(tempfile.gettempdir()) / "ok"
        ) as mock_down:
            await repo.download_input_pdf(
                "job1", Path(tempfile.gettempdir()) / "input.pdf"
            )
            mock_down.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_output_files_success(self, repo, tmp_path):
        p = tmp_path / "mono.pdf"
        p.write_text("content")
        with patch.object(repo, "upload_file", return_value="gs://b/o") as mock_up:
            uris = await repo.upload_output_files("job1", {"mono": p})
            assert uris["mono"] == "gs://b/o"

    @pytest.mark.asyncio
    async def test_upload_output_files_not_exists(self, repo):
        uris = await repo.upload_output_files("job1", {"missing": Path("/nonexistent")})
        assert uris == {}

    @pytest.mark.asyncio
    async def test_upload_output_files_error(self, repo, tmp_path):
        p = tmp_path / "err.pdf"
        p.write_text("content")
        with patch.object(repo, "upload_file", side_effect=StorageError("Fail")):
            with pytest.raises(StorageError):
                await repo.upload_output_files("job1", {"err": p})

    @pytest.mark.asyncio
    async def test_download_asset(self, repo):
        with patch.object(
            repo, "download_file", return_value=Path(tempfile.gettempdir()) / "a"
        ) as mock_down:
            await repo.download_asset("path", Path(tempfile.gettempdir()) / "dest")
            mock_down.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_assets(self, repo):
        with patch.object(repo, "list_files", return_value=[]) as mock_list:
            res = await repo.list_assets()
            assert res == []

    @pytest.mark.asyncio
    async def test_download_glossary(self, repo):
        with patch.object(
            repo, "download_file", return_value=Path(tempfile.gettempdir()) / "g"
        ) as mock_down:
            await repo.download_glossary(
                "job1", Path(tempfile.gettempdir()) / "d", "g.json"
            )
            mock_down.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_job_files_keep_output(self, repo):
        with patch.object(repo, "delete_files", return_value=5) as mock_del:
            res = await repo.cleanup_job_files("job1", keep_output=True)
            assert res == 5
            # Should use input prefix
            args = mock_del.call_args[0][0]
            assert "input" in args

    @pytest.mark.asyncio
    async def test_cleanup_job_files_all(self, repo):
        with patch.object(repo, "delete_files", return_value=10) as mock_del:
            res = await repo.cleanup_job_files("job1", keep_output=False)
            assert res == 10
            # Should use job prefix
            args = mock_del.call_args[0][0]
            assert "translation/job1/" in args
