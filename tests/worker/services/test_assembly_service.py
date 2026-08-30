from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.worker.services.assembly_service import AssemblyService


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.build_job_path = MagicMock()
    return storage


@pytest.fixture
def service(mock_storage):
    return AssemblyService(storage=mock_storage)


class TestAssemblyService:
    async def test_upload_output_not_exists(self, service):
        res = await service.upload_output("job1", Path("none.pdf"))
        assert res is None

    async def test_upload_output_zero_byte_file_not_uploaded(
        self, service, mock_storage, tmp_path
    ):
        """B.3.4: a zero-byte output must never be uploaded as a
        'successful' translation (e.g. a scanned/no-text-layer document
        that somehow produced an empty output file)."""
        p = tmp_path / "empty.pdf"
        p.touch()  # zero bytes
        res = await service.upload_output("job1", p)
        assert res is None
        mock_storage.upload_file.assert_not_called()

    async def test_upload_output_success(self, service, mock_storage, tmp_path):
        p = tmp_path / "out.pdf"
        p.write_text("content")
        mock_storage.build_job_path.return_value = "job1/output/out.pdf"
        mock_storage.upload_file.return_value = "gs://bucket/job1/output/out.pdf"

        res = await service.upload_output("job1", p)
        assert res == "gs://bucket/job1/output/out.pdf"
        mock_storage.upload_file.assert_called_once()

    async def test_upload_outputs(self, service, mock_storage, tmp_path):
        p1 = tmp_path / "out1.pdf"
        p1.write_text("c1")
        p2 = tmp_path / "out2.pdf"
        p2.write_text("c2")

        output_files = {"mono": p1, "dual": p2}
        mock_storage.upload_file.side_effect = ["gs://p1", "gs://p2"]

        res = await service.upload_outputs("job1", output_files)
        assert res["mono"] == "gs://p1"
        assert res["dual"] == "gs://p2"

    async def test_upload_outputs_skips_zero_byte_files(
        self, service, mock_storage, tmp_path
    ):
        """B.3.4: zero-byte outputs are skipped, non-empty siblings still upload."""
        empty = tmp_path / "empty.pdf"
        empty.touch()
        real = tmp_path / "real.pdf"
        real.write_text("content")

        mock_storage.upload_file.return_value = "gs://real"
        res = await service.upload_outputs("job1", {"mono": empty, "dual": real})
        assert "mono" not in res
        assert res["dual"] == "gs://real"
        mock_storage.upload_file.assert_called_once()
