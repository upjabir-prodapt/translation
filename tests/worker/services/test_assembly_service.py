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
