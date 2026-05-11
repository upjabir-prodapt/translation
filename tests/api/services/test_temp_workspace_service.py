import pytest
from pathlib import Path
from src.api.services.temp_workspace_service import TempWorkspaceService, JobWorkspace

class TestTempWorkspaceService:
    def test_init(self, tmp_path):
        service = TempWorkspaceService(base_dir=tmp_path)
        assert service._root == tmp_path

    def test_create(self, tmp_path):
        service = TempWorkspaceService(base_dir=tmp_path)
        ws = service.create("job1")
        assert ws.root.exists()
        assert ws.input_dir.exists()
        assert "job1" in str(ws.root)

    def test_attempt_dir(self, tmp_path):
        service = TempWorkspaceService(base_dir=tmp_path)
        ws = service.create("job1")
        a_dir = ws.attempt_dir(1, "model:v1")
        assert "attempt_1_model_v1" in str(a_dir)
        assert a_dir.exists()

    def test_cleanup(self, tmp_path):
        service = TempWorkspaceService(base_dir=tmp_path)
        ws = service.create("job1")
        assert ws.root.exists()
        service.cleanup("job1")
        assert not ws.root.exists()
