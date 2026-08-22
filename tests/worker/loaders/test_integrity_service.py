from pathlib import Path

import pytest
from src.worker.loaders.exceptions import AssetIntegrityError
from src.worker.loaders.services.integrity_service import is_valid
from src.worker.loaders.services.integrity_service import verify_and_raise


class TestIntegrityService:
    def test_verify_and_raise_not_found(self):
        with pytest.raises(AssetIntegrityError, match="not found"):
            verify_and_raise(Path("none.txt"), "abc")

    def test_verify_and_raise_failure(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("bad")
        with pytest.raises(AssetIntegrityError, match="integrity check failed"):
            verify_and_raise(p, "wrong")

    def test_is_valid(self, tmp_path):
        p = tmp_path / "ok.txt"
        p.write_text("ok")
        # Just verifying it calls the repo
        assert is_valid(p, "wrong") is False
