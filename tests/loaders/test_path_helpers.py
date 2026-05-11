import pytest
from pathlib import Path
from src.loaders.utils.path_helpers import get_cache_root, get_cache_file_path, get_subdir_path, _assert_under_cache_root

class TestPathHelpers:
    def test_get_cache_root(self):
        root = get_cache_root()
        assert isinstance(root, Path)

    def test_assert_under_cache_root_success(self):
        root = get_cache_root()
        path = root / "subdir" / "file.txt"
        # Should not raise
        _assert_under_cache_root(path)

    def test_assert_under_cache_root_failure(self):
        with pytest.raises(ValueError, match="escapes cache root"):
            _assert_under_cache_root(Path("/etc/passwd"))

    def test_get_cache_file_path(self):
        path = get_cache_file_path("test.txt", subdir="test_subdir")
        assert path.name == "test.txt"
        assert "test_subdir" in str(path)

    def test_get_subdir_path(self):
        path = get_subdir_path("test_only_subdir")
        assert path.name == "test_only_subdir"
        assert path.exists()
