import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.loaders.repositories.metadata_repository import _load_json_file, get_font_metadata, get_cmap_metadata, _font_cache, _cmap_cache, _cache_lock, get_font_metadata_by_name, clear_metadata_cache
from src.loaders.exceptions import MetadataNotFoundError

class TestMetadataRepository:
    def test_load_json_file_success(self, tmp_path):
        p = tmp_path / "test.json"
        data = {"a": 1}
        p.write_text(json.dumps(data))
        assert _load_json_file(p) == data

    def test_load_json_file_not_found(self):
        with pytest.raises(MetadataNotFoundError):
            _load_json_file(Path("nonexistent.json"))

    def test_get_font_metadata_success(self, tmp_path):
        # Reset cache
        with _cache_lock:
            from src.loaders.repositories import metadata_repository
            metadata_repository._font_cache = None
            
            with patch("src.loaders.repositories.metadata_repository.get_cache_file_path") as mock_path:
                p = tmp_path / "fonts.json"
                p.write_text(json.dumps({"arial.ttf": {"sha3_256": "abc"}}))
                mock_path.return_value = p
                
                meta = get_font_metadata()
                assert "arial.ttf" in meta
                assert meta["arial.ttf"].sha3_256 == "abc"

    def test_get_cmap_metadata_success(self, tmp_path):
        with _cache_lock:
            from src.loaders.repositories import metadata_repository
            metadata_repository._cmap_cache = None
            
            with patch("src.loaders.repositories.metadata_repository.get_cache_file_path") as mock_path:
                p = tmp_path / "cmap.json"
                p.write_text(json.dumps({"UniGB": {"sha3_256": "abc"}}))
                mock_path.return_value = p
                
                meta = get_cmap_metadata()
                assert "UniGB" in meta

    def test_get_font_metadata_by_name(self, tmp_path):
        with patch("src.loaders.repositories.metadata_repository.get_font_metadata", return_value={"a": MagicMock()}):
            assert get_font_metadata_by_name("a") is not None
            assert get_font_metadata_by_name("b") is None

    def test_clear_metadata_cache(self):
        clear_metadata_cache()

    def test_font_metadata_proxy(self):
        from src.loaders.repositories.metadata_repository import EMBEDDING_FONT_METADATA
        mock_meta = MagicMock()
        mock_meta.sha3_256 = "abc"
        with patch("src.loaders.repositories.metadata_repository.get_font_metadata_by_name", return_value=mock_meta), \
             patch("src.loaders.repositories.metadata_repository.get_font_metadata", return_value={"test": mock_meta}):
            res = EMBEDDING_FONT_METADATA["test"]
            assert res["sha3_256"] == "abc"
            assert "test" in EMBEDDING_FONT_METADATA
