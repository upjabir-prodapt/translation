import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.loaders.exceptions import MetadataNotFoundError
from src.worker.loaders.repositories.metadata_repository import CMAP_METADATA
from src.worker.loaders.repositories.metadata_repository import EMBEDDING_FONT_METADATA
from src.worker.loaders.repositories.metadata_repository import _load_json_file
from src.worker.loaders.repositories.metadata_repository import clear_metadata_cache
from src.worker.loaders.repositories.metadata_repository import get_cmap_metadata
from src.worker.loaders.repositories.metadata_repository import get_font_metadata
from src.worker.loaders.repositories.metadata_repository import (
    get_font_metadata_by_name,
)


class TestMetadataRepository:
    def test_load_json_file_success(self, tmp_path):
        p = tmp_path / "test.json"
        data = {"a": 1}
        p.write_text(json.dumps(data))
        assert _load_json_file(p) == data

    def test_load_json_file_not_found(self):
        with pytest.raises(MetadataNotFoundError):
            _load_json_file(Path("nonexistent.json"))

    def test_load_json_file_invalid(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("invalid json")
        with pytest.raises(MetadataNotFoundError, match="Invalid JSON"):
            _load_json_file(p)

    def test_get_font_metadata_success(self, tmp_path):
        # Reset cache
        clear_metadata_cache()
        with patch(
            "src.worker.loaders.repositories.metadata_repository.get_cache_file_path"
        ) as mock_path:
            p = tmp_path / "fonts.json"
            p.write_text(
                json.dumps(
                    {
                        "arial.ttf": {
                            "sha3_256": "abc",
                            "size": 100,
                            "url": "u",
                            "font_name": "Arial",
                        }
                    }
                )
            )
            mock_path.return_value = p

            meta = get_font_metadata()
            assert "arial.ttf" in meta
            assert meta["arial.ttf"].sha3_256 == "abc"

    def test_get_font_metadata_not_found(self, tmp_path):
        clear_metadata_cache()
        with patch(
            "src.worker.loaders.repositories.metadata_repository.get_cache_file_path",
            return_value=Path("/nonexistent"),
        ):
            meta = get_font_metadata()
            assert meta == {}

    def test_get_cmap_metadata_success(self, tmp_path):
        clear_metadata_cache()
        with patch(
            "src.worker.loaders.repositories.metadata_repository.get_cache_file_path"
        ) as mock_path:
            p = tmp_path / "cmap.json"
            p.write_text(
                json.dumps(
                    {
                        "UniGB": {
                            "sha3_256": "abc",
                            "size": 10,
                            "url": "u",
                            "cmap_name": "Uni",
                        }
                    }
                )
            )
            mock_path.return_value = p

            meta = get_cmap_metadata()
            assert "UniGB" in meta

    def test_get_font_metadata_by_name(self):
        with patch(
            "src.worker.loaders.repositories.metadata_repository.get_font_metadata",
            return_value={"a": MagicMock()},
        ):
            assert get_font_metadata_by_name("a") is not None
            assert get_font_metadata_by_name("b") is None

    def test_clear_metadata_cache(self):
        clear_metadata_cache()

    def test_font_metadata_proxy(self):
        mock_meta = MagicMock()
        mock_meta.sha3_256 = "abc"
        mock_meta.size = 100
        mock_meta.url = "u"
        mock_meta.font_name = "Arial"
        mock_meta.subset_font_path = "p"

        with (
            patch(
                "src.worker.loaders.repositories.metadata_repository.get_font_metadata_by_name",
                return_value=mock_meta,
            ),
            patch(
                "src.worker.loaders.repositories.metadata_repository.get_font_metadata",
                return_value={"test": mock_meta},
            ),
        ):
            res = EMBEDDING_FONT_METADATA["test"]
            assert res["sha3_256"] == "abc"
            assert "test" in EMBEDDING_FONT_METADATA
            assert len(EMBEDDING_FONT_METADATA) == 1
            assert list(EMBEDDING_FONT_METADATA.keys()) == ["test"]
            assert list(EMBEDDING_FONT_METADATA.values())[0]["sha3_256"] == "abc"
            assert list(EMBEDDING_FONT_METADATA.items())[0][0] == "test"

    def test_font_metadata_proxy_key_error(self):
        with patch(
            "src.worker.loaders.repositories.metadata_repository.get_font_metadata_by_name",
            return_value=None,
        ):
            with pytest.raises(KeyError):
                _ = EMBEDDING_FONT_METADATA["missing"]

    def test_cmap_metadata_proxy(self):
        mock_meta = MagicMock()
        mock_meta.sha3_256 = "abc"
        mock_meta.size = 10
        mock_meta.url = "u"
        mock_meta.cmap_name = "Uni"

        with (
            patch(
                "src.worker.loaders.repositories.metadata_repository.get_cmap_metadata_by_name",
                return_value=mock_meta,
            ),
            patch(
                "src.worker.loaders.repositories.metadata_repository.get_cmap_metadata",
                return_value={"UniGB": mock_meta},
            ),
        ):
            res = CMAP_METADATA["UniGB"]
            assert res["sha3_256"] == "abc"
            assert "UniGB" in CMAP_METADATA
            assert len(CMAP_METADATA) == 1
            assert list(CMAP_METADATA.keys()) == ["UniGB"]
            assert list(CMAP_METADATA.values())[0]["sha3_256"] == "abc"
            assert list(CMAP_METADATA.items())[0][0] == "UniGB"

    def test_cmap_metadata_proxy_key_error(self):
        with patch(
            "src.worker.loaders.repositories.metadata_repository.get_cmap_metadata_by_name",
            return_value=None,
        ):
            with pytest.raises(KeyError):
                _ = CMAP_METADATA["missing"]
