import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.loaders.assets import get_doclayout_onnx_model_path
from src.worker.loaders.assets import get_font_and_metadata
from src.worker.loaders.exceptions import AssetIntegrityError
from src.worker.loaders.exceptions import MetadataNotFoundError


class TestAssets:
    @patch("src.worker.loaders.assets.get_or_download_model")
    def test_get_doclayout_onnx_model_path_success(self, mock_get):
        path = Path(tempfile.gettempdir()) / "model.onnx"
        mock_get.return_value = path
        assert get_doclayout_onnx_model_path() == path

    @patch("src.worker.loaders.assets.get_or_download_model")
    def test_get_doclayout_onnx_model_path_failure(self, mock_get):
        mock_get.side_effect = Exception("error")
        with pytest.raises(AssetIntegrityError):
            get_doclayout_onnx_model_path()

    @patch("src.worker.loaders.assets.get_font_metadata_by_name")
    def test_get_font_and_metadata_not_found(self, mock_get):
        mock_get.return_value = None
        with pytest.raises(MetadataNotFoundError):
            get_font_and_metadata("missing.ttf")

    @patch("src.worker.loaders.assets.get_font_metadata_by_name")
    @patch("src.worker.loaders.assets.download_and_verify")
    @patch("src.worker.loaders.assets.get_cache_file_path")
    def test_get_font_and_metadata_success(self, mock_path, mock_down, mock_get):
        mock_meta = MagicMock()
        mock_meta.sha3_256 = "abc"
        mock_get.return_value = mock_meta
        path = Path(tempfile.gettempdir()) / "font.ttf"
        mock_path.return_value = path

        path_result, meta = get_font_and_metadata("arial.ttf")
        assert path_result == path
        assert meta is not None
