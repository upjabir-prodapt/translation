import pytest
from pathlib import Path
from src.loaders.assets import get_doclayout_onnx_model_path, get_font_and_metadata
from src.loaders.exceptions import AssetIntegrityError, MetadataNotFoundError
from unittest.mock import patch, MagicMock

class TestAssets:
    @patch("src.loaders.assets.get_or_download_model")
    def test_get_doclayout_onnx_model_path_success(self, mock_get):
        mock_get.return_value = Path("/tmp/model.onnx")
        assert get_doclayout_onnx_model_path() == Path("/tmp/model.onnx")

    @patch("src.loaders.assets.get_or_download_model")
    def test_get_doclayout_onnx_model_path_failure(self, mock_get):
        mock_get.side_effect = Exception("error")
        with pytest.raises(AssetIntegrityError):
            get_doclayout_onnx_model_path()

    @patch("src.loaders.assets.get_font_metadata_by_name")
    def test_get_font_and_metadata_not_found(self, mock_get):
        mock_get.return_value = None
        with pytest.raises(MetadataNotFoundError):
            get_font_and_metadata("missing.ttf")

    @patch("src.loaders.assets.get_font_metadata_by_name")
    @patch("src.loaders.assets.download_and_verify")
    @patch("src.loaders.assets.get_cache_file_path")
    def test_get_font_and_metadata_success(self, mock_path, mock_down, mock_get):
        mock_meta = MagicMock()
        mock_meta.sha3_256 = "abc"
        mock_get.return_value = mock_meta
        mock_path.return_value = Path("/tmp/font.ttf")
        
        path, meta = get_font_and_metadata("arial.ttf")
        assert path == Path("/tmp/font.ttf")
        assert meta is not None
