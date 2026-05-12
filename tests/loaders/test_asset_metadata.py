from src.loaders.models.asset_metadata import AssetMetadata
from src.loaders.models.asset_metadata import CMapMetadata
from src.loaders.models.asset_metadata import FontMetadata
from src.loaders.models.asset_metadata import ModelMetadata


class TestAssetMetadata:
    def test_asset_metadata_from_dict(self):
        data = {"sha3_256": "abc", "size": 100, "url": "http://"}
        meta = AssetMetadata.from_dict("test", data)
        assert meta.name == "test"
        assert meta.sha3_256 == "abc"
        assert meta.size == 100

    def test_font_metadata_from_dict(self):
        data = {"sha3_256": "abc", "font_name": "Arial"}
        meta = FontMetadata.from_dict("test", data)
        assert meta.font_name == "Arial"
        assert meta.sha3_256 == "abc"

    def test_cmap_metadata_from_dict(self):
        data = {"cmap_name": "UniGB-UTF8-H"}
        meta = CMapMetadata.from_dict("test", data)
        assert meta.cmap_name == "UniGB-UTF8-H"

    def test_model_metadata(self):
        meta = ModelMetadata(name="m", sha3_256="s", model_type="onnx")
        assert meta.model_type == "onnx"
