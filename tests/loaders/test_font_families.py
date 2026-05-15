import pytest
from src.loaders.models.font_families import FontFamilyConfig
from src.loaders.models.font_families import get_font_family


class TestFontFamilies:
    def test_font_family_config_validation(self):
        with pytest.raises(TypeError, match="must be a list"):
            FontFamilyConfig(script="not a list", normal=[], fallback=[], base=[])

        with pytest.raises(TypeError, match="must be a string"):
            FontFamilyConfig(script=[123], normal=[], fallback=[], base=[])

    def test_all_fonts(self):
        config = FontFamilyConfig(
            script=["s"], normal=["n"], fallback=["f"], base=["b"]
        )
        fonts = list(config.all_fonts())
        assert sorted(fonts) == ["b", "f", "n", "s"]

    def test_get_font_family(self):
        assert get_font_family("KR") is not None
        assert get_font_family("JP") is not None
        assert get_font_family("unknown") is not None  # Defaults to EN
