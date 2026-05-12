"""Font family configurations for different languages."""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class FontFamilyConfig:
    """Configuration for a font family (script, normal, fallback, base)."""

    script: list[str]
    normal: list[str]
    fallback: list[str]
    base: list[str]

    def __post_init__(self) -> None:
        """Validate font family structure."""
        for category in ["script", "normal", "fallback", "base"]:
            fonts = getattr(self, category)
            if not isinstance(fonts, list):
                raise TypeError(f"Font family {category} must be a list")
            for font in fonts:
                if not isinstance(font, str):
                    raise TypeError(f"Font name must be a string, got {type(font)}")

    def all_fonts(self) -> Iterator[str]:
        """Iterate over all fonts in the family."""
        return itertools.chain(self.script, self.normal, self.fallback, self.base)


SOURCE_HAN_SANS_CN_REGULAR = "SourceHanSansCN-Regular.ttf"
GO_NOTO_KURRENT_REGULAR = "GoNotoKurrent-Regular.ttf"
GO_NOTO_KURRENT_BOLD = "GoNotoKurrent-Bold.ttf"

# Chinese (Simplified)
CN_FONT_FAMILY = FontFamilyConfig(
    script=[
        "LXGWWenKaiGB-Regular.1.520.ttf",
    ],
    normal=[
        "SourceHanSerifCN-Bold.ttf",
        "SourceHanSerifCN-Regular.ttf",
        "SourceHanSansCN-Bold.ttf",
<<<<<<< HEAD
        "SourceHanSansCN-Regular.ttf",
    ],
        GO_NOTO_KURRENT_REGULAR,
        GO_NOTO_KURRENT_BOLD,
    ],
    base=[SOURCE_HAN_SANS_CN_REGULAR],
>>>>>>> origin/feature/sonarqube_fix
)

HK_FONT_FAMILY = FontFamilyConfig(
    script=["LXGWWenKaiTC-Regular.1.520.ttf"],
    normal=[
        "SourceHanSerifHK-Bold.ttf",
        "SourceHanSerifHK-Regular.ttf",
        "SourceHanSansHK-Bold.ttf",
        "SourceHanSansHK-Regular.ttf",
    ],
    fallback=[
        GO_NOTO_KURRENT_REGULAR,
        GO_NOTO_KURRENT_BOLD,
    ],
    base=[SOURCE_HAN_SANS_CN_REGULAR],
)

# Taiwan (Traditional)
TW_FONT_FAMILY = FontFamilyConfig(
    script=["LXGWWenKaiTC-Regular.1.520.ttf"],
    normal=[
        "SourceHanSerifTW-Bold.ttf",
        "SourceHanSerifTW-Regular.ttf",
        "SourceHanSansTW-Bold.ttf",
        "SourceHanSansTW-Regular.ttf",
    ],
    fallback=[
        GO_NOTO_KURRENT_REGULAR,
        GO_NOTO_KURRENT_BOLD,
    ],
    base=[SOURCE_HAN_SANS_CN_REGULAR],
)

# Korean
KR_FONT_FAMILY = FontFamilyConfig(
    script=["MaruBuri-Regular.ttf"],
    normal=[
        "SourceHanSerifKR-Bold.ttf",
        "SourceHanSerifKR-Regular.ttf",
        "SourceHanSansKR-Bold.ttf",
        "SourceHanSansKR-Regular.ttf",
    ],
    fallback=[
        GO_NOTO_KURRENT_REGULAR,
        GO_NOTO_KURRENT_BOLD,
    ],
    base=[SOURCE_HAN_SANS_CN_REGULAR],
)

# Japanese
JP_FONT_FAMILY = FontFamilyConfig(
    script=["KleeOne-Regular.ttf"],
    normal=[
        "SourceHanSerifJP-Bold.ttf",
        "SourceHanSerifJP-Regular.ttf",
        "SourceHanSansJP-Bold.ttf",
        "SourceHanSansJP-Regular.ttf",
    ],
    fallback=[
        GO_NOTO_KURRENT_REGULAR,
        GO_NOTO_KURRENT_BOLD,
    ],
    base=[SOURCE_HAN_SANS_CN_REGULAR],
)

# English/Western
EN_FONT_FAMILY = FontFamilyConfig(
    script=[
        "NotoSans-Italic.ttf",
        "NotoSans-BoldItalic.ttf",
        "NotoSerif-Italic.ttf",
        "NotoSerif-BoldItalic.ttf",
    ],
    normal=[
        "NotoSerif-Regular.ttf",
        "NotoSerif-Bold.ttf",
        "NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf",
    ],
    fallback=[
        GO_NOTO_KURRENT_REGULAR,
        GO_NOTO_KURRENT_BOLD,
    ],
    base=[
        "NotoSans-Regular.ttf",
    ],
)

# Mapping of language codes to font families
ALL_FONT_FAMILIES: dict[str, FontFamilyConfig] = {
    "CN": CN_FONT_FAMILY,
    "TW": TW_FONT_FAMILY,
    "HK": HK_FONT_FAMILY,
    "KR": KR_FONT_FAMILY,
    "JP": JP_FONT_FAMILY,
    "JA": JP_FONT_FAMILY,  # Alternative code for Japanese
    "EN": EN_FONT_FAMILY,
}


def get_font_family(lang_code: str) -> FontFamilyConfig:
    """Get the appropriate font family for a language code.

    Args:
        lang_code: Language code (e.g., "CN", "JA", "EN")

    Returns:
        FontFamilyConfig for the language

    Raises:
        ValueError: If no matching font family is found
    """
    lang_upper = lang_code.upper()

    # Check for specific language codes
    for key in ["KR", "JP", "JA", "HK", "TW", "EN", "CN"]:
        if key in lang_upper:
            return ALL_FONT_FAMILIES[key]

    # Default to English
    return ALL_FONT_FAMILIES["EN"]


def _merge_fonts(family1: FontFamilyConfig, family2: FontFamilyConfig, added_fonts: set[str]) -> None:
    for category in ["script", "normal", "fallback", "base"]:
        fonts_list = getattr(family1, category)
        for font in getattr(family2, category):
            if font not in added_fonts:
                fonts_list.append(font)
                added_fonts.add(font)


def _add_fallback_to_font_families() -> None:
    """Add fonts from other families as fallbacks to each family."""
    for lang1, family1 in ALL_FONT_FAMILIES.items():
        added_fonts: set[str] = set(family1.all_fonts())
        for lang2, family2 in ALL_FONT_FAMILIES.items():
            if lang1 != lang2:
                _merge_fonts(family1, family2, added_fonts)


# Initialize fallback fonts
_add_fallback_to_font_families()
