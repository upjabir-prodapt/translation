import enum
import functools
import logging
import re
from pathlib import Path

import pymupdf

from src.worker.doctranslator.format.pdf.document_il import PdfFont
from src.worker.doctranslator.format.pdf.document_il import il_version_1
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.loaders import assets
from src.worker.loaders.exceptions import MetadataNotFoundError

logger = logging.getLogger(__name__)


class PrimaryFontFamily(enum.IntEnum):
    SERIF = 1
    SANS_SERIF = 2
    SCRIPT = 3
    NONE = 4

    @classmethod
    def from_str(cls, value: str):
        if value == "serif":
            return cls.SERIF
        elif value == "sans-serif":
            return cls.SANS_SERIF
        elif value == "script":
            return cls.SCRIPT
        else:
            return cls.NONE


class FontMapper:
    stage_name = "Add Fonts"

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        if translation_config.primary_font_family not in [
            None,
            "serif",
            "sans-serif",
            "script",
        ]:
            raise ValueError(
                f"primary_font_family must be one of None, 'serif', 'sans-serif', 'script'; got {translation_config.primary_font_family!r}"
            )
        self.primary_font_family = PrimaryFontFamily.from_str(
            translation_config.primary_font_family,
        )

        font_family = assets.get_font_family(translation_config.lang_out)
        self.font_file_names = []
        for k in (
            "normal",
            "script",
            "fallback",
            "base",
        ):
            self.font_file_names.extend(getattr(font_family, k))

        self.fonts: dict[str, pymupdf.Font] = {}
        self.fontid2fontpath: dict[str, Path] = {}
        self._load_fonts_from_file_names()

        if not self.fonts:
            raise RuntimeError("No usable fonts loaded from metadata/cache")

        loaded_font_ids = set(self.fontid2fontpath.keys())
        self._init_font_id_lists(font_family, loaded_font_ids)
        self._init_font_lookup_tables()

        self.has_char = functools.lru_cache(maxsize=10240, typed=True)(self.has_char)
        self.map_in_type = functools.lru_cache(maxsize=10240, typed=True)(
            self.map_in_type
        )

    def _load_single_font(self, font_file_name: str) -> None:
        """Load one font file into self.fonts and self.fontid2fontpath."""
        if font_file_name in self.fontid2fontpath:
            return
        try:
            font_path, font_metadata = assets.get_font_and_metadata(font_file_name)
        except MetadataNotFoundError as exc:
            logger.warning(
                f"Skipping unavailable font metadata for {font_file_name}: {exc}"
            )
            return
        pymupdf_font = pymupdf.Font(fontfile=str(font_path))
        pymupdf_font.has_glyph = functools.lru_cache(maxsize=10240, typed=True)(
            pymupdf_font.has_glyph,
        )
        pymupdf_font.char_lengths = functools.lru_cache(maxsize=10240, typed=True)(
            pymupdf_font.char_lengths,
        )
        self.fonts[font_file_name] = pymupdf_font
        self.fontid2fontpath[font_file_name] = font_path
        self.fonts[font_file_name].font_id = font_file_name
        self.fonts[font_file_name].font_path = font_path
        ascent = font_metadata.get("ascent")
        if ascent is None:
            ascent = pymupdf_font.ascender
        descent = font_metadata.get("descent")
        if descent is None:
            descent = pymupdf_font.descender
        encoding_length = font_metadata.get("encoding_length")
        if encoding_length is None:
            # DocTranslator glyph hex width; 2 matches most bundled CJK / ToUnicode layouts.
            encoding_length = 2
        self.fonts[font_file_name].ascent_fontmap = ascent
        self.fonts[font_file_name].descent_fontmap = descent
        self.fonts[font_file_name].encoding_length = encoding_length

    def _load_fonts_from_file_names(self) -> None:
        """Iterate self.font_file_names and load each font."""
        for font_file_name in self.font_file_names:
            self._load_single_font(font_file_name)

    def _init_font_id_lists(self, font_family, loaded_font_ids: set) -> None:
        """Populate normal/script/fallback/base font ID lists from *font_family*."""
        self.normal_font_ids = [
            font_id for font_id in font_family.normal if font_id in loaded_font_ids
        ]
        self.script_font_ids = [
            font_id for font_id in font_family.script if font_id in loaded_font_ids
        ]
        self.fallback_font_ids = [
            font_id for font_id in font_family.fallback if font_id in loaded_font_ids
        ]
        self.base_font_ids = [
            font_id for font_id in font_family.base if font_id in loaded_font_ids
        ]

        if not self.normal_font_ids:
            self.normal_font_ids = list(loaded_font_ids)
        if not self.fallback_font_ids:
            self.fallback_font_ids = list(loaded_font_ids)
        if not self.base_font_ids:
            # Use first loaded font as base when configured base font is unavailable.
            self.base_font_ids = [next(iter(loaded_font_ids))]

    def _init_font_lookup_tables(self) -> None:
        """Build fontid2fontpath 'base' alias and all convenience font lists."""
        self.fontid2fontpath["base"] = self.fontid2fontpath[self.base_font_ids[0]]

        self.fontid2font: dict[str, pymupdf.Font] = {
            f.font_id: f for f in self.fonts.values()
        }
        self.fontid2font["base"] = self.fontid2font[self.base_font_ids[0]]

        self.normal_fonts: list[pymupdf.Font] = [
            self.fontid2font[font_id] for font_id in self.normal_font_ids
        ]
        self.script_fonts: list[pymupdf.Font] = [
            self.fontid2font[font_id] for font_id in self.script_font_ids
        ]
        self.fallback_fonts: list[pymupdf.Font] = [
            self.fontid2font[font_id] for font_id in self.fallback_font_ids
        ]

        self.base_font = self.fontid2font["base"]

        self.type2font: dict[str, list[pymupdf.Font]] = {
            "normal": self.normal_fonts,
            "script": self.script_fonts,
            "fallback": self.fallback_fonts,
            "base": [self.base_font],
        }

    def has_char(self, char_unicode: str):
        if len(char_unicode) != 1:
            return False
        current_char = ord(char_unicode)
        for font in self.fonts.values():
            if font.has_glyph(current_char):
                return True
        return False

    def map_in_type(
        self,
        bold: bool,
        italic: bool,
        monospaced: bool,
        serif: bool,
        char_unicode: str,
        font_type: str,
    ):
        if font_type == "script" and not italic:
            return None
        current_char = ord(char_unicode)
        for font in self.type2font[font_type]:
            if not font.has_glyph(current_char):
                continue
            if bool(bold) != bool(font.is_bold):
                continue
            # 不知道什么原因，思源黑体的 serif 属性为 1，先 workaround
            if bool(serif) and "serif" not in font.font_id.lower():
                continue
            if not bool(serif) and "serif" in font.font_id.lower():
                continue
            return font

        return None

    def _extract_font_attributes(self, original_font: PdfFont, char_unicode: str):
        """Return (bold, italic, monospaced, serif) from *original_font*, or None on error."""
        if isinstance(original_font, pymupdf.Font):
            return (
                original_font.is_bold,
                original_font.is_italic,
                original_font.is_monospaced,
                original_font.is_serif,
            )
        if isinstance(original_font, PdfFont):
            return (
                original_font.bold,
                original_font.italic,
                original_font.monospace,
                original_font.serif,
            )
        logger.error(
            f"Unknown font type: {type(original_font)}. "
            f"Original font: {original_font}. "
            f"Char unicode: {char_unicode}. ",
        )
        return None

    def _apply_primary_family_override(self, bold, italic, monospaced, serif):
        """Adjust (bold, italic, monospaced, serif) for the configured primary font family."""
        if self.primary_font_family == PrimaryFontFamily.SERIF:
            serif = True
        elif self.primary_font_family == PrimaryFontFamily.SANS_SERIF:
            serif = False
        elif self.primary_font_family == PrimaryFontFamily.SCRIPT:
            serif = False
            italic = True
        return bold, italic, monospaced, serif

    def map(self, original_font: PdfFont, char_unicode: str):
        attrs = self._extract_font_attributes(original_font, char_unicode)
        if attrs is None:
            return None
        bold, italic, monospaced, serif = self._apply_primary_family_override(*attrs)
        current_char = ord(char_unicode)

        script_font_map_result = self.map_in_type(
            bold, italic, monospaced, serif, char_unicode, "script"
        )
        if script_font_map_result:
            return script_font_map_result

        for script_font in self.script_fonts:
            if italic and script_font.has_glyph(current_char):
                return script_font

        normal_font_map_result = self.map_in_type(
            bold, italic, monospaced, serif, char_unicode, "normal"
        )
        if normal_font_map_result is not None:
            return normal_font_map_result

        fallback_font_map_result = self.map_in_type(
            bold, italic, monospaced, serif, char_unicode, "fallback"
        )
        if fallback_font_map_result is not None:
            return fallback_font_map_result

        for font in self.fallback_fonts:
            if font.has_glyph(current_char):
                return font

        logger.warning(
            f"Can't find font for {char_unicode}({current_char}). "
            f"Original font: {original_font.name}[{original_font.font_id}]. "
            f"Char unicode: {char_unicode}. ",
        )
        return None

    def _collect_font_ids_from_page(self, page, result: set) -> None:
        """Add all font IDs referenced on *page* into *result*."""
        for char in page.pdf_character:
            if char.pdf_style and char.pdf_style.font_id:
                result.add(char.pdf_style.font_id)
        for para in page.pdf_paragraph:
            for comp in para.pdf_paragraph_composition:
                if char := comp.pdf_character:
                    if char.pdf_style and char.pdf_style.font_id:
                        result.add(char.pdf_style.font_id)

    def get_used_font_ids(self, il: il_version_1.Document) -> set[str]:
        result = set()
        for page in il.page:
            self._collect_font_ids_from_page(page, result)
        return result

    def _insert_fonts_into_dict_xref(
        self,
        doc_zh: pymupdf.Document,
        xref: int,
        target_key_prefix: str,
        font_list: list,
        font_id: dict,
    ) -> None:
        """Write missing font entries into an xref that is known to be a dict."""
        for font in font_list:
            target_key = f"{target_key_prefix}{font[0]}"
            font_exist = doc_zh.xref_get_key(xref, target_key)
            if font_exist[0] == "null":
                doc_zh.xref_set_key(xref, target_key, f"{font_id[font[0]]} 0 R")

    def _register_font_in_xref(
        self,
        doc_zh: pymupdf.Document,
        xref: int,
        font_list: list,
        font_id: dict,
    ) -> None:
        """Try to add all fonts in *font_list* to the font resources of *xref*."""
        for label in ["Resources/", ""]:  # may be based on xobj resources
            try:
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                if font_res is None:
                    continue
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""
                if font_res[0] == "dict":
                    self._insert_fonts_into_dict_xref(
                        doc_zh, xref, target_key_prefix, font_list, font_id
                    )
            except Exception:
                pass

    def _build_pdf_font(self, font_name: str, font_id: dict) -> "il_version_1.PdfFont":
        """Build a PdfFont IL object for *font_name* using cached mupdf metadata."""
        if font_name not in self.fontid2font:
            raise KeyError(f"Font {font_name} not found")
        mupdf_font = self.fontid2font[font_name]
        return il_version_1.PdfFont(
            name=font_name,
            xref_id=font_id[font_name],
            font_id=font_name,
            encoding_length=mupdf_font.encoding_length,
            bold=mupdf_font.is_bold,
            italic=mupdf_font.is_italic,
            monospace=mupdf_font.is_monospaced,
            serif=mupdf_font.is_serif,
            descent=mupdf_font.descent_fontmap,
            ascent=mupdf_font.ascent_fontmap,
        )

    def add_font(self, doc_zh: pymupdf.Document, il: il_version_1.Document):
        used_font_ids = self.get_used_font_ids(il)
        font_list = [
            (k, v) for k, v in self.fontid2fontpath.items() if k in used_font_ids
        ]

        font_id = {}
        xreflen = doc_zh.xref_length()
        total = xreflen - 1 + len(font_list) + len(il.page) + len(font_list)
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            if not il.page:
                pbar.advance(total)
                return
            for font in font_list:
                if font[0] in font_id:
                    continue
                font_id[font[0]] = doc_zh[0].insert_font(font[0], font[1])
                pbar.advance(1)
            for xref in range(1, xreflen):
                pbar.advance(1)
                self._register_font_in_xref(doc_zh, xref, font_list, font_id)

            # Build and attach PdfFont objects for all used fonts
            pdf_fonts = [
                self._build_pdf_font(font_name, font_id)
                for font_name, _ in font_list
                if pbar.advance(1) is None  # side-effect: advance progress
            ]

            for page in il.page:
                page.pdf_font.extend(pdf_fonts)
                for xobj in page.pdf_xobject:
                    xobj.pdf_font.extend(pdf_fonts)
                pbar.advance(1)
