from __future__ import annotations

import concurrent.futures
import copy
import logging
import re
import statistics
import threading
import unicodedata
from functools import cache

import pymupdf
import regex
from rtree import index

from src.config.constants import settings
from src.doctranslator.format.pdf.document_il import Box
from src.doctranslator.format.pdf.document_il import PdfCharacter
from src.doctranslator.format.pdf.document_il import PdfCurve
from src.doctranslator.format.pdf.document_il import PdfForm
from src.doctranslator.format.pdf.document_il import PdfFormula
from src.doctranslator.format.pdf.document_il import PdfParagraphComposition
from src.doctranslator.format.pdf.document_il import PdfStyle
from src.doctranslator.format.pdf.document_il import il_version_1
from src.doctranslator.format.pdf.document_il.utils.fontmap import FontMapper
from src.doctranslator.format.pdf.document_il.utils.formular_helper import (
    update_formula_data,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import box_to_tuple
from src.doctranslator.format.pdf.translation_config import TranslationConfig
from src.doctranslator.format.pdf.translation_config import WatermarkOutputMode

logger = logging.getLogger(__name__)

LINE_BREAK_REGEX = regex.compile(
    r"^["
    r"a-z"
    r"A-Z"
    r"0-9"
    r"\u00C0-\u00FF"  # Latin-1 Supplement
    r"\u0100-\u017F"  # Latin Extended A
    r"\u0180-\u024F"  # Latin Extended B
    r"\u1E00-\u1EFF"  # Latin Extended Additional
    r"\u2C60-\u2C7F"  # Latin Extended C
    r"\uA720-\uA7FF"  # Latin Extended D
    r"\uAB30-\uAB6F"  # Latin Extended E
    r"\u0250-\u02A0"  # IPA Extensions
    r"\u0400-\u04FF"  # Cyrillic
    r"\u0300-\u036F"  # Combining Diacritical Marks
    r"\u0500-\u052F"  # Cyrillic Supplement
    r"\u0370-\u03FF"  # Greek and Coptic
    r"\u2DE0-\u2DFF"  # Cyrillic Extended-A
    r"\uA650-\uA69F"  # Cyrillic Extended-B
    r"\u1200-\u137F"  # Ethiopic
    r"\u1380-\u139F"  # Ethiopic Supplement
    r"\u2D80-\u2DDF"  # Ethiopic Extended
    r"\uAB00-\uAB2F"  # Ethiopic Extended-A
    r"\U0001E7E0-\U0001E7FF"  # Ethiopic Extended-B
    r"\u0E80-\u0EFF"  # Lao
    r"\u0D00-\u0D7F"  # Malayalam
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0E00-\u0E7F"  # Thai
    r"\u1000-\u109F"  # Myanmar
    r"\uAA60-\uAA7F"  # Myanmar Extended-A
    r"\uA9E0-\uA9FF"  # Myanmar Extended-B
    r"\U000116D0-\U000116FF"  # Myanmar Extended-C
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0B00-\u0B7F"  # Oriya
    r"\u0530-\u058F"  # Armenian
    r"\u10A0-\u10FF"  # Georgian
    r"\u1C90-\u1CBF"  # Georgian Extended
    r"\u2D00-\u2D2F"  # Georgian Supplement
    r"\u1780-\u17FF"  # Khmer
    r"\u19E0-\u19FF"  # Khmer Symbols
    r"\U00010B00-\U00010B3F"  # Avestan
    r"\u1D00-\u1D7F"  # Phonetic Extensions
    r"\u1400-\u167F"  # Unified Canadian Aboriginal Syllabics
    r"\u0B00-\u0B7F"  # Oriya
    r"\u0780-\u07BF"  # Thaana
    r"\U0001E900-\U0001E95F"  # Adlam
    r"\u1C80-\u1C8F"  # Cyrillic Extended-C
    r"\U0001E030-\U0001E08F"  # Cyrillic Extended-D
    r"\uA000-\uA48F"  # Yi Syllables
    r"\uA490-\uA4CF"  # Yi Radicals
    r"'"
    r"-"  # Hyphen
    r"·"  # Middle Dot (U+00B7) For Català
    r"ʻ"  # Spacing Modifier Letters U+02BB
    r"]+$"
)


class TypesettingUnit:
    def __str__(self):
        return self.try_get_unicode() or ""

    def __init__(
        self,
        char: PdfCharacter | None = None,
        formular: PdfFormula | None = None,
        unicode: str | None = None,
        font: pymupdf.Font | None = None,
        original_font: il_version_1.PdfFont | None = None,
        font_size: float | None = None,
        style: PdfStyle | None = None,
        xobj_id: int | None = None,
        debug_info: bool = False,
    ):
        assert (char is not None) + (formular is not None) + (
            unicode is not None
        ) == 1, "Only one of chars and formular can be not None"
        self.char = char
        self.formular = formular
        self.unicode = unicode
        self.x = None
        self.y = None
        self.scale = None
        self.debug_info = debug_info

        # Cache variables
        self.box_cache: Box | None = None
        self.can_break_line_cache: bool | None = None
        self.is_cjk_char_cache: bool | None = None
        self.mixed_character_blacklist_cache: bool | None = None
        self.is_space_cache: bool | None = None
        self.is_hung_punctuation_cache: bool | None = None
        self.is_cannot_appear_in_line_end_punctuation_cache: bool | None = None
        self.can_passthrough_cache: bool | None = None
        self.width_cache: float | None = None
        self.height_cache: float | None = None

        self.font_size: float | None = None

        if unicode:
            assert font_size, "Font size must be provided when unicode is provided"
            assert style, "Style must be provided when unicode is provided"
            assert len(unicode) == 1, "Unicode must be a single character"
            assert xobj_id is not None, (
                "Xobj id must be provided when unicode is provided"
            )

            self.font = font
            if font is not None and hasattr(font, "font_id"):
                self.font_id = font.font_id
            else:
                self.font_id = "base"
            if original_font:
                self.original_font = original_font
            else:
                self.original_font = None

            self.font_size = font_size
            self.style = style
            self.xobj_id = xobj_id

    def try_resue_cache(self, old_tu: TypesettingUnit):
        if old_tu.is_cjk_char_cache is not None:
            self.is_cjk_char_cache = old_tu.is_cjk_char_cache

        if old_tu.can_break_line_cache is not None:
            self.can_break_line_cache = old_tu.can_break_line_cache

        if old_tu.is_space_cache is not None:
            self.is_space_cache = old_tu.is_space_cache

        if old_tu.is_hung_punctuation_cache is not None:
            self.is_hung_punctuation_cache = old_tu.is_hung_punctuation_cache

        if old_tu.is_cannot_appear_in_line_end_punctuation_cache is not None:
            self.is_cannot_appear_in_line_end_punctuation_cache = (
                old_tu.is_cannot_appear_in_line_end_punctuation_cache
            )

        if old_tu.can_passthrough_cache is not None:
            self.can_passthrough_cache = old_tu.can_passthrough_cache

        if old_tu.mixed_character_blacklist_cache is not None:
            self.mixed_character_blacklist_cache = (
                old_tu.mixed_character_blacklist_cache
            )

    def try_get_unicode(self) -> str | None:
        if self.char:
            return self.char.char_unicode
        elif self.formular:
            return None
        elif self.unicode:
            return self.unicode

    @property
    def mixed_character_blacklist(self):
        if self.mixed_character_blacklist_cache is None:
            self.mixed_character_blacklist_cache = self.calc_mixed_character_blacklist()

        return self.mixed_character_blacklist_cache

    def calc_mixed_character_blacklist(self):
        unicode = self.try_get_unicode()
        if unicode:
            return unicode in [
                "。",
                "，",
                "：",
                "？",
                "！",
            ]
        return False

    @property
    def can_break_line(self):
        if self.can_break_line_cache is None:
            self.can_break_line_cache = self.calc_can_break_line()

        return self.can_break_line_cache

    def calc_can_break_line(self):
        unicode = self.try_get_unicode()
        if not unicode:
            return True
        if LINE_BREAK_REGEX.match(unicode):
            return False
        return True

    @property
    def is_cjk_char(self):
        if self.is_cjk_char_cache is None:
            self.is_cjk_char_cache = self.calc_is_cjk_char()

        return self.is_cjk_char_cache

    _CJK_PUNCTUATION_SET = frozenset(
        [
            "（",
            "）",
            "【",
            "】",
            "《",
            "》",
            "〔",
            "〕",
            "〈",
            "〉",
            "〖",
            "〗",
            "「",
            "」",
            "『",
            "』",
            "、",
            "。",
            "：",
            "？",
            "！",
            "，",
        ]
    )

    def _is_cjk_by_punctuation_list(self, unicode: str) -> bool:
        """Return True if the character is in the CJK punctuation list."""
        return unicode in self._CJK_PUNCTUATION_SET

    def _is_cjk_by_regex(self, unicode: str) -> bool:
        """Return True if the character matches CJK Unicode ranges via regex."""
        return bool(
            re.match(
                r"^["
                r"　-〿"  # CJK Symbols and Punctuation
                r"぀-ゟ"  # Hiragana
                r"゠-ヿ"  # Katakana
                r"㄀-ㄯ"  # Bopomofo
                r"가-힯"  # Hangul Syllables
                r"ᄀ-ᇿ"  # Hangul Jamo
                r"㄰-㆏"  # Hangul Compatibility Jamo
                r"ꥠ-꥿"  # Hangul Jamo Extended-A
                r"ힰ-퟿"  # Hangul Jamo Extended-B
                r"㆐-㆟"  # Kanbun
                r"㈀-㋿"  # Enclosed CJK Letters and Months
                r"㌀-㏿"  # CJK Compatibility
                r"︰-﹏"  # CJK Compatibility Forms
                r"一-鿿"  # CJK Unified Ideographs
                r"⺀-⻿"  # CJK Radicals Supplement
                r"㇀-㇯"  # CJK Strokes
                r"⼀-⿟"  # Kangxi Radicals
                r"︐-︟"  # Vertical Forms
                r"]+$",
                unicode,
            )
        )

    def _is_cjk_by_unicode_name(self, unicode: str) -> bool:
        """Return True if the character's Unicode name indicates CJK or FULLWIDTH."""
        try:
            unicodedata_name = unicodedata.name(unicode)
            return (
                "CJK UNIFIED IDEOGRAPH" in unicodedata_name
                or "FULLWIDTH" in unicodedata_name
            )
        except ValueError:
            return False

    def calc_is_cjk_char(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        if not unicode:
            return False
        if "(cid" in unicode:
            return False
        if len(unicode) > 1:
            return False
        assert len(unicode) == 1, "Unicode must be a single character"
        if self._is_cjk_by_punctuation_list(unicode):
            return True
        if self._is_cjk_by_regex(unicode):
            return True
        return self._is_cjk_by_unicode_name(unicode)

    @property
    def is_space(self):
        if self.is_space_cache is None:
            self.is_space_cache = self.calc_is_space()

        return self.is_space_cache

    def calc_is_space(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        return unicode == " "

    @property
    def is_hung_punctuation(self):
        if self.is_hung_punctuation_cache is None:
            self.is_hung_punctuation_cache = self.calc_is_hung_punctuation()

        return self.is_hung_punctuation_cache

    def calc_is_hung_punctuation(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()

        if unicode:
            return unicode in [
                ",",
                ".",
                ":",
                ";",
                "?",
                "!",
                "，",
                "。",
                "．",
                "、",
                "：",
                "；",
                "！",
                "‼",
                "？",
                "⁇",
                """,  # right double quotation mark
                "'",  # right single quotation mark
                "」",
                "』",
                ")",
                "]",
                "}",
                "）",
                "〕",
                "〉",
                "】",
                "〗",
                "］",
                "｝",
                "》",
                "～",
                "-",
                "–",
                "—",
                "·",
                "・",
                "‧",
                "/",
                "／",
                "⁄",
            ]
        return False

    @property
    def is_cannot_appear_in_line_end_punctuation(self):
        if self.is_cannot_appear_in_line_end_punctuation_cache is None:
            self.is_cannot_appear_in_line_end_punctuation_cache = (
                self.calc_is_cannot_appear_in_line_end_punctuation()
            )

        return self.is_cannot_appear_in_line_end_punctuation_cache

    def calc_is_cannot_appear_in_line_end_punctuation(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        if not unicode:
            return False
        return unicode in [
            """,  # left double quotation mark
            "'",  # left single quotation mark
            "「",
            "『",
            "(",
            "[",
            "{",
            "（",
            "〔",
            "〈",
            "《",
            "〖",
            "〘",
            "〚",
        ]

    def passthrough(
        self,
    ) -> tuple[list[PdfCharacter], list[PdfCurve], list[PdfForm]]:
        if self.char:
            return [self.char], [], []
        elif self.formular:
            return (
                self.formular.pdf_character,
                self.formular.pdf_curve,
                self.formular.pdf_form,
            )
        elif self.unicode:
            logger.error(f"Cannot passthrough unicode. TypesettingUnit: {self}. ")
            logger.error(f"Cannot passthrough unicode. TypesettingUnit: {self}. ")
            return [], [], []

    @property
    def can_passthrough(self):
        if self.can_passthrough_cache is None:
            self.can_passthrough_cache = self.calc_can_passthrough()

        return self.can_passthrough_cache

    def calc_can_passthrough(self):
        return self.unicode is None

    def calculate_box(self):
        if self.char:
            if self.char.box is None:
                return Box(0, 0, 0, 0)
            box = copy.deepcopy(self.char.box)
            if self.char.visual_bbox and self.char.visual_bbox.box:
                box.y = self.char.visual_bbox.box.y
                box.y2 = self.char.visual_bbox.box.y2

            return box
        elif self.formular:
            return self.formular.box
        elif self.unicode:
            char_width = self.font.char_lengths(self.unicode, self.font_size)[0]
            if self.x is None or self.y is None or self.scale is None:
                return Box(0, 0, char_width, self.font_size)
            return Box(self.x, self.y, self.x + char_width, self.y + self.font_size)

    @property
    def box(self):
        if not self.box_cache:
            self.box_cache = self.calculate_box()

        return self.box_cache

    @property
    def width(self):
        if self.width_cache is None:
            self.width_cache = self.calc_width()

        return self.width_cache

    def calc_width(self):
        box = self.box
        return box.x2 - box.x

    @property
    def height(self):
        if self.height_cache is None:
            self.height_cache = self.calc_height()

        return self.height_cache

    def calc_height(self):
        box = self.box
        return box.y2 - box.y

    def _relocate_char(self, x: float, y: float, scale: float) -> TypesettingUnit:
        """Helper: relocate a char-based TypesettingUnit."""
        new_char = PdfCharacter(
            pdf_character_id=self.char.pdf_character_id,
            char_unicode=self.char.char_unicode,
            box=Box(
                x=x,
                y=y,
                x2=x + self.width * scale,
                y2=y + self.height * scale,
            ),
            pdf_style=PdfStyle(
                font_id=self.char.pdf_style.font_id,
                font_size=self.char.pdf_style.font_size * scale,
                graphic_state=self.char.pdf_style.graphic_state,
            ),
            scale=scale,
            vertical=self.char.vertical,
            advance=self.char.advance * scale if self.char.advance else None,
            debug_info=self.debug_info,
            xobj_id=self.char.xobj_id,
        )
        new_tu = TypesettingUnit(char=new_char)
        new_tu.try_resue_cache(self)
        return new_tu

    def _relocate_formula_char(
        self, char, x: float, y: float, scale: float, min_x: float, min_y: float
    ) -> PdfCharacter:
        """Helper: build a relocated PdfCharacter for a formula character."""
        rel_x = char.box.x - min_x
        rel_y = char.box.y - min_y
        visual_rel_x = char.visual_bbox.box.x - min_x
        visual_rel_y = char.visual_bbox.box.y - min_y
        return PdfCharacter(
            pdf_character_id=char.pdf_character_id,
            char_unicode=char.char_unicode,
            box=Box(
                x=x + (rel_x + self.formular.x_offset) * scale,
                y=y + (rel_y + self.formular.y_offset) * scale,
                x2=x
                + (rel_x + (char.box.x2 - char.box.x) + self.formular.x_offset) * scale,
                y2=y
                + (rel_y + (char.box.y2 - char.box.y) + self.formular.y_offset) * scale,
            ),
            visual_bbox=il_version_1.VisualBbox(
                box=Box(
                    x=x + (visual_rel_x + self.formular.x_offset) * scale,
                    y=y + (visual_rel_y + self.formular.y_offset) * scale,
                    x2=x
                    + (
                        visual_rel_x
                        + (char.visual_bbox.box.x2 - char.visual_bbox.box.x)
                        + self.formular.x_offset
                    )
                    * scale,
                    y2=y
                    + (
                        visual_rel_y
                        + (char.visual_bbox.box.y2 - char.visual_bbox.box.y)
                        + self.formular.y_offset
                    )
                    * scale,
                ),
            ),
            pdf_style=PdfStyle(
                font_id=char.pdf_style.font_id,
                font_size=char.pdf_style.font_size * scale,
                graphic_state=char.pdf_style.graphic_state,
            ),
            scale=scale,
            vertical=char.vertical,
            advance=char.advance * scale if char.advance else None,
            xobj_id=char.xobj_id,
        )

    def _relocate_formula(self, x: float, y: float, scale: float) -> TypesettingUnit:
        """Helper: relocate a formula-based TypesettingUnit."""
        min_x = self.formular.box.x
        min_y = self.formular.box.y
        new_chars = [
            self._relocate_formula_char(char, x, y, scale, min_x, min_y)
            for char in self.formular.pdf_character
        ]

        bbox_min_x = min(char.visual_bbox.box.x for char in new_chars)
        bbox_min_y = min(char.visual_bbox.box.y for char in new_chars)
        bbox_max_x = max(char.visual_bbox.box.x2 for char in new_chars)
        bbox_max_y = max(char.visual_bbox.box.y2 for char in new_chars)

        new_formula = PdfFormula(
            box=Box(x=bbox_min_x, y=bbox_min_y, x2=bbox_max_x, y2=bbox_max_y),
            pdf_character=new_chars,
            x_offset=self.formular.x_offset * scale,
            y_offset=self.formular.y_offset * scale,
            x_advance=self.formular.x_advance * scale,
        )

        new_formula.pdf_curve = [
            self._transform_curve_for_relocation(
                curve, self.formular.box.x, self.formular.box.y, x, y, scale
            )
            for curve in self.formular.pdf_curve
        ]
        new_formula.pdf_form = [
            self._transform_form_for_relocation(
                form, self.formular.box.x, self.formular.box.y, x, y, scale
            )
            for form in self.formular.pdf_form
        ]

        update_formula_data(new_formula)
        new_tu = TypesettingUnit(formular=new_formula)
        new_tu.try_resue_cache(self)
        return new_tu

    def _relocate_unicode(self, x: float, y: float, scale: float) -> TypesettingUnit:
        """Helper: relocate a unicode-based TypesettingUnit."""
        new_unit = TypesettingUnit(
            unicode=self.unicode,
            font=self.font,
            original_font=self.original_font,
            font_size=self.font_size * scale,
            style=self.style,
            xobj_id=self.xobj_id,
            debug_info=self.debug_info,
        )
        new_unit.x = x
        new_unit.y = y
        new_unit.scale = scale
        new_unit.try_resue_cache(self)
        return new_unit

    def relocate(
        self,
        x: float,
        y: float,
        scale: float,
    ) -> TypesettingUnit:
        """重定位并缩放排版单元

        Args:
            x: 新的 x 坐标
            y: 新的 y 坐标
            scale: 缩放因子

        Returns:
            新的排版单元
        """
        if self.char:
            return self._relocate_char(x, y, scale)
        elif self.formular:
            return self._relocate_formula(x, y, scale)
        elif self.unicode:
            return self._relocate_unicode(x, y, scale)

    def _transform_curve_for_relocation(
        self,
        curve,
        original_formula_x: float,
        original_formula_y: float,
        new_x: float,
        new_y: float,
        scale: float,
    ):
        """Transform a curve for formula relocation."""
        import copy

        new_curve = copy.deepcopy(curve)

        if new_curve.box:
            # Calculate relative position to formula's original position (same as chars)
            rel_x = new_curve.box.x - original_formula_x
            rel_y = new_curve.box.y - original_formula_y

            # Apply same transformation as characters
            new_curve.box = Box(
                x=new_x + (rel_x + self.formular.x_offset) * scale,
                y=new_y + (rel_y + self.formular.y_offset) * scale,
                x2=new_x
                + (
                    rel_x
                    + (new_curve.box.x2 - new_curve.box.x)
                    + self.formular.x_offset
                )
                * scale,
                y2=new_y
                + (
                    rel_y
                    + (new_curve.box.y2 - new_curve.box.y)
                    + self.formular.y_offset
                )
                * scale,
            )

        # Set relocation transform instead of modifying original CTM
        translation_x = (
            new_x + self.formular.x_offset * scale - original_formula_x * scale
        )
        translation_y = (
            new_y + self.formular.y_offset * scale - original_formula_y * scale
        )

        # Create relocation transformation matrix
        from src.doctranslator.format.pdf.document_il.utils.matrix_helper import (
            create_translation_and_scale_matrix,
        )

        relocation_matrix = create_translation_and_scale_matrix(
            translation_x, translation_y, scale
        )
        new_curve.relocation_transform = list(relocation_matrix)

        return new_curve

    def _transform_form_for_relocation(
        self,
        form,
        original_formula_x: float,
        original_formula_y: float,
        new_x: float,
        new_y: float,
        scale: float,
    ):
        """Transform a form for formula relocation."""
        import copy

        new_form = copy.deepcopy(form)

        if new_form.box:
            # Calculate relative position to formula's original position (same as chars)
            rel_x = new_form.box.x - original_formula_x
            rel_y = new_form.box.y - original_formula_y

            # Apply same transformation as characters
            new_form.box = Box(
                x=new_x + (rel_x + self.formular.x_offset) * scale,
                y=new_y + (rel_y + self.formular.y_offset) * scale,
                x2=new_x
                + (rel_x + (new_form.box.x2 - new_form.box.x) + self.formular.x_offset)
                * scale,
                y2=new_y
                + (rel_y + (new_form.box.y2 - new_form.box.y) + self.formular.y_offset)
                * scale,
            )

        # Set relocation transform instead of modifying original matrices
        translation_x = (
            new_x + self.formular.x_offset * scale - original_formula_x * scale
        )
        translation_y = (
            new_y + self.formular.y_offset * scale - original_formula_y * scale
        )

        # Create relocation transformation matrix
        from src.doctranslator.format.pdf.document_il.utils.matrix_helper import (
            create_translation_and_scale_matrix,
        )

        relocation_matrix = create_translation_and_scale_matrix(
            translation_x, translation_y, scale
        )
        new_form.relocation_transform = list(relocation_matrix)

        return new_form

    def render(
        self,
    ) -> tuple[list[PdfCharacter], list[PdfCurve], list[PdfForm]]:
        """渲染排版单元为 PdfCharacter 列表

        Returns:
            PdfCharacter 列表
        """
        if self.can_passthrough:
            return self.passthrough()
        elif self.unicode:
            assert self.x is not None, (
                "x position must be set, should be set by `relocate`"
            )
            assert self.y is not None, (
                "y position must be set, should be set by `relocate`"
            )
            assert self.scale is not None, (
                "scale must be set, should be set by `relocate`"
            )
            x = self.x
            y = self.y

            # 计算字符宽度
            char_width = self.width

            new_char = PdfCharacter(
                pdf_character_id=self.font.has_glyph(ord(self.unicode)),
                char_unicode=self.unicode,
                box=Box(
                    x=x,  # 使用存储的位置
                    y=y,
                    x2=x + char_width,
                    y2=y + self.font_size,
                ),
                pdf_style=PdfStyle(
                    font_id=self.font_id,
                    font_size=self.font_size,
                    graphic_state=self.style.graphic_state,
                ),
                scale=self.scale,
                vertical=False,
                advance=char_width,
                xobj_id=self.xobj_id,
                debug_info=self.debug_info,
            )
            return [new_char], [], []
        else:
            logger.error(f"Unknown typesetting unit. TypesettingUnit: {self}. ")
            logger.error(f"Unknown typesetting unit. TypesettingUnit: {self}. ")
            return [], [], []


class Typesetting:
    stage_name = "Typesetting"

    def __init__(self, translation_config: TranslationConfig):
        self.font_mapper = FontMapper(translation_config)
        self.translation_config = translation_config
        self.lang_code = self.translation_config.lang_out.upper()
        self.is_cjk = (
            # Why zh-CN/zh-HK/zh-TW here but not zh-Hans and so on?
            # See https://funstory-ai.github.io/BabelDOC/supported_languages/
            ("ZH" in self.lang_code)  # C
            or ("JA" in self.lang_code)
            or ("JP" in self.lang_code)  # J
            or ("KR" in self.lang_code)  # K
            or ("CN" in self.lang_code)
            or ("HK" in self.lang_code)
            or ("TW" in self.lang_code)
        )

    def _build_page_fonts(
        self, page: il_version_1.Page
    ) -> dict[str | int, il_version_1.PdfFont | dict[str, il_version_1.PdfFont]]:
        """Build the combined font lookup map for a page (including xobject fonts)."""
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ] = {f.font_id: f for f in page.pdf_font if f.font_id}
        page_fonts = {f.font_id: f for f in page.pdf_font if f.font_id}
        for k, v in self.font_mapper.fontid2font.items():
            fonts[k] = v
        for xobj in page.pdf_xobject:
            if xobj.xobj_id is not None:
                fonts[xobj.xobj_id] = page_fonts.copy()
                for font in xobj.pdf_font:
                    if (
                        xobj.xobj_id in fonts
                        and isinstance(fonts[xobj.xobj_id], dict)
                        and font.font_id
                    ):
                        fonts[xobj.xobj_id][font.font_id] = font
        return fonts

    def _compute_paragraph_optimal_scale(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        fonts,
    ) -> tuple[float, int]:
        """Compute the optimal scale for a paragraph and return (scale, unit_count)."""
        typesetting_units = self.create_typesetting_units(paragraph, fonts)
        unit_count = len(typesetting_units)
        for unit in typesetting_units:
            if unit.formular:
                unit_count += len(unit.formular.pdf_character) - 1
        if all(unit.can_passthrough for unit in typesetting_units):
            return 1.0, unit_count
        return self._get_optimal_scale(paragraph, page, typesetting_units), unit_count

    def _preprocess_page(
        self,
        page: il_version_1.Page,
        pbar_lock: threading.Lock,
        pbar,
    ) -> tuple[list[float], list[il_version_1.PdfParagraph]]:
        """Compute optimal scales for all paragraphs on a single page."""
        page_scales: list[float] = []
        page_paragraphs: list[il_version_1.PdfParagraph] = []
        fonts = self._build_page_fonts(page)

        for paragraph in page.pdf_paragraph:
            page_paragraphs.append(paragraph)
            unit_count = 0
            try:
                optimal_scale, unit_count = self._compute_paragraph_optimal_scale(
                    paragraph, page, fonts
                )
                paragraph.optimal_scale = optimal_scale
            except Exception as e:
                logger.warning(f"预处理段落时出错：{e}")
                paragraph.optimal_scale = 1.0

            if paragraph.optimal_scale is not None:
                page_scales.extend([paragraph.optimal_scale] * unit_count)

        with pbar_lock:
            pbar.advance()
        return page_scales, page_paragraphs

    def _cap_scales_at_mode(
        self,
        all_scales: list[float],
        all_paragraphs: list[il_version_1.PdfParagraph],
    ):
        """Clamp every paragraph's optimal_scale down to the modal scale value."""
        if not all_scales:
            logger.error(
                "document_scales is empty, there seems no paragraph in this PDF"
            )
            return
        try:
            modes = statistics.multimode(all_scales)
            mode_scale = min(modes)
        except statistics.StatisticsError:
            logger.warning(
                "Could not find a mode for paragraph scales. Falling back to median."
            )
            mode_scale = statistics.median(all_scales)
        for paragraph in all_paragraphs:
            if (
                paragraph.optimal_scale is not None
                and paragraph.optimal_scale > mode_scale
            ):
                paragraph.optimal_scale = mode_scale

    def preprocess_document(self, document: il_version_1.Document, pbar):
        """预处理文档，获取每个段落的最优缩放因子，不执行实际排版"""
        all_scales: list[float] = []
        all_paragraphs: list[il_version_1.PdfParagraph] = []
        scale_lock = threading.Lock()
        pbar_lock = threading.Lock()

        max_workers = max(1, int(settings.TYPESETTING_MAX_WORKERS))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._preprocess_page, page, pbar_lock, pbar)
                for page in document.page
            ]
            for future in concurrent.futures.as_completed(futures):
                page_scales, page_paragraphs = future.result()
                with scale_lock:
                    all_scales.extend(page_scales)
                    all_paragraphs.extend(page_paragraphs)

        self._cap_scales_at_mode(all_scales, all_paragraphs)

    def _apply_typeset_units_to_paragraph(
        self,
        typeset_units: list[TypesettingUnit],
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        scale: float,
    ) -> list[TypesettingUnit]:
        """Write the rendered typeset units back into the paragraph and page."""
        paragraph.scale = scale
        paragraph.pdf_paragraph_composition = []
        for unit in typeset_units:
            chars, curves, forms = unit.render()
            for char in chars:
                paragraph.pdf_paragraph_composition.append(
                    PdfParagraphComposition(pdf_character=char),
                )
            for curve in curves:
                page.pdf_curve.append(curve)
            for form in forms:
                page.pdf_form.append(form)
        return typeset_units

    def _try_expand_box_downward(
        self,
        box: Box,
        page: il_version_1.Page,
        paragraph: il_version_1.PdfParagraph,
        apply_layout: bool,
    ) -> tuple[Box, bool]:
        """Attempt to expand the layout box downward. Returns (new_box, expanded)."""
        try:
            min_y = self.get_max_bottom_space(box, page) + 2
            if min_y < box.y:
                expanded_box = Box(x=box.x, y=min_y, x2=box.x2, y2=box.y2)
                if apply_layout:
                    paragraph.box = expanded_box
                return expanded_box, True
        except Exception:
            return box, False
        return box, False

    def _try_expand_box_rightward(
        self,
        box: Box,
        page: il_version_1.Page,
        paragraph: il_version_1.PdfParagraph,
        apply_layout: bool,
    ) -> tuple[Box, bool]:
        """Attempt to expand the layout box rightward. Returns (new_box, expanded)."""
        try:
            max_x = self.get_max_right_space(box, page) - 5
            if max_x > box.x2:
                expanded_box = Box(x=box.x, y=box.y, x2=max_x, y2=box.y2)
                if apply_layout:
                    paragraph.box = expanded_box
                return expanded_box, True
        except Exception:
            return box, False
        return box, False

    def _try_layout_at_scale(
        self,
        typesetting_units: list[TypesettingUnit],
        box: Box,
        scale: float,
        line_skip: float,
        paragraph: il_version_1.PdfParagraph,
        use_english_line_break: bool,
    ) -> tuple[list[TypesettingUnit] | None, bool]:
        """Attempt a single layout pass. Returns (typeset_units, all_fit) or (None, False)."""
        try:
            return self._layout_typesetting_units(
                typesetting_units,
                box,
                scale,
                line_skip,
                paragraph,
                use_english_line_break,
            )
        except Exception as e:
            logger.warning(
                f"Layout failed at scale {scale} for paragraph {getattr(paragraph, 'debug_id', '?')}: "
                f"{type(e).__name__}: {e}"
            )
            return None, False

    def _attempt_expand_space(
        self,
        scale: float,
        expand_space_flag: int,
        box: Box,
        page: il_version_1.Page,
        paragraph: il_version_1.PdfParagraph,
        apply_layout: bool,
    ) -> tuple[Box, int, float]:
        """Try to expand the layout box when scale drops below 0.7.

        Returns (new_box, new_expand_flag, new_scale).  new_scale may be reset
        to 1.0 when an expansion succeeds but the flag resets the loop.
        """
        if expand_space_flag == 0:
            box, expanded = self._try_expand_box_downward(
                box, page, paragraph, apply_layout
            )
            expand_space_flag = 1
            if expanded:
                return box, expand_space_flag, scale
        elif expand_space_flag == 1:
            box, expanded = self._try_expand_box_rightward(
                box, page, paragraph, apply_layout
            )
            expand_space_flag = 2
            if expanded:
                return box, expand_space_flag, scale

        if expand_space_flag < 2:
            scale = 1.0
        return box, expand_space_flag, scale

    def _find_optimal_scale_and_layout(  # NOSONAR - layout search has several PDF-specific fit branches
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        initial_scale: float = 1.0,
        use_english_line_break: bool = True,
        apply_layout: bool = False,
    ) -> tuple[float, list[TypesettingUnit] | None]:
        """查找最优缩放因子并可选择性地执行布局

        Args:
            paragraph: 段落对象
            page: 页面对象
            typesetting_units: 排版单元列表
            initial_scale: 初始缩放因子
            use_english_line_break: 是否使用英文换行规则
            apply_layout: 是否应用布局到 paragraph（True 时执行实际排版）

        Returns:
            tuple[float, list[TypesettingUnit] | None]: (最终缩放因子，排版后的单元列表或 None)
        """
        if not paragraph.box:
            return initial_scale, None

        box = paragraph.box
        scale = initial_scale
        line_skip = 1.50 if self.is_cjk else 1.3
        min_scale = 0.1
        expand_space_flag = 0
        final_typeset_units = None
        last_attempted_units: list[TypesettingUnit] | None = None

        while scale >= min_scale:
            typeset_units, all_units_fit = self._try_layout_at_scale(
                typesetting_units,
                box,
                scale,
                line_skip,
                paragraph,
                use_english_line_break,
            )

            if typeset_units is not None:
                last_attempted_units = typeset_units

            if all_units_fit and typeset_units is not None:
                if apply_layout:
                    final_typeset_units = self._apply_typeset_units_to_paragraph(
                        typeset_units, paragraph, page, scale
                    )
                return scale, final_typeset_units

            if not hasattr(paragraph, "debug_id") or not paragraph.debug_id:
                return scale, final_typeset_units

            scale = scale - 0.05 if scale > 0.6 else scale - 0.1

            if scale < 0.7:
                box, expand_space_flag, scale = self._attempt_expand_space(
                    scale, expand_space_flag, box, page, paragraph, apply_layout
                )

        # 如果仍然放不下，尝试去除英文换行限制
        if use_english_line_break:
            return self._find_optimal_scale_and_layout(
                paragraph,
                page,
                typesetting_units,
                initial_scale,
                use_english_line_break=False,
                apply_layout=apply_layout,
            )

        # Force-apply the last attempted layout at minimum scale so the paragraph
        # is never left with empty composition (which causes silent text loss).
        if apply_layout:
            if last_attempted_units is not None:
                final_typeset_units = self._apply_typeset_units_to_paragraph(
                    last_attempted_units, paragraph, page, min_scale
                )
            else:
                # Every layout attempt threw an exception. At minimum mark the paragraph
                # as processed so pdf_creater does not log a spurious error.
                paragraph.scale = min_scale

        return min_scale, final_typeset_units

    def _get_optimal_scale(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        use_english_line_break: bool = True,
    ) -> float:
        """获取段落的最优缩放因子，不执行实际排版"""
        scale, _ = self._find_optimal_scale_and_layout(
            paragraph,
            page,
            typesetting_units,
            1.0,
            use_english_line_break,
            apply_layout=False,
        )
        return scale

    def retypeset_with_precomputed_scale(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        precomputed_scale: float,
        use_english_line_break: bool = True,
    ):
        """使用预计算的缩放因子进行排版"""
        if not paragraph.box:
            return

        # 使用通用方法进行排版，传入预计算的缩放因子作为初始值
        self._find_optimal_scale_and_layout(
            paragraph,
            page,
            typesetting_units,
            precomputed_scale,
            use_english_line_break,
            apply_layout=True,
        )

    def typesetting_document(self, document: il_version_1.Document):
        # 原有的排版逻辑
        if self.translation_config.progress_monitor:
            with self.translation_config.progress_monitor.stage_start(
                self.stage_name,
                len(document.page) * 2,
            ) as pbar:
                # 预处理：获取所有段落的最优缩放因子
                self.preprocess_document(document, pbar)
                pbar_lock = threading.Lock()
                max_workers = max(1, int(settings.TYPESETTING_MAX_WORKERS))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = [
                        executor.submit(self.render_page, page)
                        for page in document.page
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        self.translation_config.raise_if_cancelled()
                        future.result()
                        with pbar_lock:
                            pbar.advance()
        else:
            max_workers = max(1, int(settings.TYPESETTING_MAX_WORKERS))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(self.render_page, page) for page in document.page
                ]
                for future in concurrent.futures.as_completed(futures):
                    self.translation_config.raise_if_cancelled()
                    future.result()

    def _adjust_paragraph_positions(self, page: il_version_1.Page):
        """Push overlapping paragraphs apart so they don't collide vertically."""
        para_index = index.Index()
        para_map = {}
        valid_paras = [
            p
            for p in page.pdf_paragraph
            if p.box
            and all(c is not None for c in [p.box.x, p.box.y, p.box.x2, p.box.y2])
        ]
        for i, para in enumerate(valid_paras):
            para_map[i] = para
            para_index.insert(i, box_to_tuple(para.box))

        for i, p_upper in para_map.items():
            if not (p_upper.box and p_upper.box.y is not None):
                continue
            para_height = p_upper.box.y2 - p_upper.box.y
            required_gap = 0.5 if para_height < 36 else 3
            self._resolve_paragraph_overlap(
                i, p_upper, required_gap, para_index, para_map
            )

    def _resolve_paragraph_overlap(
        self, idx, p_upper, required_gap, para_index, para_map
    ):
        """Shift p_upper down if any lower paragraph overlaps within required_gap."""
        check_area = il_version_1.Box(
            x=p_upper.box.x,
            y=p_upper.box.y - required_gap,
            x2=p_upper.box.x2,
            y2=p_upper.box.y,
        )
        candidate_ids = list(para_index.intersection(box_to_tuple(check_area)))
        conflicting_paras = [
            para_map[pid]
            for pid in candidate_ids
            if pid != idx
            and not (
                para_map[pid].box
                and p_upper.box
                and para_map[pid].box.x2 < p_upper.box.x
                or para_map[pid].box.x > p_upper.box.x2
            )
        ]
        if conflicting_paras:
            max_y2 = max(
                p.box.y2 for p in conflicting_paras if p.box and p.box.y2 is not None
            )
            new_y = max_y2 + required_gap
            if p_upper.box and new_y < p_upper.box.y2:
                p_upper.box.y = new_y

    def render_page(self, page: il_version_1.Page):
        fonts = self._build_page_fonts(page)
        if (
            page.page_number == 0
            and self.translation_config.watermark_output_mode
            == WatermarkOutputMode.Watermarked
        ):
            self.add_watermark(page)
        try:
            self._adjust_paragraph_positions(page)
        except Exception as e:
            logger.warning(
                f"Failed to adjust paragraph positions on page {page.page_number}: {e}"
            )
        for paragraph in page.pdf_paragraph:
            try:
                self.render_paragraph(paragraph, page, fonts)
            except Exception as e:
                logger.warning(
                    f"Failed to render paragraph {getattr(paragraph, 'debug_id', '?')} "
                    f"on page {page.page_number}: {type(e).__name__}: {e}"
                )

    def add_watermark(self, page: il_version_1.Page):
        page_width = page.cropbox.box.x2 - page.cropbox.box.x
        page_height = page.cropbox.box.y2 - page.cropbox.box.y
        style = il_version_1.PdfStyle(
            font_id="base",
            font_size=6,
            graphic_state=il_version_1.GraphicState(),
        )
        text = f"本文档由 funstory.ai 的开源 PDF 翻译库 DocTranslator {settings.WATERMARK_VERSION} (http://yadt.io) 翻译，本仓库正在积极的建设当中，欢迎 star 和关注。"
        if self.translation_config.debug:
            text += "\n 当前为 DEBUG 模式，将显示更多辅助信息。请注意，部分框的位置对应原文，但在译文中可能不正确。"
        page.pdf_paragraph.append(
            il_version_1.PdfParagraph(
                first_line_indent=False,
                box=il_version_1.Box(
                    x=page.cropbox.box.x + page_width * 0.05,
                    y=page.cropbox.box.y,
                    x2=page.cropbox.box.x2,
                    y2=page.cropbox.box.y2 - page_height * 0.05,
                ),
                vertical=False,
                pdf_style=style,
                pdf_paragraph_composition=[
                    il_version_1.PdfParagraphComposition(
                        pdf_same_style_unicode_characters=il_version_1.PdfSameStyleUnicodeCharacters(
                            unicode=text,
                            pdf_style=style,
                        ),
                    ),
                ],
                xobj_id=-1,
            ),
        )

    def render_paragraph(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ],
    ):
        typesetting_units = self.create_typesetting_units(paragraph, fonts)
        # 如果所有单元都可以直接传递，则直接传递
        if all(unit.can_passthrough for unit in typesetting_units):
            paragraph.scale = 1.0
            paragraph.pdf_paragraph_composition = self.create_passthrough_composition(
                typesetting_units,
            )
        else:
            # 使用预计算的缩放因子进行重排版
            precomputed_scale = (
                paragraph.optimal_scale if paragraph.optimal_scale is not None else 1.0
            )

            # 如果有单元无法直接传递，则进行重排版
            paragraph.pdf_paragraph_composition = []
            self.retypeset_with_precomputed_scale(
                paragraph, page, typesetting_units, precomputed_scale
            )

            # 重排版后，重新设置段落各字符的 render order
            self._update_paragraph_render_order(paragraph)

    def _get_width_before_next_break_point(
        self, typesetting_units: list[TypesettingUnit], scale: float
    ) -> float:
        if not typesetting_units:
            return 0
        if typesetting_units[0].can_break_line:
            return 0

        total_width = 0
        for unit in typesetting_units:
            if unit.can_break_line:
                return total_width * scale
            total_width += unit.width
        return total_width * scale

    def _collect_font_sizes(
        self, typesetting_units: list[TypesettingUnit]
    ) -> list[float]:
        """Gather all font sizes referenced by the given typesetting units."""
        font_sizes = []
        for unit in typesetting_units:
            if unit.font_size:
                font_sizes.append(unit.font_size)
            if unit.char and unit.char.pdf_style and unit.char.pdf_style.font_size:
                font_sizes.append(unit.char.pdf_style.font_size)
        font_sizes.sort()
        return font_sizes

    def _calc_avg_height(
        self, typesetting_units: list[TypesettingUnit], scale: float
    ) -> float:
        """Return the modal (or average) unit height scaled by *scale*."""
        unit_heights = (
            [unit.height for unit in typesetting_units] if typesetting_units else []
        )
        if not unit_heights:
            return 0.0
        if len(unit_heights) == 1:
            return unit_heights[0] * scale
        try:
            return statistics.mode(unit_heights) * scale
        except statistics.StatisticsError:
            return sum(unit_heights) / len(unit_heights) * scale

    def _needs_cjk_mixed_space(
        self,
        last_unit: TypesettingUnit,
        unit: TypesettingUnit,
        current_x: float,
        current_y: float,
        line_height: float,
        box: Box,
    ) -> bool:
        """Return True when a half-width space should be injected at a CJK/Latin boundary."""
        cjk_end_puncts = {"。", "！", "？", "；", "：", "，"}
        return (
            last_unit is not None
            and last_unit.is_cjk_char ^ unit.is_cjk_char
            and last_unit.box
            and last_unit.box.y
            and current_y - 0.1 <= last_unit.box.y2 <= current_y + line_height + 0.1
            and not last_unit.mixed_character_blacklist
            and not unit.mixed_character_blacklist
            and current_x > box.x
            and unit.try_get_unicode() != " "
            and last_unit.try_get_unicode() != " "
            and last_unit.try_get_unicode() not in cjk_end_puncts
        )

    def _unit_overflows_line(
        self,
        unit: TypesettingUnit,
        unit_width: float,
        current_x: float,
        box: Box,
        use_english_line_break: bool,
        width_before_next_break_point: float,
    ) -> bool:
        """Return True when the unit cannot fit on the current line."""
        if unit.is_hung_punctuation:
            return False
        if current_x + unit_width > box.x2:
            return True
        if (
            use_english_line_break
            and current_x + unit_width + width_before_next_break_point > box.x2
        ):
            return True
        if (
            unit.is_cannot_appear_in_line_end_punctuation
            and current_x + unit_width * 2 > box.x2
        ):
            return True
        return False

    def _advance_to_next_line(
        self,
        current_line_heights: list[float],
        line_skip: float,
        current_y: float,
        line_ys: list[float],
        box: Box,
    ) -> tuple[float, bool, list[float]]:
        """Compute the y-coordinate for the next line. Returns (new_y, all_fit, cleared_heights)."""
        max_height = max(current_line_heights)
        try:
            mode_height = statistics.mode(current_line_heights)
        except statistics.StatisticsError:
            mode_height = max_height
        new_y = current_y - max(mode_height * line_skip, max_height * 1.05)
        line_ys.append(new_y)
        all_fit = new_y >= box.y
        return new_y, all_fit, []

    def _layout_typesetting_units(  # NOSONAR - text layout flow keeps PDF line-breaking decisions together
        self,
        typesetting_units: list[TypesettingUnit],
        box: Box,
        scale: float,
        line_skip: float,
        paragraph: il_version_1.PdfParagraph,
        use_english_line_break: bool = True,
    ) -> tuple[list[TypesettingUnit], bool]:
        """布局排版单元。

        Args:
            typesetting_units: 要布局的排版单元列表
            box: 布局边界框
            scale: 缩放因子

        Returns:
            tuple[list[TypesettingUnit], bool]: (已布局的排版单元列表，是否所有单元都放得下)
        """
        font_sizes = self._collect_font_sizes(typesetting_units)
        try:
            font_size = statistics.mode(font_sizes)
        except statistics.StatisticsError:
            font_size = statistics.median(font_sizes) if font_sizes else 12.0
        space_width = (
            self.font_mapper.base_font.char_lengths("你", font_size * scale)[0] * 0.5
        )
        avg_height = self._calc_avg_height(typesetting_units, scale)

        current_x = box.x
        current_y = box.y2 - avg_height
        box = copy.deepcopy(box)
        line_height = 0
        current_line_heights: list[float] = []
        typeset_units: list[TypesettingUnit] = []
        all_units_fit = True
        last_unit: TypesettingUnit | None = None
        line_ys = [current_y]
        if paragraph.first_line_indent:
            current_x += space_width * 4

        for i, unit in enumerate(typesetting_units):
            unit_width = unit.width * scale
            unit_height = unit.height * scale

            if current_x == box.x and unit.is_space:
                continue

            if self._needs_cjk_mixed_space(
                last_unit, unit, current_x, current_y, line_height, box
            ):
                current_x += space_width * 0.5

            width_before_next_break_point = (
                self._get_width_before_next_break_point(typesetting_units[i:], scale)
                if use_english_line_break
                else 0
            )

            if self._unit_overflows_line(
                unit,
                unit_width,
                current_x,
                box,
                use_english_line_break,
                width_before_next_break_point,
            ):
                if current_line_heights:
                    current_x = box.x
                    current_y, line_all_fit, current_line_heights = (
                        self._advance_to_next_line(
                            current_line_heights, line_skip, current_y, line_ys, box
                        )
                    )
                    line_height = 0.0
                    if not line_all_fit:
                        all_units_fit = False
                    if unit.is_space:
                        line_height = max(line_height, unit_height)
                        continue
                else:
                    # Unit wider than box at current scale — place it anyway to
                    # prevent empty composition (which silently drops all text).
                    all_units_fit = False

            relocated_unit = unit.relocate(current_x, current_y, scale)
            typeset_units.append(relocated_unit)
            if not unit.is_space:
                current_line_heights.append(unit_height)

            prev_x = current_x
            current_x = relocated_unit.box.x2
            if prev_x > current_x:
                logger.warning(f"坐标回绕！！！TypesettingUnit: {unit.box}, ")

            last_unit = relocated_unit

        return typeset_units, all_units_fit

    def _units_from_unicode_composition(
        self,
        composition,
        paragraph: il_version_1.PdfParagraph,
        get_font,
    ) -> list[TypesettingUnit] | None:
        """Convert a pdf_same_style_unicode_characters composition to TypesettingUnits.

        Returns None when the composition should be skipped due to missing style/font.
        """
        style = composition.pdf_same_style_unicode_characters.pdf_style
        if style is None:
            logger.warning(
                f"Style is None. Composition: {composition}. Paragraph: {paragraph}. "
            )
            return None
        font_id = style.font_id
        if font_id is None:
            logger.warning(
                f"Font ID is None. Composition: {composition}. Paragraph: {paragraph}. "
            )
            return None
        font = get_font(font_id, paragraph.xobj_id)
        unicode_text = composition.pdf_same_style_unicode_characters.unicode
        if not unicode_text:
            return []
        debug_flag = composition.pdf_same_style_unicode_characters.debug_info or False
        return [
            TypesettingUnit(
                unicode=char_unicode,
                font=self.font_mapper.map(font, char_unicode),
                original_font=font,
                font_size=style.font_size,
                style=style,
                xobj_id=paragraph.xobj_id,
                debug_info=debug_flag,
            )
            for char_unicode in unicode_text
            if char_unicode not in ("\n",)
        ]

    def _units_from_composition(
        self,
        composition,
        paragraph: il_version_1.PdfParagraph,
        get_font,
    ) -> list[TypesettingUnit] | None:
        """Convert a single composition item to TypesettingUnits.

        Returns a list of units, or None to signal an unknown/fatal composition type.
        Returns an empty list for compositions that produce no units.
        """
        if composition.pdf_line:
            return [
                TypesettingUnit(char=char)
                for char in composition.pdf_line.pdf_character
            ]
        if composition.pdf_character:
            return [
                TypesettingUnit(
                    char=composition.pdf_character, debug_info=paragraph.debug_info
                )
            ]
        if composition.pdf_same_style_characters:
            return [
                TypesettingUnit(char=char)
                for char in composition.pdf_same_style_characters.pdf_character
            ]
        if composition.pdf_same_style_unicode_characters:
            return self._units_from_unicode_composition(
                composition, paragraph, get_font
            )
        if composition.pdf_formula:
            return [TypesettingUnit(formular=composition.pdf_formula)]
        logger.error(
            f"Unknown composition type. "
            f"Composition: {composition}. "
            f"Paragraph: {paragraph}. ",
        )
        return []

    def create_typesetting_units(
        self,
        paragraph: il_version_1.PdfParagraph,
        fonts: dict[str, il_version_1.PdfFont],
    ) -> list[TypesettingUnit]:
        if not paragraph.pdf_paragraph_composition:
            return []
        result = []

        @cache
        def get_font(font_id: str, xobj_id: int | None):
            if xobj_id in fonts:
                font = fonts[xobj_id][font_id]
            else:
                font = fonts[font_id]
            return font

        for composition in paragraph.pdf_paragraph_composition:
            if composition is None:
                continue
            units = self._units_from_composition(composition, paragraph, get_font)
            if units is not None:
                result.extend(units)

        result = list(filter(lambda x: x.unicode is None or x.font is not None, result))

        if any(x.width < 0 for x in result):
            logger.warning("有排版单元宽度小于 0，请检查字体映射是否正确。")
        return result

    def create_passthrough_composition(
        self,
        typesetting_units: list[TypesettingUnit],
    ) -> list[PdfParagraphComposition]:
        """从排版单元创建直接传递的段落组合。

        Args:
            typesetting_units: 排版单元列表

        Returns:
            段落组合列表
        """
        composition = []
        for unit in typesetting_units:
            if unit.formular:
                # 对于公式单元，直接创建包含完整公式的组合
                composition.append(PdfParagraphComposition(pdf_formula=unit.formular))
            else:
                # 对于字符单元，使用原有逻辑
                chars, curves, forms = unit.passthrough()
                composition.extend(
                    [PdfParagraphComposition(pdf_character=char) for char in chars],
                )
        return composition

    def _boxes_overlap_vertically(self, box_a: Box, box_b: Box) -> bool:
        """Return True when box_a and box_b share any vertical range."""
        return not (box_a.y >= box_b.y2 or box_a.y2 <= box_b.y)

    def _boxes_overlap_horizontally(self, box_a: Box, box_b: Box) -> bool:
        """Return True when box_a and box_b share any horizontal range."""
        return not (box_a.x >= box_b.x2 or box_a.x2 <= box_b.x)

    def _is_right_blocker(self, element_box: Box, current_box: Box) -> bool:
        """Return True if element_box is to the right of and vertically overlaps current_box."""
        return element_box.x > current_box.x and self._boxes_overlap_vertically(
            element_box, current_box
        )

    def _is_below_blocker(self, element_box: Box, current_box: Box) -> bool:
        """Return True if element_box is below and horizontally overlaps current_box."""
        return element_box.y2 < current_box.y and self._boxes_overlap_horizontally(
            element_box, current_box
        )

    def _iter_blocker_boxes_right(self, current_box: Box, page):
        """Yield the x coordinate of every element that lies to the right of current_box
        and has vertical overlap with it."""
        for para in page.pdf_paragraph:
            if para.box is None or para.box == current_box:
                continue
            if self._is_right_blocker(para.box, current_box):
                yield para.box.x
        for char in page.pdf_character:
            if self._is_right_blocker(char.box, current_box):
                yield char.box.x
        for figure in page.pdf_figure:
            if self._is_right_blocker(figure.box, current_box):
                yield figure.box.x

    def _iter_blocker_boxes_below(self, current_box: Box, page):
        """Yield the y2 coordinate of every element that lies below current_box
        and has horizontal overlap with it."""
        for para in page.pdf_paragraph:
            if para.box is None or para.box == current_box:
                continue
            if self._is_below_blocker(para.box, current_box):
                yield para.box.y2
        for char in page.pdf_character:
            if self._is_below_blocker(char.box, current_box):
                yield char.box.y2
        for figure in page.pdf_figure:
            if self._is_below_blocker(figure.box, current_box):
                yield figure.box.y2

    def get_max_right_space(self, current_box: Box, page) -> float:
        """获取段落右侧最大可用空间

        Args:
            current_box: 当前段落的边界框
            page: 当前页面

        Returns:
            可以扩展到的最大 x 坐标
        """
        max_x = page.cropbox.box.x2 * 0.9
        for x in self._iter_blocker_boxes_right(current_box, page):
            max_x = min(max_x, x)
        return max_x

    def get_max_bottom_space(self, current_box: Box, page: il_version_1.Page) -> float:
        """获取段落下方最大可用空间

        Args:
            current_box: 当前段落的边界框
            page: 当前页面

        Returns:
            可以扩展到的最小 y 坐标
        """
        min_y = page.cropbox.box.y * 1.1
        for y2 in self._iter_blocker_boxes_below(current_box, page):
            min_y = max(min_y, y2)
        return min_y

    def _update_paragraph_render_order(self, paragraph: il_version_1.PdfParagraph):
        """
        重新设置段落各字符的 render order
        主 render order 等于 paragraph 的 renderorder，sub render order 从 1 开始自增
        """
        if not hasattr(paragraph, "render_order") or paragraph.render_order is None:
            return

        main_render_order = paragraph.render_order
        sub_render_order = 1

        # 遍历段落的所有组成部分
        for composition in paragraph.pdf_paragraph_composition:
            # 检查单个字符
            if composition.pdf_character:
                char = composition.pdf_character
                char.render_order = main_render_order
                char.sub_render_order = sub_render_order
                sub_render_order += 1
