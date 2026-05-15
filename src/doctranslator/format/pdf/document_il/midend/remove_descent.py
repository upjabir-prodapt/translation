import logging
from collections import Counter
from functools import cache

from src.doctranslator.format.pdf.document_il import il_version_1
from src.doctranslator.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)


class RemoveDescent:
    stage_name = "Remove Char Descent"

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config

    def _remove_char_descent(
        self,
        char: il_version_1.PdfCharacter,
        font: il_version_1.PdfFont,
    ) -> float | None:
        """Remove descent from a single character and return the descent value.

        Args:
            char: The character to process
            font: The font used by this character

        Returns:
            The descent value if it was removed, None otherwise
        """
        if (
            char.box
            and char.box.y is not None
            and char.box.y2 is not None
            and font
            and hasattr(font, "descent")
        ):
            descent = font.descent * char.pdf_style.font_size / 1000
            if char.vertical:
                # For vertical text, remove descent from x coordinates
                char.box.x += descent
                char.box.x2 += descent
            else:
                # For horizontal text, remove descent from y coordinates
                char.box.y -= descent
                char.box.y2 -= descent
            return descent
        return None

    def process(self, document: il_version_1.Document):
        """Process the document to remove descent adjustments from character boxes.

        Args:
            document: The document to process
        """
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            len(document.page),
        ) as pbar:
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.process_page(page)
                pbar.advance()

    def _build_font_map(self, page: il_version_1.Page):
        """Build a font lookup map for the page including xobject fonts."""
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ] = {f.font_id: f for f in page.pdf_font}
        page_fonts = {f.font_id: f for f in page.pdf_font}
        for xobj in page.pdf_xobject:
            fonts[xobj.xobj_id] = page_fonts.copy()
            for font in xobj.pdf_font:
                fonts[xobj.xobj_id][font.font_id] = font
        return fonts

    def _make_get_font(self, fonts):
        """Return a cached font-lookup closure for the given font map."""

        @cache
        def get_font(
            font_id: str,
            xobj_id: int | None = None,
        ) -> il_version_1.PdfFont | None:
            if xobj_id is not None and xobj_id in fonts:
                font_map = fonts[xobj_id]
                if isinstance(font_map, dict) and font_id in font_map:
                    return font_map[font_id]
            return (
                fonts.get(font_id)
                if isinstance(fonts.get(font_id), il_version_1.PdfFont)
                else None
            )

        return get_font

    def _process_chars_in_composition(
        self, comp, get_font, descent_values, vertical_chars
    ):
        """Remove descent from all characters in a single composition and collect results."""
        if comp.pdf_character:
            font = get_font(
                comp.pdf_character.pdf_style.font_id, comp.pdf_character.xobj_id
            )
            if font:
                descent = self._remove_char_descent(comp.pdf_character, font)
                if descent is not None:
                    descent_values.append(descent)
                    vertical_chars.append(comp.pdf_character.vertical)
        elif comp.pdf_line:
            self._process_char_list(
                comp.pdf_line.pdf_character, get_font, descent_values, vertical_chars
            )
        elif comp.pdf_formula:
            self._process_char_list(
                comp.pdf_formula.pdf_character, get_font, descent_values, vertical_chars
            )
        elif comp.pdf_same_style_characters:
            self._process_char_list(
                comp.pdf_same_style_characters.pdf_character,
                get_font,
                descent_values,
                vertical_chars,
            )

    def _process_char_list(self, chars, get_font, descent_values, vertical_chars):
        """Remove descent from a list of characters and collect descent/vertical data."""
        for char in chars:
            if font := get_font(char.pdf_style.font_id, char.xobj_id):
                descent = self._remove_char_descent(char, font)
                if descent is not None:
                    descent_values.append(descent)
                    vertical_chars.append(char.vertical)

    def _adjust_paragraph_box(self, paragraph, descent_values, vertical_chars):
        """Shift a paragraph's bounding box by the modal descent value."""
        if not descent_values or not paragraph.box:
            return
        most_common_descent = Counter(descent_values).most_common(1)[0][0]
        is_vertical = all(vertical_chars) if vertical_chars else False
        if paragraph.box.y is not None and paragraph.box.y2 is not None:
            if is_vertical:
                paragraph.box.x += most_common_descent
                paragraph.box.x2 += most_common_descent
            else:
                paragraph.box.y -= most_common_descent
                paragraph.box.y2 -= most_common_descent

    def process_page(self, page: il_version_1.Page):
        """Process a single page to remove descent adjustments.

        Args:
            page: The page to process
        """
        fonts = self._build_font_map(page)
        get_font = self._make_get_font(fonts)

        # Process all standalone characters in the page
        for char in page.pdf_character:
            if font := get_font(char.pdf_style.font_id, char.xobj_id):
                self._remove_char_descent(char, font)

        # Process all paragraphs
        for paragraph in page.pdf_paragraph:
            descent_values: list[float] = []
            vertical_chars: list[bool] = []
            for comp in paragraph.pdf_paragraph_composition:
                self._process_chars_in_composition(
                    comp, get_font, descent_values, vertical_chars
                )
            self._adjust_paragraph_box(paragraph, descent_values, vertical_chars)
