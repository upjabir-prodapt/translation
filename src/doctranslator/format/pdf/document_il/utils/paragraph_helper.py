import logging
import re

from src.doctranslator.format.pdf.document_il import il_version_1

logger = logging.getLogger(__name__)


def _collect_chars_from_composition(
    composition: il_version_1.PdfParagraphComposition,
    paragraph: il_version_1.PdfParagraph,
) -> list:
    """Return the character list contributed by *composition*."""
    if composition.pdf_line:
        return list(composition.pdf_line.pdf_character)
    if composition.pdf_same_style_characters:
        return list(composition.pdf_same_style_characters.pdf_character)
    if composition.pdf_same_style_unicode_characters:
        return []
    if composition.pdf_formula:
        return list(composition.pdf_formula.pdf_character)
    if composition.pdf_character:
        return [composition.pdf_character]
    logger.error(
        f"Unknown composition type. "
        f"Composition: {composition}. "
        f"Paragraph: {paragraph}. ",
    )
    return []


def _count_cid_chars(chars: list) -> int:
    """Return the number of characters whose unicode value matches the CID pattern."""
    cid_pattern = re.compile(r"^\(cid:\d+\)$")
    return sum(1 for char in chars if cid_pattern.match(char.char_unicode))


def is_cid_paragraph(paragraph: il_version_1.PdfParagraph):
    chars: list[il_version_1.PdfCharacter] = []
    for composition in paragraph.pdf_paragraph_composition:
        chars.extend(_collect_chars_from_composition(composition, paragraph))

    cid_count = _count_cid_chars(chars)
    return cid_count > len(chars) * 0.8


NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def is_pure_numeric_paragraph(paragraph) -> bool:
    """只检查段落是否为纯数字（支持整数、小数、负数）"""

    if not paragraph or not getattr(paragraph, "unicode", None):
        return False

    text = paragraph.unicode.strip()
    if not text:
        return False

    return bool(NUMERIC_PATTERN.match(text))


def _composition_is_whitespace_only(composition) -> bool:
    """Return True if a paragraph composition contains only whitespace characters.

    Returns False if the composition type is unknown or contains non-whitespace.
    Returns None if the composition is a formula (which is always allowed).
    """
    if composition.pdf_formula:
        return True
    if composition.pdf_character:
        return composition.pdf_character.char_unicode.isspace()
    if composition.pdf_line:
        return all(
            char.char_unicode.isspace() for char in composition.pdf_line.pdf_character
        )
    if composition.pdf_same_style_characters:
        return all(
            char.char_unicode.isspace()
            for char in composition.pdf_same_style_characters.pdf_character
        )
    if composition.pdf_same_style_unicode_characters:
        return composition.pdf_same_style_unicode_characters.unicode.isspace()
    return False


def is_placeholder_only_paragraph(paragraph: il_version_1.PdfParagraph) -> bool:
    """Check if a paragraph contains only placeholders and whitespace.

    Args:
        paragraph: PDF paragraph to check

    Returns:
        True if the paragraph contains only placeholders (formula or style tags)
        and whitespace, False otherwise
    """
    if not paragraph or not paragraph.unicode:
        return False

    return all(
        _composition_is_whitespace_only(composition)
        for composition in paragraph.pdf_paragraph_composition
    )
