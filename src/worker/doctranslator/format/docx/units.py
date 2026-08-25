"""Extraction of translatable paragraph units from a DOCX document.

A "unit" is one python-docx Paragraph found anywhere in the document
(body, tables -- including nested tables, headers, footers) together with a
stable integer id and a label describing where it came from. Units are
collected in document reading order: headers first (by section), then body
(including inline tables), then footers.

This module is 100% language-independent -- the same units are produced
regardless of target language, which is why the extraction step can be
shared across a multi-target-language batch (see shared_document_prep.py
for the equivalent optimization on the PDF side).
"""

from __future__ import annotations

from dataclasses import dataclass

from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass(slots=True)
class TranslatableUnit:
    """One paragraph eligible for translation, with a stable id."""

    unit_id: int
    paragraph: Paragraph
    label: str  # "header" | "footer" | "title" | "text" | "table_cell"
    text: str


def _iter_table_paragraphs(table: Table):
    """Yield every paragraph in a table's cells, recursing into nested tables."""
    for row in table.rows:
        for cell in row.cells:
            yield from cell.paragraphs
            for nested_table in cell.tables:
                yield from _iter_table_paragraphs(nested_table)


def _guess_label(paragraph: Paragraph, *, default: str) -> str:
    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
    if "title" in style_name or "heading" in style_name:
        return "title"
    return default


def _paragraph_text(paragraph: Paragraph) -> str:
    return paragraph.text or ""


def extract_units(document: DocxDocument) -> list[TranslatableUnit]:
    """Walk a DOCX document in reading order and return all translatable units.

    Order: for each section, its header paragraphs; then the body (top-level
    paragraphs and top-level tables, in the order they appear in the
    document body -- python-docx exposes this combined order via
    `document.element.body`); then for each section, its footer paragraphs.

    Empty/whitespace-only paragraphs are skipped entirely (they never need
    translation and would only waste an LLM-batch slot).
    """
    units: list[TranslatableUnit] = []
    next_id = 0

    def _add(paragraph: Paragraph, label: str) -> None:
        nonlocal next_id
        text = _paragraph_text(paragraph)
        if not text.strip():
            return
        units.append(
            TranslatableUnit(
                unit_id=next_id,
                paragraph=paragraph,
                label=label,
                text=text,
            )
        )
        next_id += 1

    seen_header_ids: set[int] = set()
    seen_footer_ids: set[int] = set()

    for section in document.sections:
        for header in (
            section.header,
            section.first_page_header,
            section.even_page_header,
        ):
            if header is None or id(header) in seen_header_ids:
                continue
            seen_header_ids.add(id(header))
            if header.is_linked_to_previous:
                continue
            for paragraph in header.paragraphs:
                _add(paragraph, "header")
            for table in header.tables:
                for paragraph in _iter_table_paragraphs(table):
                    _add(paragraph, "table_cell")

    # Body: iterate in true document order (paragraphs and tables
    # interleaved), by walking the underlying body XML element children.
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            _add(paragraph, _guess_label(paragraph, default="text"))
        elif tag == "tbl":
            table = Table(child, document)
            for paragraph in _iter_table_paragraphs(table):
                _add(paragraph, "table_cell")

    for section in document.sections:
        for footer in (
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ):
            if footer is None or id(footer) in seen_footer_ids:
                continue
            seen_footer_ids.add(id(footer))
            if footer.is_linked_to_previous:
                continue
            for paragraph in footer.paragraphs:
                _add(paragraph, "footer")
            for table in footer.tables:
                for paragraph in _iter_table_paragraphs(table):
                    _add(paragraph, "table_cell")

    return units


def write_translated_text(unit: TranslatableUnit, translated_text: str) -> None:
    """Replace a paragraph's text with the translated string.

    Keeps the paragraph's FIRST run's formatting (font, bold, italic, size,
    color, language) and removes any additional runs -- this preserves
    paragraph-level fidelity (alignment, list numbering, style, table cell
    membership) but does not preserve character-level mixed formatting
    *within* one paragraph (e.g. a single bold word mid-sentence). This is
    a documented, intentional v1 tradeoff.

    Images, embedded objects, and every other part of the document
    (styles.xml, numbering.xml, section properties, media relationships)
    are never touched -- only run text within this paragraph is mutated.
    """
    paragraph = unit.paragraph
    runs = paragraph.runs
    if not runs:
        # Paragraph has no runs (rare: e.g. purely field-code paragraph) --
        # add a single run with the translated text.
        paragraph.add_run(translated_text)
        return

    first_run = runs[0]
    first_run.text = translated_text
    for extra_run in runs[1:]:
        extra_run.text = ""
        extra_run._element.getparent().remove(extra_run._element)  # noqa: SLF001
