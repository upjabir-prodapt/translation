"""AI-translation cover page for native DOCX output.

Mirrors the PDF pipeline's cover page (see
`src/worker/services/processor_service.py::_draw_cover_page`) but uses
native DOCX paragraphs instead of a redrawn PDF page, since OOXML has no
page-drawing primitives -- see
docs/architecture/pdf-vs-docx-translation-architecture.md for the full
PDF-vs-DOCX comparison.

The `TranslationCoverPageMetadata` dataclass (original/target language,
model used, domain, translation date, confidence score, sections
translated, disclaimer text) is shared verbatim with the PDF pipeline so
both formats always show the exact same fields and wording.
"""

from __future__ import annotations

from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_BREAK
from docx.shared import Pt
from docx.shared import RGBColor
from docx.text.paragraph import Paragraph

from src.worker.doctranslator.format.pdf.translation_config import (
    TranslationCoverPageMetadata,
)

_DARK_GRAY = (0x59, 0x59, 0x59)
_DISCLAIMER_RED = (0x8C, 0x26, 0x26)


def add_cover_page(
    document: DocxDocument, metadata: TranslationCoverPageMetadata
) -> None:
    """Prepend an AI-translation cover page to a DOCX document, in place.

    Appends the cover-page paragraphs (they land at the end of the body,
    right before the last section's `sectPr`, which is where python-docx
    always inserts new paragraphs) and then relocates them to the very
    front of the document body, preserving their relative order. A page
    break is inserted after the cover content so the original document
    content always starts on its own page.
    """
    body = document.element.body
    new_paragraphs: list[Paragraph] = []

    def _add(
        text: str,
        *,
        bold: bool = False,
        size: float | None = None,
        color: tuple[int, int, int] | None = None,
        align: WD_ALIGN_PARAGRAPH | None = None,
    ) -> Paragraph:
        paragraph = document.add_paragraph()
        if align is not None:
            paragraph.alignment = align
        run = paragraph.add_run(text)
        run.bold = bold
        if size is not None:
            run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = RGBColor(*color)
        new_paragraphs.append(paragraph)
        return paragraph

    _add(
        "AI Translated Document",
        bold=True,
        size=20,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    _add(
        "This cover page summarizes the generated translation output.",
        size=11,
        color=_DARK_GRAY,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    _add("")

    for label, value in metadata.iter_rows():
        _add(f"{label}: {value}", size=11)

    _add("")
    disclaimer_paragraph = _add(
        metadata.DISCLAIMER, bold=True, size=10, color=_DISCLAIMER_RED
    )
    disclaimer_paragraph.add_run().add_break(WD_BREAK.PAGE)

    for paragraph in reversed(new_paragraphs):
        element = paragraph._p  # noqa: SLF001 - relocating our own new elements
        body.remove(element)
        body.insert(0, element)
