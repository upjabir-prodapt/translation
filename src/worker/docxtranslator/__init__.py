"""Native DOCX translation: Word in, Word out, structure preserved.

Unlike the PDF path, nothing here rasterizes, converts, or re-typesets the
document. The .docx package is opened as OOXML, ``w:t`` text nodes are
translated in place, and the package is written back — so tables stay real
Word tables, styles and numbering keep working, and the output remains fully
editable.
"""

from src.worker.docxtranslator.package import DocxPackage
from src.worker.docxtranslator.package import InvalidDocxError
from src.worker.docxtranslator.segment_translator import DocxSegmentTranslator
from src.worker.docxtranslator.segment_translator import TranslationOutcome
from src.worker.docxtranslator.segments import Segment
from src.worker.docxtranslator.segments import collect_segments
from src.worker.docxtranslator.segments import extract_text

__all__ = [
    "DocxPackage",
    "DocxSegmentTranslator",
    "InvalidDocxError",
    "Segment",
    "TranslationOutcome",
    "collect_segments",
    "extract_text",
]
