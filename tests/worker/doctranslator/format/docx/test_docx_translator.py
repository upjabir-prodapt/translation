"""Unit tests for top-level translate_docx orchestration and attempt reuse."""

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from src.worker.doctranslator.format.docx.docx_translator import translate_docx
from src.worker.doctranslator.glossary import ExtractedGlossaryTerm
from src.worker.services.dlp_service import DlpProvider
from src.worker.services.dlp_service import DlpResult


class _FakeTranslator:
    provider = "gemini"
    model = "gemini-3.5-flash"
    lang_in = "en"

    def __init__(self, output_text: str = '[{"id": 0, "output": "Bonjour le monde"}]'):
        self.output_text = output_text
        self.calls: list[str] = []

    def llm_translate(self, prompt, response_schema=None, batch_items=None):
        self.calls.append(prompt)
        return self.output_text

    def translate(self, text, rate_limit_params=None):
        return f"[single:{text}]"


def test_translate_docx_reuses_extracted_terms(tmp_path: Path):
    """C2: When extracted_terms are passed in, term extraction LLM calls are skipped."""
    input_file = tmp_path / "input.docx"
    output_file = tmp_path / "output.docx"

    doc = Document()
    doc.add_paragraph("Hello world")
    doc.save(str(input_file))

    translator = _FakeTranslator()
    pre_extracted = [
        ExtractedGlossaryTerm(source="Hello", target="Bonjour", source_language="en")
    ]

    with patch(
        "src.worker.doctranslator.format.docx.docx_translator.DocxTermExtractor"
    ) as mock_extractor_cls:
        result = translate_docx(
            input_path=input_file,
            output_path=output_file,
            translator=translator,
            lang_out="fr",
            job_id="job-123",
            source_language="en",
            enable_dlp=False,
            auto_extract_glossary=True,
            extracted_terms=pre_extracted,
        )

        # Extractor class should NOT have been instantiated because pre_extracted terms were supplied
        mock_extractor_cls.assert_not_called()

    assert result.extracted_terms == pre_extracted
    assert output_file.exists()


def test_translate_docx_reuses_dlp_result(tmp_path: Path):
    """C2: When cached dlp_result is passed in, DlpService.mask_chunks is skipped."""
    input_file = tmp_path / "input.docx"
    output_file = tmp_path / "output.docx"

    doc = Document()
    doc.add_paragraph("My email is test@example.com")
    doc.save(str(input_file))

    translator = _FakeTranslator('[{"id": 0, "output": "Mon email est [EMAIL_1]"}]')
    cached_dlp = DlpResult(
        masked_chunks=["My email is [EMAIL_1]"],
        token_rows=[{"token": "[EMAIL_1]", "original_value": "test@example.com"}],
        dlp_provider=DlpProvider.REGEX_FALLBACK,
    )

    with patch(
        "src.worker.doctranslator.format.docx.docx_translator.DlpService"
    ) as mock_dlp_cls:
        result = translate_docx(
            input_path=input_file,
            output_path=output_file,
            translator=translator,
            lang_out="fr",
            job_id="job-123",
            source_language="en",
            enable_dlp=True,
            auto_extract_glossary=False,
            dlp_result=cached_dlp,
        )

        mock_dlp_cls.assert_not_called()

    assert result.dlp_provider == DlpProvider.REGEX_FALLBACK
    assert result.dlp_token_rows == cached_dlp.token_rows
    # Restored text should replace [EMAIL_1] with test@example.com
    assert "test@example.com" in result.translated_text


def _make_docx_with_footnote_bytes() -> bytes:
    """Real DOCX with a footnote, built via zip/XML surgery (no python-docx
    authoring API exists for footnotes.xml -- see test_units.py)."""
    doc = Document()
    doc.add_paragraph("Hello world")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    zin = zipfile.ZipFile(buf)
    names = zin.namelist()
    content_types = zin.read("[Content_Types].xml").decode()
    doc_xml = zin.read("word/document.xml").decode()
    doc_rels = zin.read("word/_rels/document.xml.rels").decode()

    doc_xml = doc_xml.replace(
        "<w:t>Hello world</w:t></w:r>",
        '<w:t>Hello world</w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r>',
    )
    footnotes_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:footnotes xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main">'
        '<w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r>'
        '<w:r><w:t xml:space="preserve"> Note text</w:t></w:r>'
        "</w:p></w:footnote></w:footnotes>"
    )
    content_types = content_types.replace(
        "</Types>",
        '<Override PartName="/word/footnotes.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.wordprocessingml'
        '.footnotes+xml"/></Types>',
    )
    doc_rels = doc_rels.replace(
        "</Relationships>",
        '<Relationship Id="rIdFootnotes1" Type='
        '"http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/footnotes" Target="footnotes.xml"/></Relationships>',
    )

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            if name == "[Content_Types].xml":
                zout.writestr(name, content_types)
            elif name == "word/document.xml":
                zout.writestr(name, doc_xml)
            elif name == "word/_rels/document.xml.rels":
                zout.writestr(name, doc_rels)
            else:
                zout.writestr(name, zin.read(name))
        zout.writestr("word/footnotes.xml", footnotes_xml)
    return out.getvalue()


def test_translate_docx_translates_and_persists_footnote_text(tmp_path: Path):
    """implementation_plan.md D.2.1: footnote text must be translated AND
    survive all the way through translate_docx()'s document.save()."""
    input_file = tmp_path / "input.docx"
    output_file = tmp_path / "output.docx"
    input_file.write_bytes(_make_docx_with_footnote_bytes())

    translator = _FakeTranslator(
        '[{"id": 0, "output": "Bonjour le monde"}, '
        '{"id": 1, "output": "Texte de la note"}]'
    )

    result = translate_docx(
        input_path=input_file,
        output_path=output_file,
        translator=translator,
        lang_out="fr",
        job_id="job-footnote",
        source_language="en",
        enable_dlp=False,
        auto_extract_glossary=False,
    )

    assert "Texte de la note" in result.translated_text
    reloaded = Document(str(output_file))
    footnotes_part = reloaded.part.part_related_by(RT.FOOTNOTES)
    assert b"Texte de la note" in footnotes_part.blob
    assert b"Note text" not in footnotes_part.blob


def test_translate_docx_returns_aligned_segment_pairs(tmp_path: Path):
    """The judge chunks on pairs, not on two concatenated blobs.

    Target text runs 10-20% longer than source in many language pairs, so
    chunking the two texts independently would compare paragraph N of the
    source against paragraph N-3 of the translation and report the
    difference as omission plus hallucination. DOCX is single-pass and
    in-process, so the pairs are handed over in memory rather than round
    tripped through JSON as on the PDF path.
    """
    input_file = tmp_path / "input.docx"
    output_file = tmp_path / "output.docx"

    doc = Document()
    doc.add_paragraph("Hello world")
    doc.add_paragraph("Second paragraph")
    doc.save(str(input_file))

    result = translate_docx(
        input_path=input_file,
        output_path=output_file,
        translator=_FakeTranslator(),
        lang_out="fr",
        job_id="job-seg",
        source_language="en",
        enable_dlp=False,
        auto_extract_glossary=False,
    )

    assert result.segments
    # One pair per translated unit, and the two views of the document agree.
    assert [s for s, _ in result.segments] == result.source_text.split("\n")
    assert [t for _, t in result.segments] == result.translated_text.split("\n")
