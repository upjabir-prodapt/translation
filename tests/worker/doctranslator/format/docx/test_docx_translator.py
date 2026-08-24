"""Unit tests for top-level translate_docx orchestration and attempt reuse."""

from pathlib import Path
from unittest.mock import patch

from docx import Document
from src.worker.doctranslator.format.docx.docx_translator import translate_docx
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
    pre_extracted = [("Hello", "Bonjour")]

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
