"""Tests for LanguageDetectionService (implementation_plan.md Phase C.2/C.6)."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.services.language_detection_core import MixedLanguageError
from src.worker.services.language_detection_service import LanguageDetectionService


def _make_docx(tmp_path, paragraphs: list[str]):
    from docx import Document as DocxDocument

    document = DocxDocument()
    for text in paragraphs:
        document.add_paragraph(text)
    path = tmp_path / "input.docx"
    document.save(str(path))
    return path


@pytest.fixture
def service():
    with patch(
        "src.worker.services.language_detection_service.JobProcessor"
    ) as mock_processor_cls:
        mock_processor = MagicMock()
        mock_processor.MAX_DETECTION_CHARS = 10000
        mock_processor_cls.return_value = mock_processor
        yield LanguageDetectionService()


class TestDetectDocx:
    def test_dominant_language_detected(self, service, tmp_path):
        """C.6.1: DOCX 70% DE / 30% EN -> detects `de`."""
        german_text = (
            "Dies ist ein langer deutscher Satz mit ausreichend Buchstaben "
            "fuer die Spracherkennung."
        )
        english_text = "This is a reasonably long English sentence for detection."
        docx_path = _make_docx(
            tmp_path,
            [german_text, german_text, english_text],
        )
        detected = service.detect(docx_path, is_docx=True)
        assert detected == "de"

    def test_short_ambiguous_paragraphs_raise_explicit_error(self, service, tmp_path):
        """C.6.3/EC-09: short/ambiguous text never produces a silent
        confident-looking wrong guess -- it raises asking the caller to
        choose the source language explicitly."""
        docx_path = _make_docx(tmp_path, ["Information", "Total", "OK", "2026"])
        with pytest.raises(ValueError, match="choose the source language"):
            service.detect(docx_path, is_docx=True)

    def test_empty_docx_raises_explicit_error(self, service, tmp_path):
        docx_path = _make_docx(tmp_path, [""])
        with pytest.raises(ValueError, match="choose the source language"):
            service.detect(docx_path, is_docx=True)


class TestDetectTxt:
    def test_dominant_language_detected(self, service, tmp_path):
        txt_path = tmp_path / "input.txt"
        txt_path.write_text(
            "Ceci est une phrase francaise suffisamment longue pour la "
            "detection automatique.\n"
            "Ceci est une autre phrase francaise assez longue.\n",
            encoding="utf-8",
        )
        detected = service.detect(txt_path, is_txt=True)
        assert detected == "fr"

    def test_whitespace_only_raises_explicit_error(self, service, tmp_path):
        txt_path = tmp_path / "blank.txt"
        txt_path.write_text("   \n\t  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="choose the source language"):
            service.detect(txt_path, is_txt=True)


class TestDetectUnitsMixedLanguage:
    """DOCX/TXT share the PDF pipeline's char-share dominance rule: a
    document is rejected only when no single language dominates it, not
    because several languages were spotted."""

    def _english(self, count: int) -> list[str]:
        return [
            f"This is sentence number {i} written in English for testing purposes only."
            for i in range(count)
        ]

    def test_evenly_split_document_is_rejected(self, service):
        texts = self._english(8) + [
            "Ceci est une phrase francaise suffisamment longue pour tester ceci.",
            "Ceci est une autre phrase francaise assez longue pour la detection.",
            "Nous ecrivons ici plusieurs phrases francaises pour equilibrer le texte.",
            "Voici encore une phrase francaise de longueur raisonnable pour tester.",
            "La derniere phrase francaise ajoutee pour completer cette moitie ici.",
            "Une phrase francaise supplementaire afin de bien equilibrer les parts.",
            "Encore une autre phrase francaise pour atteindre la moitie du texte.",
            "Cette phrase francaise termine la moitie francaise de ce document.",
        ]
        with pytest.raises(MixedLanguageError, match="mixed-language"):
            service._detect_units(texts, source_label="test")

    def test_dominant_language_survives_incidental_other_languages(self, service):
        """The document-level equivalent of the PDF regression: mostly
        English, with a couple of stray non-English units, translates as
        English instead of failing."""
        texts = self._english(20) + [
            "Ceci est une phrase francaise suffisamment longue pour tester ceci.",
        ]
        winner, _distribution = service._detect_units(texts, source_label="test")
        assert winner == "en"
