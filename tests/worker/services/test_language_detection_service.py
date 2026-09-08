"""Tests for LanguageDetectionService (implementation_plan.md Phase C.2/C.6)."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
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
        confident-looking wrong guess -- it raises.

        These paragraphs are all below `is_detectable_text`'s floor, so no
        candidate characters are accumulated and this is the "nothing worth
        classifying" branch, not the "nothing classifiable" one.
        """
        docx_path = _make_docx(tmp_path, ["Information", "Total", "OK", "2026"])
        with pytest.raises(ValueError, match="no sufficiently long text was found"):
            service.detect(docx_path, is_docx=True)

    def test_long_but_unclassifiable_text_reports_the_other_error(
        self, service, tmp_path
    ):
        """Text present and long enough, but no language identifiable.

        Reported differently from "no text at all" so the user is not told
        to supply readable text they already supplied. Detection is stubbed
        because producing genuinely long-yet-undetermined prose is not
        reliably reproducible across lingua versions.
        """
        long_text = (
            "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
            "eiusmod tempor incididunt ut labore et dolore magna aliqua."
        )
        docx_path = _make_docx(tmp_path, [long_text, long_text])
        with patch(
            "src.worker.services.language_detection_service.detect_language_for_text",
            return_value=None,
        ):
            with pytest.raises(
                ValueError, match="no passage was long or distinctive enough"
            ):
                service.detect(docx_path, is_docx=True)

    def test_empty_docx_raises_explicit_error(self, service, tmp_path):
        docx_path = _make_docx(tmp_path, [""])
        with pytest.raises(ValueError, match="no sufficiently long text was found"):
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
        with pytest.raises(ValueError, match="no sufficiently long text was found"):
            service.detect(txt_path, is_txt=True)


class TestManyDistinctLanguagesAreAccepted:
    """The MAX_DISTINCT_LANGUAGES_PER_DOCUMENT count guard is gone.

    It measured detection noise, not multilingualism: per-block detection on
    a long real document routinely yields dozens of one-block "languages".
    Mixed documents are now translated, and the only remaining rejection is
    the share-based supported-language coverage gate in
    `PipelineOrchestrator._assert_language_supported`.
    """

    def test_five_languages_return_a_full_distribution(self, service):
        long_texts = [
            f"This is sentence number {i} written in English for testing purposes only."
            for i in range(3)
        ] + [
            "Ceci est une phrase francaise suffisamment longue pour tester.",
            "Dies ist ein deutscher Satz der lang genug ist zum Testen heute.",
            "Questo e un testo abbastanza lungo in italiano per il test.",
            "Esta es una oracion en espanol suficientemente larga para probar.",
        ]
        winner, distribution = service._detect_units(long_texts, source_label="test")
        assert winner == "en"
        assert len(distribution) >= 4
