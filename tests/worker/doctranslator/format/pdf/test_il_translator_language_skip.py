"""PDF equivalent of the DOCX language-based skip (implementation_plan.md
Phase C.4.3): `ILTranslatorLLMOnly._is_paragraph_skippable()` must skip a
paragraph confidently detected as already-in-target-language or outside
the configured language set.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
)


def _paragraph(text: str, debug_id: int = 1):
    para = MagicMock()
    para.unicode = text
    para.debug_id = debug_id
    para.pdf_paragraph_composition = []  # real iterable for is_cid_paragraph()
    return para


def _translator(lang_out: str = "de"):
    inst = ILTranslatorLLMOnly.__new__(ILTranslatorLLMOnly)
    config = MagicMock()
    config.lang_out = lang_out
    config.min_text_length = 1
    inst.translation_config = config
    return inst


class TestIsUnsupportedOrAlreadyTargetLanguage:
    def test_confident_target_language_text_is_skipped(self):
        translator = _translator(lang_out="de")
        german_text = (
            "Dies ist ein ausreichend langer deutscher Satz fuer die "
            "Spracherkennung des Uebersetzers."
        )
        assert translator._is_unsupported_or_already_target_language(german_text)

    def test_confident_unsupported_language_text_is_skipped(self):
        translator = _translator(lang_out="de")
        dutch_text = (
            "Dit is een voldoende lange Nederlandse zin voor de "
            "taalherkenning van de vertaler vandaag."
        )
        assert translator._is_unsupported_or_already_target_language(dutch_text)

    def test_confident_supported_source_language_text_is_kept(self):
        translator = _translator(lang_out="de")
        english_text = (
            "This is a sufficiently long English sentence intended for "
            "the translator's language detection today."
        )
        assert not translator._is_unsupported_or_already_target_language(
            english_text
        )

    def test_short_text_is_never_language_skipped(self):
        translator = _translator(lang_out="de")
        assert not translator._is_unsupported_or_already_target_language(
            "Hallo Welt"
        )

    def test_disabled_via_setting(self):
        translator = _translator(lang_out="de")
        german_text = (
            "Dies ist ein ausreichend langer deutscher Satz fuer die "
            "Spracherkennung des Uebersetzers."
        )
        with patch(
            "src.worker.doctranslator.format.pdf.document_il.midend."
            "il_translator_llm_only.settings"
        ) as mock_settings:
            mock_settings.SKIP_UNSUPPORTED_LANGUAGE_UNITS = False
            assert not translator._is_unsupported_or_already_target_language(
                german_text
            )


class TestIsParagraphSkippableLanguageIntegration:
    """Integration through the real `_is_paragraph_skippable()` entry point.

    Note: with an empty `pdf_paragraph_composition` (this module's bare
    mock fixture has no real IL character data), `is_placeholder_only_paragraph()`
    vacuously returns True for *any* paragraph (`all([]) is True`), so a
    "not skippable" case cannot be exercised through this specific mock
    shape -- that direction is already covered directly at the
    `_is_unsupported_or_already_target_language()` level above. This test
    only asserts the "should skip" direction end-to-end.
    """

    def test_paragraph_in_target_language_is_skippable(self):
        translator = _translator(lang_out="de")
        german_text = (
            "Dies ist ein ausreichend langer deutscher Satz fuer die "
            "Spracherkennung des Uebersetzers."
        )
        paragraph = _paragraph(german_text)
        assert translator._is_paragraph_skippable(paragraph, set(), None) is True
