"""DLP masking must survive into the text actually sent to the LLM.

`apply_dlp_to_document()` masks `PdfParagraph.unicode`, but
`get_translate_input()` rebuilds the prompt text from the paragraph's
*characters* for any paragraph with more than one composition, and
`translate_paragraph()` rebuilds it unconditionally when this translator
runs as the batch translator's fallback. Both rebuilt strings dropped the
masking, so raw PII was sent to the translation model for most paragraphs.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.format.pdf.document_il.midend.il_translator import (
    ILTranslator,
)

_TOKEN_ROWS = [
    {
        "token": "__DLP_TOKEN_0001__",
        "original_value": "Max Mustermann",
        "info_type": "PERSON_NAME",
    },
    {
        "token": "__DLP_TOKEN_0002__",
        "original_value": "max@example.de",
        "info_type": "EMAIL_ADDRESS",
    },
]


def _translator(token_rows=_TOKEN_ROWS, *, enable_dlp: bool = True):
    translator = ILTranslator.__new__(ILTranslator)
    config = MagicMock()
    config.enable_dlp = enable_dlp
    config.dlp_token_rows = list(token_rows)
    config.min_text_length = 1
    config.disable_rich_text_translate = True
    translator.translation_config = config
    translator.support_llm_translate = True
    return translator


def _paragraph(masked_unicode: str):
    paragraph = MagicMock()
    paragraph.unicode = masked_unicode
    paragraph.debug_id = "9f1Bh"
    paragraph.vertical = False
    paragraph.xobj_id = 0
    return paragraph


def _translate_input(text: str):
    return ILTranslator.TranslateInput(text, [], MagicMock())


class TestApplyDlpMaskToText:
    def test_rebuilt_text_is_masked(self):
        assert (
            _translator().apply_dlp_mask_to_text(
                "Unterzeichnet von Max Mustermann.", "paragraph 1"
            )
            == "Unterzeichnet von __DLP_TOKEN_0001__."
        )

    def test_already_masked_text_is_unchanged(self):
        text = "Unterzeichnet von __DLP_TOKEN_0001__."
        assert _translator().apply_dlp_mask_to_text(text, "paragraph 1") == text

    def test_masking_is_skipped_when_dlp_is_disabled(self):
        text = "Unterzeichnet von Max Mustermann."
        assert (
            _translator(enable_dlp=False).apply_dlp_mask_to_text(text, "paragraph 1")
            == text
        )

    def test_masking_is_skipped_when_there_are_no_token_rows(self):
        text = "Unterzeichnet von Max Mustermann."
        assert _translator([]).apply_dlp_mask_to_text(text, "paragraph 1") == text

    def test_value_that_cannot_be_remasked_is_reported(self, caplog):
        """A rich-text placeholder split the value, so the token the masking
        pass assigned never reaches the prompt."""
        with caplog.at_level(logging.WARNING):
            _translator().apply_dlp_mask_to_text(
                "Unterzeichnet von Max {v1}Mustermann.",
                "paragraph 9f1Bh",
                reference_masked_text="Unterzeichnet von __DLP_TOKEN_0001__.",
            )
        assert "PERSON_NAME" in caplog.text
        assert "paragraph 9f1Bh" in caplog.text

    def test_the_sensitive_value_is_never_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            _translator().apply_dlp_mask_to_text(
                "Unterzeichnet von Max {v1}Mustermann.",
                "paragraph 9f1Bh",
                reference_masked_text="Unterzeichnet von __DLP_TOKEN_0001__.",
            )
        assert "Mustermann" not in caplog.text

    def test_no_warning_when_everything_was_masked(self, caplog):
        with caplog.at_level(logging.WARNING):
            _translator().apply_dlp_mask_to_text(
                "Unterzeichnet von Max Mustermann.",
                "paragraph 9f1Bh",
                reference_masked_text="Unterzeichnet von __DLP_TOKEN_0001__.",
            )
        assert caplog.text == ""

    def test_applier_is_rebuilt_when_token_rows_grow(self):
        """An optional post-translation DLP pass appends rows mid-run."""
        translator = _translator([])
        assert translator.apply_dlp_mask_to_text("Max Mustermann", "p") == (
            "Max Mustermann"
        )
        translator.translation_config.dlp_token_rows = list(_TOKEN_ROWS)
        assert translator.apply_dlp_mask_to_text("Max Mustermann", "p") == (
            "__DLP_TOKEN_0001__"
        )


class TestPreTranslateParagraphMasksTheLlmInput:
    def _run(self, rebuilt_text: str, masked_unicode: str):
        translator = _translator()
        tracker = MagicMock()
        with patch.object(
            ILTranslator,
            "get_translate_input",
            return_value=_translate_input(rebuilt_text),
        ):
            return translator.pre_translate_paragraph(
                _paragraph(masked_unicode), tracker, {}, {}
            ), tracker

    def test_text_sent_to_the_model_is_masked(self):
        """The regression: get_translate_input() rebuilds from characters, so
        the returned text used to carry the raw PII."""
        (text, translate_input), _tracker = self._run(
            "Bitte kontaktieren Sie Max Mustermann unter max@example.de.",
            "Bitte kontaktieren Sie __DLP_TOKEN_0001__ unter __DLP_TOKEN_0002__.",
        )
        assert text == (
            "Bitte kontaktieren Sie __DLP_TOKEN_0001__ unter __DLP_TOKEN_0002__."
        )
        assert translate_input.unicode == text
        assert "Mustermann" not in text
        assert "max@example.de" not in text

    def test_tracker_records_the_masked_input(self):
        """Debug artefacts must not carry the unmasked values either."""
        _result, tracker = self._run(
            "Unterzeichnet von Max Mustermann.",
            "Unterzeichnet von __DLP_TOKEN_0001__.",
        )
        tracker.set_input.assert_called_once_with(
            "Unterzeichnet von __DLP_TOKEN_0001__."
        )
