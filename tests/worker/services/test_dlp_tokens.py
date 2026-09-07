"""Tests for shared DLP placeholder-token handling (src/worker/services/dlp_tokens.py).

Regression coverage for the German failure mode: a PII finding covers only
part of a German word, the placeholder is emitted welded to the leftover
letters, and the model then reformats it while translating the word it
appears to belong to. Exact-string restoration missed those tokens, so the
DOCX pipeline shipped `__DLP_TOKEN_0001__` glued to the neighbouring word
and the PDF pipeline deleted the sensitive value outright.
"""

from __future__ import annotations

from src.worker.services.dlp_tokens import build_mask_applier
from src.worker.services.dlp_tokens import build_token_map
from src.worker.services.dlp_tokens import find_leaked_tokens
from src.worker.services.dlp_tokens import format_token
from src.worker.services.dlp_tokens import restore_tokens
from src.worker.services.dlp_tokens import strip_leaked_tokens_from_text

_ROWS = [
    {"token": "__DLP_TOKEN_0001__", "original_value": "Mustermann"},
    {"token": "__DLP_TOKEN_0002__", "original_value": "Müller-Schmidt"},
]


def _map() -> dict[int, str]:
    return build_token_map(_ROWS)


class TestTokenMap:
    def test_format_token_is_zero_padded(self):
        assert format_token(1) == "__DLP_TOKEN_0001__"
        assert format_token(1234) == "__DLP_TOKEN_1234__"

    def test_build_token_map_is_keyed_by_numeric_id(self):
        assert _map().by_id == {1: "Mustermann", 2: "Müller-Schmidt"}
        assert _map().literal == {}

    def test_rows_without_token_or_value_are_ignored(self):
        assert not build_token_map([{"token": "", "original_value": "x"}])
        assert not build_token_map([{"token": "__DLP_TOKEN_0001__"}])

    def test_non_canonical_token_shape_falls_back_to_exact_matching(self):
        """Callers may pre-compute a DLP result using their own placeholder
        format; those can only be matched literally, but must still work."""
        token_map = build_token_map(
            [{"token": "[EMAIL_1]", "original_value": "test@example.com"}]
        )
        assert token_map.literal == {"[EMAIL_1]": "test@example.com"}
        assert restore_tokens("Mon email est [EMAIL_1]", token_map) == (
            "Mon email est test@example.com",
            1,
        )


class TestRestoreTokens:
    def test_canonical_token_is_restored(self):
        assert restore_tokens("Herr __DLP_TOKEN_0001__.", _map()) == (
            "Herr Mustermann.",
            1,
        )

    def test_token_welded_to_a_german_suffix_is_restored(self):
        # DLP matched "Mustermann" inside the genitive "Mustermanns".
        assert restore_tokens("__DLP_TOKEN_0001__s Unterschrift", _map()) == (
            "Mustermanns Unterschrift",
            1,
        )

    def test_model_reformatted_tokens_are_restored(self):
        mangled = "__dlp_token_0001__ und __DLP-TOKEN-0002__ sowie _DLP_TOKEN_0001_"
        restored, count = restore_tokens(mangled, _map())
        assert restored == "Mustermann und Müller-Schmidt sowie Mustermann"
        assert count == 3

    def test_adjacent_tokens_are_both_restored(self):
        assert restore_tokens("__DLP_TOKEN_0001____DLP_TOKEN_0002__", _map()) == (
            "MustermannMüller-Schmidt",
            2,
        )

    def test_unknown_token_id_is_left_in_place(self):
        assert restore_tokens("__DLP_TOKEN_0099__", _map()) == ("__DLP_TOKEN_0099__", 0)

    def test_text_without_tokens_is_unchanged(self):
        text = "Ein ganz gewöhnlicher deutscher Satz."
        assert restore_tokens(text, _map()) == (text, 0)


class TestLeakDetection:
    def test_unresolved_token_is_reported_and_stripped(self):
        cleaned, leaked = strip_leaked_tokens_from_text("Rest __DLP_TOKEN_0099__ hier")
        assert leaked == ["__DLP_TOKEN_0099__"]
        assert "DLP_TOKEN" not in cleaned

    def test_welded_leftover_token_is_detected(self):
        assert find_leaked_tokens("Wort__DLP_TOKEN_0099__x") == ["__DLP_TOKEN_0099__"]

    def test_clean_text_reports_no_leak(self):
        assert find_leaked_tokens("Mustermanns Unterschrift") == []
        assert strip_leaked_tokens_from_text("Sauber") == ("Sauber", [])


class TestMaskApplier:
    """The PDF pipeline masks PdfParagraph.unicode but sends the LLM text
    rebuilt from the paragraph's *characters*, which never carried the
    masking. MaskApplier replays the masking pass's decisions onto that
    rebuilt text."""

    _ROWS = [
        {
            "token": "__DLP_TOKEN_0001__",
            "original_value": "Mustermanns",
            "info_type": "PERSON_NAME",
        },
        {
            "token": "__DLP_TOKEN_0002__",
            "original_value": "Max Mustermann",
            "info_type": "PERSON_NAME",
        },
        {"token": "__DLP_TOKEN_0003__", "original_value": "45", "info_type": "AGE"},
    ]

    def test_values_are_masked(self):
        applier = build_mask_applier(self._ROWS)
        assert (
            applier.apply("Unterschrift des Mustermanns, Alter 45.")
            == "Unterschrift des __DLP_TOKEN_0001__, Alter __DLP_TOKEN_0003__."
        )

    def test_longest_value_wins(self):
        """ "Max Mustermann" must not be masked as "Mustermanns" first."""
        applier = build_mask_applier(self._ROWS)
        assert applier.apply("Max Mustermann") == "__DLP_TOKEN_0002__"

    def test_short_value_does_not_match_inside_a_longer_word(self):
        applier = build_mask_applier(self._ROWS)
        assert applier.apply("Rechnung 1452 vom Montag") == "Rechnung 1452 vom Montag"

    def test_applying_twice_is_idempotent(self):
        applier = build_mask_applier(self._ROWS)
        once = applier.apply("Herr Mustermanns")
        assert applier.apply(once) == once

    def test_masking_round_trips_through_restore(self):
        applier = build_mask_applier(self._ROWS)
        text = "Max Mustermann ist 45 Jahre alt."
        restored, _ = restore_tokens(applier.apply(text), build_token_map(self._ROWS))
        assert restored == text

    def test_no_token_rows_yields_an_inactive_applier(self):
        applier = build_mask_applier([])
        assert not applier
        assert applier.apply("Max Mustermann") == "Max Mustermann"

    def test_missing_info_types_reports_a_value_that_could_not_be_masked(self):
        """A placeholder inserted mid-value stops it being re-masked, so the
        token assigned to the paragraph never appears in the LLM input."""
        applier = build_mask_applier(self._ROWS)
        reference = "Unterschrift des __DLP_TOKEN_0001__"
        masked = applier.apply("Unterschrift des Muster{v1}manns")
        assert applier.missing_info_types(reference, masked) == ["PERSON_NAME"]

    def test_missing_info_types_is_empty_when_everything_was_masked(self):
        applier = build_mask_applier(self._ROWS)
        reference = "Unterschrift des __DLP_TOKEN_0001__"
        masked = applier.apply("Unterschrift des Mustermanns")
        assert applier.missing_info_types(reference, masked) == []
