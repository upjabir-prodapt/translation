"""Tests for the shared language-detection primitives (Phase C.1).

These primitives are shared by `JobProcessor` (PDF) and
`LanguageDetectionService` (DOCX/TXT) so both pipelines apply the exact
same confidence floor and thresholds.
"""

from collections import Counter
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langdetect import LangDetectException
from src.worker.services import language_detection_core as core


class TestIsDetectableText:
    def test_long_alpha_text_is_detectable(self):
        assert core.is_detectable_text("Valid text that is long enough") is True

    def test_short_text_is_not_detectable(self):
        assert core.is_detectable_text("123") is False

    def test_text_with_too_few_alpha_chars_is_not_detectable(self):
        # 20+ chars but almost no alphabetic content.
        assert core.is_detectable_text("1234567890 !@#$%^&*()") is False


class TestNormalizeDetectedLanguage:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("zh-cn", "zh"), ("zh-tw", "zh"), ("iw", "he"), ("en", "en"), ("FR", "fr")],
    )
    def test_aliases_and_passthrough(self, raw, expected):
        assert core.normalize_detected_language(raw) == expected


class TestDetectLanguageForText:
    def test_low_confidence_returns_none(self):
        mock_candidate = MagicMock(lang="en", prob=0.1)
        with patch.object(core, "detect_langs", return_value=[mock_candidate]):
            assert core.detect_language_for_text("some text") is None

    def test_confident_match_returns_normalized_language(self):
        mock_candidate = MagicMock(lang="zh-cn", prob=0.95)
        with patch.object(core, "detect_langs", return_value=[mock_candidate]):
            assert core.detect_language_for_text("some text") == "zh"

    def test_exception_returns_none(self):
        with patch.object(
            core, "detect_langs", side_effect=LangDetectException(0, "Error")
        ):
            assert core.detect_language_for_text("some text") is None

    def test_empty_candidates_returns_none(self):
        with patch.object(core, "detect_langs", return_value=[]):
            assert core.detect_language_for_text("some text") is None


class TestAggregateLanguages:
    def test_returns_dominant_language(self):
        counter = Counter({"de": 700, "en": 300})
        assert core.aggregate_languages(counter) == "de"

    def test_empty_counter_returns_none(self):
        assert core.aggregate_languages(Counter()) is None


class TestGetSupportedLanguages:
    def test_matches_language_mapper_values(self):
        supported = core.get_supported_languages()
        assert "en" in supported
        assert "fr" in supported
        assert "de" in supported
        assert "es" in supported
        assert "it" in supported
        assert "ja" in supported
        assert "zh" in supported
        # A language outside the mapper must not be considered supported.
        assert "nl" not in supported
