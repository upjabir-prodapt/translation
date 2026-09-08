"""Tests for the shared language-detection primitives (Phase C.1).

These primitives are shared by `JobProcessor` (PDF) and
`LanguageDetectionService` (DOCX/TXT) so both pipelines apply the exact
same confidence floor and thresholds.
"""

from collections import Counter
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from lingua import Language
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
        [
            # zh-cn/zh-tw are aliased to "zh" by language_mapper.json itself
            # (normalize_detected_language() looks it up directly rather
            # than keeping a second, hardcoded alias table).
            ("zh-cn", "zh"),
            ("zh-tw", "zh"),
            ("en", "en"),
            ("FR", "fr"),
        ],
    )
    def test_aliases_and_passthrough(self, raw, expected):
        assert core.normalize_detected_language(raw) == expected

    def test_code_outside_the_mapper_passes_through_unchanged(self):
        """A detected code with no language_mapper.json entry at all (e.g.
        the deprecated `iw` for Hebrew, which isn't a supported language)
        must pass through unchanged rather than raise or get dropped -- it
        still needs to appear in the distribution as "detected but
        unsupported"."""
        assert core.normalize_detected_language("iw") == "iw"
        assert core.normalize_detected_language("ru") == "ru"


class TestDetectLanguageForText:
    def test_below_relative_distance_floor_returns_none(self):
        """lingua itself returns None when the top two candidates are
        within `LANGUAGE_DETECTION_MIN_RELATIVE_DISTANCE` of each other --
        that ambiguity must propagate as "no confident language", not a
        guess."""
        mock_detector = MagicMock()
        mock_detector.detect_language_of.return_value = None
        with patch.object(core, "_get_detector", return_value=mock_detector):
            assert core.detect_language_for_text("some text") is None

    def test_confident_match_returns_normalized_language(self):
        mock_detector = MagicMock()
        mock_detector.detect_language_of.return_value = Language.CHINESE
        with patch.object(core, "_get_detector", return_value=mock_detector):
            assert core.detect_language_for_text("some text") == "zh"

    def test_exception_returns_none(self):
        mock_detector = MagicMock()
        mock_detector.detect_language_of.side_effect = RuntimeError("boom")
        with patch.object(core, "_get_detector", return_value=mock_detector):
            assert core.detect_language_for_text("some text") is None


class TestAggregateLanguages:
    def test_returns_dominant_language(self):
        counter = Counter({"de": 700, "en": 300})
        assert core.aggregate_languages(counter) == "de"

    def test_empty_counter_returns_none(self):
        assert core.aggregate_languages(Counter()) is None


class TestLanguageShares:
    def test_shares_sum_to_one(self):
        shares = core.language_shares(Counter({"en": 750, "fr": 250}))
        assert shares == {"en": 0.75, "fr": 0.25}

    def test_empty_counter_returns_empty_dict(self):
        assert core.language_shares(Counter()) == {}


class TestSignificantLanguages:
    def test_sub_noise_share_languages_are_dropped(self):
        """The real Colt-brochure distribution: `ca`/`tl`/`de`/`es` are all
        one-block artefacts under the 5% floor and must not survive."""
        counter = Counter({"en": 3283, "ca": 120, "de": 96, "tl": 97, "es": 76})
        assert core.significant_languages(counter) == Counter({"en": 3283})

    def test_language_exactly_at_the_floor_is_kept(self):
        counter = Counter({"en": 950, "fr": 50})
        assert core.significant_languages(counter) == counter

    def test_fragmented_document_keeps_everything(self):
        """If every language is below the floor, dropping them all would
        turn "many languages, none dominant" into "no evidence", which is a
        different verdict with a different message."""
        counter = Counter({f"l{i}": 10 for i in range(50)})
        assert core.significant_languages(counter) == counter

    def test_empty_counter_stays_empty(self):
        assert core.significant_languages(Counter()) == Counter()


class TestSupportedLanguageShare:
    def test_mostly_english_with_supported_and_unsupported_artefacts(self):
        """Computed on the RAW distribution, not noise-filtered: `de` and
        `es` are themselves supported languages and count toward coverage
        even though they are individually below the 5% noise floor used
        elsewhere for the dominant-language decision; only `ca`/`tl` are
        genuinely unsupported. (3283+96+76)/3672 ~= 0.941 -- not 1.0, since
        noise-filtering no longer happens before this sum (see the
        function's docstring for why: several unsupported languages each
        below the noise floor must still count against the coverage
        budget)."""
        counter = Counter({"en": 3283, "ca": 120, "de": 96, "tl": 97, "es": 76})
        assert core.supported_language_share(counter) == pytest.approx(
            (3283 + 96 + 76) / 3672
        )

    def test_wholly_unsupported_document_scores_zero(self):
        assert core.supported_language_share(Counter({"ru": 1000})) == 0.0

    def test_fifty_fifty_supported_mix_is_fully_covered(self):
        """The core new behaviour: a genuinely mixed EN/FR document is
        translatable, so it must score 1.0, not be penalised for being
        mixed."""
        counter = Counter({"en": 500, "fr": 500})
        assert core.supported_language_share(counter) == pytest.approx(1.0)

    def test_partial_coverage_is_the_supported_fraction(self):
        counter = Counter({"en": 700, "ru": 300})
        assert core.supported_language_share(counter) == pytest.approx(0.7)

    def test_empty_counter_scores_zero(self):
        assert core.supported_language_share(Counter()) == 0.0


class TestCjkDetection:
    """PR #3's `is_high_signal_unit` counted whitespace-delimited words, so
    every CJK unit was discarded and Japanese/Chinese documents silently
    dropped out of the distribution. These run the real detector to make
    sure that class of regression cannot come back unnoticed."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "本サービスの利用に関して生じたいかなる損害についても、"
                "当社は一切の責任を負わないものとします。",
                "ja",
            ),
            (
                "本公司对因使用本服务而产生的任何损失概不负责，"
                "但因本公司故意或重大过失造成的除外。",
                "zh",
            ),
        ],
    )
    def test_cjk_prose_is_detected(self, text, expected):
        assert core.detect_language_for_text(text) == expected

    def test_cjk_units_reach_the_distribution_and_count_as_supported(self):
        japanese = (
            "本サービスの利用に関して生じたいかなる損害についても、"
            "当社は一切の責任を負わないものとします。"
        )
        english = (
            "The Company shall not be liable for any damages arising from "
            "the use of the services provided under this Agreement."
        )
        counter: Counter[str] = Counter()
        for text in (japanese, english):
            assert core.is_detectable_text(text)
            detected = core.detect_language_for_text(text)
            assert detected is not None
            counter[detected] += len(text)
        assert set(counter) == {"ja", "en"}
        assert core.supported_language_share(counter) == pytest.approx(1.0)


class TestBuildUnclassifiableTextMessage:
    def test_default_matches_docx_txt_wording(self):
        """Regression: must stay byte-identical to the wording
        language_detection_service.py used before this helper existed."""
        msg = core.build_unclassifiable_text_message(subject="DOCX text")
        assert msg == (
            "Unable to detect a source language: DOCX text contains "
            "readable text, but no passage was long or distinctive enough "
            "to identify its language with confidence. Please supply a "
            "document with more continuous prose."
        )

    def test_pdf_variant_matches_processor_service_wording(self):
        """Regression: must stay byte-identical to the wording
        processor_service.py used before this helper existed."""
        msg = core.build_unclassifiable_text_message(
            subject="This PDF", text_noun="extractable text", include_prefix=False
        )
        assert msg == (
            "This PDF contains extractable text, but no passage was long "
            "or distinctive enough to identify its language with "
            "confidence. Please supply a document with more continuous "
            "prose."
        )


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
