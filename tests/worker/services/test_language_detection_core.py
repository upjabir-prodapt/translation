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


def _confidence(language: Language, value: float) -> MagicMock:
    """Stand in for a lingua `ConfidenceValue`.

    `ConfidenceValue` is a Rust-backed type that cannot be constructed
    from Python, so tests that need to force a specific confidence mock
    the two attributes the code reads.
    """
    candidate = MagicMock()
    candidate.language = language
    candidate.value = value
    return candidate


def _patch_detector(**kwargs):
    """Swap in a stub lingua detector for the duration of one test.

    Patches the module attribute rather than the `lru_cache`, so the
    real detector's cached language models are never disturbed.
    """
    return patch.object(core, "_get_detector", return_value=MagicMock(**kwargs))


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
    def test_low_confidence_returns_none(self):
        with _patch_detector(
            compute_language_confidence_values=lambda _text: [
                _confidence(Language.ENGLISH, 0.1)
            ]
        ):
            assert core.detect_language_for_text("some text") is None

    def test_confident_match_returns_iso_639_1_code(self):
        with _patch_detector(
            compute_language_confidence_values=lambda _text: [
                _confidence(Language.CHINESE, 0.95)
            ]
        ):
            assert core.detect_language_for_text("some text") == "zh"

    def test_exception_returns_none(self):
        """Detection is advisory: a backend failure degrades to "no
        language" instead of failing the job."""
        with _patch_detector(
            compute_language_confidence_values=MagicMock(
                side_effect=RuntimeError("detector unavailable")
            )
        ):
            assert core.detect_language_for_text("some text") is None

    def test_empty_candidates_returns_none(self):
        with _patch_detector(compute_language_confidence_values=lambda _text: []):
            assert core.detect_language_for_text("some text") is None

    def test_unreadable_text_is_rejected_by_the_confidence_floor(self):
        """lingua returns a full zero-valued candidate list -- not an
        empty one -- for text it cannot read, so the floor is what has
        to reject it."""
        with _patch_detector(
            compute_language_confidence_values=lambda _text: [
                _confidence(Language.FRENCH, 0.0),
                _confidence(Language.POLISH, 0.0),
            ]
        ):
            assert core.detect_language_for_text("12345 !!!") is None


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


# Text taken verbatim from a real monolingual English company brochure
# that the previous distinct-language-count guard rejected. Every one of
# these is English. Under langdetect each drew a *different* non-English
# language at >=0.85 confidence, so only the length and shape rules could
# filter them; lingua scores them low enough that the confidence floor
# would catch them too. Both layers are asserted below, because layer 1
# is what makes the outcome independent of the detector.
ENGLISH_TEXT_MISREAD_AS_FOREIGN = [
    "THE EXTRAORDINARY EVERYDAY.",
    "SUPPORTING YOUR BUSINESS WITH OUR BACKBONE",
    "AWARD WINNING SUSTAINABILITY",
    "Busan Tokyo Osaka Yokohama Nagoya Kyoto Kobe",
    "Richmond Richmond Richmond",
    "Asia Office Tokyo, Osaka, Singapore, Hong Kong, Seoul",
    "Executive Manager Katsuya Oe (Vice President, Enterprise Sales - Asia) "
    "Yasutaka Mizutani (CMO, Executive Advisor - Asia)",
]

ENGLISH_PROSE = (
    "With a global reach but a deep local presence, the company is big enough "
    "to deliver and small enough to care."
)


class TestCaseNormalizeForDetection:
    def test_all_caps_text_is_lowercased(self):
        assert core.case_normalize_for_detection("HELLO WORLD") == "hello world"

    def test_mixed_case_text_is_untouched(self):
        assert core.case_normalize_for_detection(ENGLISH_PROSE) == ENGLISH_PROSE

    def test_text_without_letters_is_untouched(self):
        assert core.case_normalize_for_detection("1,100+") == "1,100+"

    def test_all_caps_english_headings_detect_as_english(self):
        """The all-caps failure mode, exercised against the real
        detector. langdetect matched ALL-CAPS English to foreign
        lowercase n-gram profiles at ~0.99999 and needed the
        normalization above to recover; lingua is case-insensitive. The
        headings must come back as English either way."""
        for heading in (
            "THE EXTRAORDINARY EVERYDAY. WE DELIVER EVERY SINGLE DAY.",
            "SUPPORTING YOUR BUSINESS WITH OUR BACKBONE NETWORK TODAY",
        ):
            assert core.detect_language_for_text(heading) == "en"

    def test_casing_does_not_change_the_detected_language(self):
        """Guards the property the normalization exists to provide,
        rather than the normalization itself."""
        assert core.detect_language_for_text(ENGLISH_PROSE) == "en"
        assert core.detect_language_for_text(ENGLISH_PROSE.upper()) == "en"


class TestIsProperNounList:
    @pytest.mark.parametrize(
        "text",
        [
            "Busan Tokyo Osaka Yokohama Nagoya Kyoto Kobe",
            "Richmond Richmond Richmond",
            "Asia Office Tokyo, Osaka, Singapore, Hong Kong, Seoul",
        ],
    )
    def test_name_runs_are_proper_noun_lists(self, text):
        assert core.is_proper_noun_list(text) is True

    def test_prose_is_not_a_proper_noun_list(self):
        assert core.is_proper_noun_list(ENGLISH_PROSE) is False

    def test_noun_capitalizing_language_prose_is_not_a_proper_noun_list(self):
        """German capitalizes every noun; the threshold must sit high
        enough that ordinary German prose still votes on the language."""
        german = (
            "Das Unternehmen bietet praemierte digitale Infrastruktur und "
            "Netzwerke fuer Unternehmen auf der ganzen Welt an."
        )
        assert core.is_proper_noun_list(german) is False


class TestIsHighSignalUnit:
    def test_prose_is_high_signal(self):
        assert core.is_high_signal_unit(ENGLISH_PROSE) is True

    def test_short_text_is_not_high_signal(self):
        assert core.is_high_signal_unit("Employees 400 employees") is False

    def test_long_but_few_words_is_not_high_signal(self):
        assert core.is_high_signal_unit("Amsterdam Rotterdam Luxembourg") is False

    def test_proper_noun_run_is_not_high_signal(self):
        text = (
            "Executive Manager Katsuya Oe (Vice President, Enterprise Sales - Asia) "
            "Yasutaka Mizutani (CMO, Executive Advisor - Asia)"
        )
        assert core.is_high_signal_unit(text) is False


class TestCountUnitLanguages:
    def test_monolingual_brochure_noise_is_excluded_entirely(self):
        """Regression: none of the misread English units may contribute a
        language, so a monolingual document cannot look multilingual."""
        counter, _chars = core.count_unit_languages(ENGLISH_TEXT_MISREAD_AS_FOREIGN)
        assert counter == Counter()

    @pytest.mark.parametrize("text", ENGLISH_TEXT_MISREAD_AS_FOREIGN)
    def test_brochure_noise_is_also_below_the_confidence_floor(self, text):
        """Layer 2 independently rejects every unit layer 1 drops, so a
        change to the length or shape thresholds cannot on its own let
        this noise back into the distribution."""
        assert core.detect_language_for_text(text) is None

    def test_prose_units_are_char_weighted(self):
        counter, chars = core.count_unit_languages([ENGLISH_PROSE])
        assert counter == Counter({"en": len(ENGLISH_PROSE)})
        assert chars == len(ENGLISH_PROSE)

    def test_max_chars_budget_stops_iteration(self):
        counter, chars = core.count_unit_languages(
            [ENGLISH_PROSE] * 10, max_chars=len(ENGLISH_PROSE)
        )
        assert chars == len(ENGLISH_PROSE)
        assert counter == Counter({"en": len(ENGLISH_PROSE)})


class TestSignificantLanguages:
    def test_incidental_languages_are_dropped(self):
        counter = Counter({"en": 3283, "ca": 120, "de": 96, "tl": 97, "es": 76})
        assert core.significant_languages(counter) == Counter({"en": 3283})

    def test_real_minority_language_is_retained(self):
        counter = Counter({"en": 700, "fr": 300})
        assert core.significant_languages(counter) == counter

    def test_all_below_floor_falls_back_to_full_distribution(self):
        counter = Counter({f"l{i}": 100 for i in range(30)})
        assert core.significant_languages(counter) == counter

    def test_empty_counter(self):
        assert core.significant_languages(Counter()) == Counter()


class TestResolveDominantLanguage:
    def test_dominant_language_wins(self):
        counter = Counter({"en": 8000, "fr": 2000})
        assert core.resolve_dominant_language(counter, source_label="doc") == "en"

    def test_monolingual_document_with_detector_noise_is_accepted(self):
        """The production regression, at the decision layer: sub-5%
        artefacts must not drag an English document below the bar."""
        counter = Counter({"en": 3283, "ca": 120, "de": 96, "tl": 97, "es": 76})
        assert core.resolve_dominant_language(counter, source_label="doc") == "en"

    def test_evenly_mixed_document_is_rejected(self):
        counter = Counter({"en": 5000, "fr": 4500})
        with pytest.raises(core.MixedLanguageError) as excinfo:
            core.resolve_dominant_language(counter, source_label="this PDF")
        assert "mixed-language" in str(excinfo.value)
        # The shares are attached for logging/telemetry, not merely
        # formatted into the message.
        assert set(excinfo.value.language_shares) == {"en", "fr"}

    def test_mixed_error_message_reports_shares_and_source(self):
        counter = Counter({"en": 5000, "fr": 5000})
        with pytest.raises(core.MixedLanguageError, match="en 50%, fr 50%"):
            core.resolve_dominant_language(counter, source_label="this PDF")

    def test_short_document_is_accepted_rather_than_called_mixed(self):
        """Below the evidence floor there is not enough text to call a
        document mixed, so the dominant language is used instead."""
        counter = Counter({"en": 60, "fr": 55})
        assert core.resolve_dominant_language(counter, source_label="doc") == "en"

    def test_empty_counter_asks_for_explicit_source_language(self):
        with pytest.raises(ValueError, match="choose the source language"):
            core.resolve_dominant_language(Counter(), source_label="doc")


class TestLanguageShares:
    def test_shares_sum_to_one_and_are_ordered(self):
        shares = core.language_shares(Counter({"en": 750, "fr": 250}))
        assert list(shares) == ["en", "fr"]
        assert shares["en"] == pytest.approx(0.75)
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_empty_counter_has_no_shares(self):
        assert core.language_shares(Counter()) == {}


class TestDetectLanguageOfJoinedText:
    def test_short_ambiguous_units_stay_undetectable(self):
        """EC-09 must survive the fallback: a few stray words are still
        reported as undetectable rather than resolved to a guess."""
        units = ["Information", "Total", "OK", "2026"]
        assert core.detect_language_of_joined_text(units) == Counter()

    def test_many_short_units_are_recovered(self):
        """A deck built entirely from short bullets has no high-signal
        unit, but its concatenation is unambiguous."""
        units = [
            "Network to network interfaces",
            "Live circuits to public cloud",
            "Buildings directly connected",
            "Connected cloud on ramps globally",
            "Key data centres, clouds and carrier hotels connected",
            "Languages supported by our customer service teams",
            "Countries where we currently do business",
            "We deliver a range of physical and digital services",
            "Consumption based usage enables real time control",
            "Strong partnerships with the world leading technology companies",
            "Global coverage via partner solutions with a secure network",
            "We value sustainability with ambitious science based targets",
        ]
        recovered = core.detect_language_of_joined_text(units)
        assert core.aggregate_languages(recovered) == "en"
