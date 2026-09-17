"""The gate's behaviour must come from data and settings, not from its source.

These tests exist because the first version of the gate was reverse-engineered
from one UAT round: ~840 words typed into Python across six hardcoded
languages. It worked on the documents that produced it and would have degraded
silently on anything else -- a seventh language got no coverage and said
nothing, and editing a prompt quietly disabled the rule that catches the model
quoting that prompt back.

So each test here changes *data* or *configuration* and asserts the behaviour
follows. If any of them starts needing a source edit to pass, the vocabulary
has crept back into the code.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from src.config import linguistic_data
from src.config.glossary_hygiene import HygieneContext
from src.config.glossary_hygiene import Trust
from src.config.glossary_hygiene import evaluate_term_pair


@pytest.fixture(autouse=True)
def _clean_caches():
    linguistic_data.reset_caches()
    yield
    linguistic_data.reset_caches()


def _with_vocabulary(payload: dict):
    """Swap the vocabulary data file for `payload`."""
    return patch.object(
        linguistic_data,
        "_raw_vocabulary",
        lambda: payload,
    )


class TestLanguageCoverageIsDataDriven:
    def test_a_new_language_needs_no_code_change(self):
        """Portuguese is not in the shipped data; adding it must just work.

        The UAT round asked Colt to add Portuguese and Dutch. Under the
        hardcoded version those languages would have had zero closed-class
        coverage, and nothing would have said so.
        """
        assert evaluate_term_pair("embora", "although").accepted

        with _with_vocabulary({"function_words": {"pt": ["embora", "contudo"]}}):
            linguistic_data.reset_caches()
            verdict = evaluate_term_pair("embora", "although")
        assert not verdict.accepted
        assert verdict.reason == "function_word"

    def test_missing_coverage_is_reported_not_silent(self):
        with _with_vocabulary({"function_words": {"en": ["the", "and"]}}):
            linguistic_data.reset_caches()
            with patch.object(
                linguistic_data,
                "supported_languages",
                lambda: frozenset({"en", "pt", "nl"}),
            ):
                linguistic_data.languages_missing_coverage.cache_clear()
                missing = linguistic_data.languages_missing_coverage()
        assert missing == ("nl", "pt")

    def test_gate_still_works_when_the_data_file_is_unreadable(self):
        """Losing the data must degrade, not crash or open the floodgates."""
        with _with_vocabulary({}):
            linguistic_data.reset_caches()
            # Structural rules carry on regardless.
            assert evaluate_term_pair("der", "der").reason == "identity"
            assert evaluate_term_pair("__DLP_TOKEN_0001__", "x").reason == "placeholder"
            # Vocabulary-dependent rules simply stop firing, rather than
            # rejecting everything or throwing.
            assert evaluate_term_pair("their", "seu").accepted


class TestThresholdsComeFromSettings:
    @pytest.mark.parametrize(
        ("setting", "value", "source", "target", "reason"),
        [
            (
                "GLOSSARY_MAX_TERM_WORDS",
                3,
                "one two three four",
                "x y",
                "too_many_words",
            ),
            ("GLOSSARY_MAX_TERM_CHARS", 10, "a very long term indeed", "x", "too_long"),
            ("GLOSSARY_MIN_TOKEN_CHARS", 12, "router", "enrutador", "too_short"),
        ],
    )
    def test_threshold_change_is_honoured(self, setting, value, source, target, reason):
        from src.config.constants import settings

        assert evaluate_term_pair(source, target).reason != reason
        with patch.object(settings, setting, value):
            assert evaluate_term_pair(source, target).reason == reason


class TestPromptEchoTracksThePrompt:
    def test_phrases_are_derived_from_the_real_prompt_text(self):
        """A phrase is rejected because the prompt says it, not because it was typed here."""
        assert "named entities" in linguistic_data.prompt_echo_terms()
        assert evaluate_term_pair("named entities", "entidades").reason == "prompt_echo"

    def test_editing_the_prompt_changes_what_is_rejected(self):
        with patch.object(
            linguistic_data,
            "prompt_echo_terms",
            lambda: frozenset({"fictional marker phrase"}),
        ):
            assert (
                evaluate_term_pair("fictional marker phrase", "x").reason
                == "prompt_echo"
            )
            # And the previously-derived phrase is no longer special.
            assert evaluate_term_pair("named entities", "entidades").accepted


class TestMonthNamesComeFromCldr:
    @pytest.mark.parametrize(
        "term", ["02-abril-2023", "02-Aprile-2023", "28 Dezember", "16 October 2026"]
    )
    def test_month_literals_in_any_supported_locale(self, term):
        assert evaluate_term_pair(term, "x").reason == "date_literal"

    def test_month_vocabulary_is_not_hand_maintained(self):
        months = linguistic_data.month_names()
        # Far more than any hand-typed alternation would carry, and covering
        # locales nobody enumerated by hand.
        assert len(months) > 60
        assert {"abril", "aprile", "oktober", "avril"} <= months


class TestDomainVocabularyIsDerived:
    def test_glossary_terms_teach_the_name_heuristic(self):
        """`Legal Entity` and `Elon Musk` are the same shape; vocabulary separates them."""
        empty = HygieneContext(domain_vocabulary=frozenset())
        with patch.object(linguistic_data, "domain_vocabulary_seed", frozenset):
            # With no vocabulary at all, both read as names.
            assert (
                evaluate_term_pair("Widget Frobnicator", "x", context=empty).reason
                == "person_name"
            )
            # Supplying trusted terminology that uses those words settles it.
            learned = HygieneContext.from_terms(
                ["Frobnicator configuration", "Widget assembly"]
            )
            assert evaluate_term_pair(
                "Widget Frobnicator", "Frobnicador", context=learned
            ).accepted

    def test_seed_is_a_floor_not_an_alternative(self):
        """Supplying domain vocabulary must not discard the general seed.

        Treating the derived set as a replacement made `Account Executive`
        start reading as a personal name the moment any context was passed.
        """
        context = HygieneContext.from_terms(["Colt IQ Network"])
        assert evaluate_term_pair(
            "Account Executive", "ejecutivo de cuenta", context=context
        ).accepted


class TestTrustLevels:
    def test_identity_is_rejected_for_curated_pairs_too(self):
        """A human typing an identity pair has still made the original mistake."""
        assert (
            evaluate_term_pair("der", "der", trust=Trust.CURATED).reason == "identity"
        )

    def test_curated_pairs_skip_vocabulary_heuristics(self):
        """`The Hague -> La Haya` is a deliberate call, not a leaked name."""
        assert evaluate_term_pair("The Hague", "La Haya").reason == "person_name"
        assert evaluate_term_pair("The Hague", "La Haya", trust=Trust.CURATED).accepted

    def test_preserve_entries_may_be_identities(self):
        assert not evaluate_term_pair("Colt", "Colt").accepted
        assert evaluate_term_pair("Colt", "Colt", trust=Trust.PRESERVE).accepted

    def test_preserve_entries_still_reject_corruption(self):
        assert (
            evaluate_term_pair(
                "__DLP_TOKEN_0001__", "__DLP_TOKEN_0001__", trust=Trust.PRESERVE
            ).reason
            == "placeholder"
        )

    @pytest.mark.parametrize(
        ("origin", "expected"),
        [
            ("curated", Trust.CURATED),
            ("learned", Trust.LEARNED),
            ("", Trust.LEARNED),
            (None, Trust.LEARNED),
            ("something else", Trust.LEARNED),
        ],
    )
    def test_unknown_origin_defaults_to_least_trust(self, origin, expected):
        assert Trust.from_origin(origin) is expected


def test_shipped_vocabulary_file_is_well_formed():
    payload = json.loads(linguistic_data.VOCABULARY_PATH.read_text(encoding="utf-8"))
    assert payload["function_words"], "no closed-class vocabulary shipped"
    for code, words in payload["function_words"].items():
        assert isinstance(code, str) and 2 <= len(code) <= 3, code
        assert all(isinstance(w, str) and w for w in words), code
