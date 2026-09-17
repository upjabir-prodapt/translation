"""Tests for the secondary-language prompt block (config/language_prompts.py).

The block exists because the pipeline translates a whole document from one
source language: on a 97% French document with an English cover page, the
English lines were handed to the model as French. These tests pin both
halves of the contract -- which languages get named, and that a monolingual
document's prompt is left exactly as it was.
"""

from collections import Counter

import pytest
from src.config.language_prompts import build_secondary_language_block
from src.config.language_prompts import describe_language
from src.config.language_prompts import format_share
from src.config.language_prompts import select_secondary_languages

# The real distribution measured from index_fr.pdf: a French document whose
# English cover page and German contract terms both sit far below the 5%
# noise share the mixed-language guard uses.
REAL_FRENCH_DOCUMENT = Counter({"fr": 11554, "en": 217, "de": 90})


class TestSelectSecondaryLanguages:
    def test_real_document_names_both_minority_languages(self):
        selected = select_secondary_languages(
            REAL_FRENCH_DOCUMENT, primary_language="fr", target_language="es"
        )
        assert [code for code, _share in selected] == ["en", "de"]

    def test_sub_noise_share_languages_are_still_named(self):
        """The languages this block exists for are exactly the ones the
        mixed-language decision discards as incidental, so the floor here
        must sit well below LANGUAGE_DETECTION_NOISE_SHARE."""
        selected = dict(
            select_secondary_languages(
                REAL_FRENCH_DOCUMENT, primary_language="fr", target_language="es"
            )
        )
        assert selected["de"] < 0.05

    def test_primary_language_is_excluded(self):
        selected = select_secondary_languages(
            REAL_FRENCH_DOCUMENT, primary_language="fr", target_language="es"
        )
        assert "fr" not in dict(selected)

    def test_target_language_is_excluded(self):
        """ "Translate this from Spanish into Spanish" has no useful reading."""
        selected = select_secondary_languages(
            REAL_FRENCH_DOCUMENT, primary_language="fr", target_language="en"
        )
        assert "en" not in dict(selected)
        assert [code for code, _share in selected] == ["de"]

    def test_languages_below_the_floor_are_dropped(self):
        counter = Counter({"fr": 10000, "en": 500, "tl": 10})
        selected = dict(
            select_secondary_languages(
                counter, primary_language="fr", target_language="es", min_share=0.01
            )
        )
        assert "en" in selected
        assert "tl" not in selected

    def test_result_is_capped_and_ordered_by_share(self):
        counter = Counter({"fr": 10000, "en": 900, "de": 800, "it": 700, "pt": 600})
        selected = select_secondary_languages(
            counter,
            primary_language="fr",
            target_language="es",
            min_share=0.0,
            max_languages=2,
        )
        assert [code for code, _share in selected] == ["en", "de"]

    def test_monolingual_document_selects_nothing(self):
        selected = select_secondary_languages(
            Counter({"fr": 10000}), primary_language="fr", target_language="es"
        )
        assert selected == []

    @pytest.mark.parametrize("distribution", [None, Counter(), Counter({"fr": 0})])
    def test_empty_or_zero_distribution_is_safe(self, distribution):
        assert (
            select_secondary_languages(
                distribution, primary_language="fr", target_language="es"
            )
            == []
        )


class TestFormatShare:
    def test_small_shares_do_not_render_as_zero(self):
        """A 0.8% share rendering as "about 0%" would contradict the list
        it appears in by saying the language is not there."""
        assert format_share(0.0076) == "under 1%"

    @pytest.mark.parametrize(
        ("share", "expected"),
        [(0.0183, "about 1.8%"), (0.42, "about 42%"), (0.0999, "about 10.0%")],
    )
    def test_larger_shares_keep_a_figure(self, share, expected):
        assert format_share(share) == expected

    def test_qualifier_is_never_doubled(self):
        """Guards the "about under 1%" phrasing bug: the qualifier lives
        here, so no caller may add its own."""
        for share in (0.0001, 0.005, 0.02, 0.5):
            rendered = format_share(share)
            assert not rendered.startswith("about under")


class TestDescribeLanguage:
    def test_known_language_gets_a_name_and_code(self):
        assert describe_language("fr") == "French (fr)"

    def test_unnamed_code_is_left_bare(self):
        """`get_language_display_name` title-cases unknown codes ("nl" ->
        "Nl"), which reads as a typo; the bare code is clearer."""
        assert describe_language("nl") == "nl"


class TestBuildSecondaryLanguageBlock:
    @pytest.fixture
    def block(self):
        return build_secondary_language_block(
            select_secondary_languages(
                REAL_FRENCH_DOCUMENT, primary_language="fr", target_language="es"
            ),
            primary_language="fr",
            target_language="es",
        )

    def test_monolingual_document_renders_nothing(self):
        """The block is called unconditionally from every prompt builder,
        so a monolingual document must add nothing to its prompt."""
        assert (
            build_secondary_language_block(
                [], primary_language="fr", target_language="es"
            )
            == ""
        )

    def test_names_every_selected_language(self, block):
        assert "English (en)" in block
        assert "German (de)" in block

    def test_states_the_shares(self, block):
        assert "about 1.8%" in block
        assert "under 1%" in block

    def test_instructs_the_model_to_judge_each_segment(self, block):
        assert "Judge from the text itself" in block

    def test_instructs_translation_from_the_secondary_language(self, block):
        assert "translate it from that language into Spanish (es)" in block

    def test_keeps_the_primary_language_as_the_default(self, block):
        assert "Otherwise treat it as French (fr)" in block

    def test_forbids_leaving_a_segment_untranslated(self, block):
        """Without this the model can read "this is not the source
        language" as licence to pass the segment through unchanged."""
        assert "never leave one untranslated" in block
        assert "never copy it through unchanged" in block

    def test_marks_the_shares_as_document_wide_hints(self, block):
        """The shares describe the document, not the batch in the prompt,
        so the model must not treat them as per-segment labels."""
        assert "describe the whole document" in block
        assert "trust the text in front of you" in block
