"""Regression tests for UAT S-03 (D-09) and EC-08 (D-03).

D-09  a one-word status cell ("Live") was passed through untranslated,
      because the minimum-length guard applied to any short string rather
      than to short strings with nothing to translate.
D-03  every batch was an independent LLM call with nothing tying one
      batch's terminology to another's, so a 20-page agreement rendered
      the same defined term two different ways.
"""

from __future__ import annotations

from src.worker.doctranslator.format.docx.paragraph_translator import (
    DocxParagraphTranslator,
)
from src.worker.doctranslator.format.docx.paragraph_translator import (
    build_binding_terminology_block,
)
from src.worker.doctranslator.format.docx.units import TranslatableUnit


class _StubTranslator:
    lang_in = "en"
    domain = None
    provider = "gemini_vertexai"
    model = "gemini-3.5-flash"

    def __init__(self) -> None:
        self.calls: list[str] = []


def _unit(text: str, unit_id: int = 0) -> TranslatableUnit:
    return TranslatableUnit(
        unit_id=unit_id, paragraph=None, label="table_cell", text=text
    )


class TestShortUnitsReachTheModel:
    """D-09: short *words* are translatable; short punctuation is not."""

    def _translator(self) -> DocxParagraphTranslator:
        return DocxParagraphTranslator(_StubTranslator(), "fr")

    def test_short_word_is_not_skipped(self):
        translator = self._translator()
        assert translator._should_skip_llm(_unit("Live")) is False

    def test_single_letter_is_not_skipped(self):
        translator = self._translator()
        assert translator._should_skip_llm(_unit("a")) is False

    def test_short_non_word_is_still_skipped(self):
        translator = self._translator()
        for text in ("--", "()", "  ", "12", "3.5"):
            assert translator._should_skip_llm(_unit(text)) is True, text

    def test_numeric_and_placeholder_filters_still_apply(self):
        translator = self._translator()
        assert translator._should_skip_llm(_unit("1 200,00")) is True
        assert translator._should_skip_llm(_unit("__DLP_TOKEN_0001__")) is True


class TestBindingTerminologyBlock:
    """D-03: the same resolved terms are handed to every batch."""

    def test_terms_are_rendered_as_binding_pairs(self):
        block = build_binding_terminology_block(
            [("the Supplier", "le Fournisseur"), ("the Customer", "le Client")],
            "French",
        )
        assert "the Supplier -> le Fournisseur" in block
        assert "the Customer -> le Client" in block
        assert "at every occurrence" in block

    def test_first_rendering_of_a_term_wins(self):
        """A term the extractor returned twice must not produce a prompt
        that itself offers two renderings."""
        block = build_binding_terminology_block(
            [("the Supplier", "le Fournisseur"), ("the Supplier", "le Prestataire")],
            "French",
        )
        assert "le Fournisseur" in block
        assert "le Prestataire" not in block

    def test_empty_pairs_are_dropped(self):
        block = build_binding_terminology_block(
            [("", "le Client"), ("NOC", "")], "French"
        )
        assert block == ""

    def test_approved_do_not_translate_terms_get_their_own_list(self):
        """UAT EC-01 (D-04): the business declares which names are written
        the same way in every language."""
        block = build_binding_terminology_block(
            None, "German", do_not_translate=["Colt IQ Network", "IP VPN"]
        )
        assert "Copy these exactly as written" in block
        assert "- Colt IQ Network" in block
        assert "- IP VPN" in block
        assert "->" not in block

    def test_extractor_identity_pairs_never_protect_a_word(self):
        """UAT S-03: an identity pair from automatic term extraction means
        the extractor had no translation, not that the word is protected.
        Treating one as protected left "Owner" and "Pending" in English."""
        block = build_binding_terminology_block(
            [("Owner", "Owner"), ("Pending", "Pending")], "French"
        )
        assert block == ""

    def test_renderings_and_do_not_translate_terms_coexist(self):
        block = build_binding_terminology_block(
            [("the Supplier", "le Fournisseur")],
            "French",
            do_not_translate=["SD-WAN"],
        )
        assert "the Supplier -> le Fournisseur" in block
        assert "- SD-WAN" in block

    def test_block_is_deterministically_ordered(self):
        """UAT S-02: an unordered block made otherwise identical
        submissions of the same document produce different prompts."""
        forward = build_binding_terminology_block(
            [("b term", "b terme"), ("a term", "a terme")], "French"
        )
        reverse = build_binding_terminology_block(
            [("a term", "a terme"), ("b term", "b terme")], "French"
        )
        assert forward == reverse

    def test_no_terms_leaves_the_prompt_untouched(self):
        assert build_binding_terminology_block(None, "French") == ""
        assert build_binding_terminology_block([], "French") == ""

    def test_block_is_capped(self):
        pairs = [(f"term{i}", f"terme{i}") for i in range(500)]
        block = build_binding_terminology_block(pairs, "French", max_terms=10)
        assert block.count(" -> ") == 10

    def test_terms_reach_the_batch_prompt(self):
        from src.worker.doctranslator.format.docx.paragraph_translator import (
            _build_prompt,
        )

        prompt = _build_prompt(
            [_unit("The Supplier shall provide the Services.", 1)],
            "French",
            glossary_terms=[("the Supplier", "le Fournisseur")],
        )
        assert "Binding Terminology" in prompt
        assert "the Supplier -> le Fournisseur" in prompt

    def test_prompt_without_terms_has_no_terminology_section(self):
        from src.worker.doctranslator.format.docx.paragraph_translator import (
            _build_prompt,
        )

        prompt = _build_prompt([_unit("Hello.", 1)], "French")
        assert "Binding Terminology" not in prompt


class TestVerbatimRulesInPrompts:
    """D-04: identifiers, product names and numeric literals are pinned."""

    def test_batch_prompt_carries_the_verbatim_rules(self):
        from src.worker.doctranslator.format.docx.paragraph_translator import (
            _build_prompt,
        )

        prompt = _build_prompt([_unit("Colt IQ Network delivers IP VPN.", 1)], "German")
        assert "Numeric literals" in prompt
        assert "decimal point for a comma" in prompt

    def test_single_unit_prompt_carries_the_verbatim_rules(self):
        from src.worker.doctranslator.translator.prompts import build_translation_prompt

        prompt = build_translation_prompt("99.99% SLA", "English", "German")
        assert "Numeric literals" in prompt
        assert "copied exactly as written in the source" in prompt


class TestRepeatedClauseMemo:
    """D-03: a clause repeated through a long agreement must come back
    worded identically every time, not re-invented per batch."""

    def _translator(self) -> DocxParagraphTranslator:
        return DocxParagraphTranslator(_StubTranslator(), "fr")

    def test_clauses_differing_only_by_number_are_translated_once(self):
        translator = self._translator()
        units = [
            _unit("2.1 Subject to clause 12, the Supplier shall provide.", 1),
            _unit("3.1 Subject to clause 12, the Supplier shall provide.", 2),
            _unit("4.1 An entirely different obligation applies.", 3),
        ]
        to_translate, repeats = translator._extract_repeated_units(units)

        assert [u.unit_id for u in to_translate] == [1, 3]
        assert repeats == {2: (1, "3.1 ", "2.1 ")}

    def test_repeat_reuses_the_translation_with_its_own_clause_number(self):
        translator = self._translator()
        units = [
            _unit("2.1 Subject to clause 12, the Supplier shall provide.", 1),
            _unit("3.1 Subject to clause 12, the Supplier shall provide.", 2),
        ]
        _, repeats = translator._extract_repeated_units(units)

        results = {1: "2.1 Sous réserve de la clause 12, le Prestataire fournit."}
        translator._apply_repeated_units(results, repeats)

        assert results[2] == "3.1 Sous réserve de la clause 12, le Prestataire fournit."

    def test_unlabelled_twin_gets_the_body_without_a_number(self):
        translator = self._translator()
        units = [
            _unit("2.1 The Supplier shall provide the Services.", 1),
            _unit("The Supplier shall provide the Services.", 2),
        ]
        _, repeats = translator._extract_repeated_units(units)

        results = {1: "2.1 Le Prestataire fournit les Services."}
        translator._apply_repeated_units(results, repeats)

        assert results[2] == "Le Prestataire fournit les Services."

    def test_distinct_text_is_never_collapsed(self):
        translator = self._translator()
        units = [
            _unit("2.1 The Supplier shall provide the Services.", 1),
            _unit("2.2 The Customer shall pay the Charges.", 2),
        ]
        to_translate, repeats = translator._extract_repeated_units(units)

        assert [u.unit_id for u in to_translate] == [1, 2]
        assert repeats == {}

    def test_whitespace_differences_do_not_defeat_the_memo(self):
        translator = self._translator()
        units = [
            _unit("2.1 The Supplier shall provide  the Services.", 1),
            _unit("3.1 The Supplier shall provide the  Services.", 2),
        ]
        to_translate, repeats = translator._extract_repeated_units(units)

        assert [u.unit_id for u in to_translate] == [1]
        assert 2 in repeats

    def test_missing_representative_translation_leaves_the_repeat_alone(self):
        """A batch failure must not silently blank a repeated clause."""
        translator = self._translator()
        units = [
            _unit("2.1 The Supplier shall provide.", 1),
            _unit("3.1 The Supplier shall provide.", 2),
        ]
        _, repeats = translator._extract_repeated_units(units)

        results: dict[int, str] = {}
        translator._apply_repeated_units(results, repeats)

        assert 2 not in results


class TestTerminologyIsPartOfTheCacheKey:
    """The pinned glossary changes the prompt, so it must change the key --
    otherwise a document translated under one approved glossary is served
    from a cache entry produced under a different one."""

    def _key(self, **kwargs) -> str:
        translator = DocxParagraphTranslator(_StubTranslator(), "fr", **kwargs)
        return translator._unit_cache_key(_unit("The Supplier shall provide.", 1))

    def test_different_glossaries_produce_different_keys(self):
        plain = self._key()
        pinned = self._key(glossary_terms=[("the Supplier", "le Fournisseur")])
        other = self._key(glossary_terms=[("the Supplier", "le Prestataire")])

        assert plain != pinned
        assert pinned != other

    def test_same_glossary_produces_the_same_key(self):
        first = self._key(glossary_terms=[("the Supplier", "le Fournisseur")])
        second = self._key(glossary_terms=[("the Supplier", "le Fournisseur")])
        assert first == second

    def test_do_not_translate_list_changes_the_key(self):
        plain = self._key()
        protected = self._key(do_not_translate=["Colt IQ Network"])
        assert plain != protected
