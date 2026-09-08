"""DOCX term extraction: CJK-aware batch sizing (B1 batch-sizing tuning)."""

from unittest.mock import MagicMock

import pytest
from src.worker.doctranslator.format.docx.term_extractor import DocxTermExtractor
from src.worker.doctranslator.format.docx.units import TranslatableUnit
from src.worker.doctranslator.glossary import ExtractedGlossaryTerm
from src.worker.doctranslator.translator.provider_types import LLMProvider


def _unit(unit_id: int, text: str = "hi") -> TranslatableUnit:
    return TranslatableUnit(
        unit_id=unit_id, paragraph=MagicMock(), label="text", text=text
    )


class _FakeEngine:
    """Minimal stand-in for BaseTranslator, avoiding real Gemini/Claude clients."""

    provider = LLMProvider.GEMINI_VERTEXAI
    model = "gemini-3.5-flash"
    lang_in = "en"

    def llm_translate(self, prompt, response_schema=None, batch_items=None):
        return "[]"


class TestCjkAwareBatchSizing:
    """Mirrors DocxParagraphTranslator's equivalent tests: term-extraction
    batches should scale by the same CJK token multiplier
    (translation_config.get_token_multiplier) the PDF pipeline uses instead
    of a flat paragraph cap.

    tests/test.env fixes LLM_TERM_EXTRACTION_BATCH_MAX_PARAGRAPHS=60,
    LLM_TOKEN_MULTIPLIER_DEFAULT=1.0, LLM_TOKEN_MULTIPLIER_CJK=0.5.
    """

    def _short_units(self, count: int) -> list[TranslatableUnit]:
        return [_unit(i) for i in range(count)]

    def test_non_cjk_pair_uses_default_multiplier(self):
        extractor = DocxTermExtractor(_FakeEngine(), "fr")
        assert extractor._token_multiplier == pytest.approx(1.0)
        # _batch_units closes a batch once it *exceeds* max_paragraphs (a
        # `>` rather than `>=` cutoff), so with a max of 60 the first batch
        # holds 61 units before rolling over.
        batches = extractor._batch_units(self._short_units(65))
        assert [len(b) for b in batches] == [61, 4]

    def test_cjk_target_language_shrinks_batches(self):
        extractor = DocxTermExtractor(_FakeEngine(), "zh")
        assert extractor._token_multiplier == pytest.approx(0.5)
        # Halved max (60 * 0.5 = 30) means noticeably smaller batches than
        # the non-CJK case above, still with the same off-by-one cutoff.
        batches = extractor._batch_units(self._short_units(65))
        assert [len(b) for b in batches] == [31, 31, 3]

    def test_cjk_source_language_also_shrinks_batches(self):
        class _CjkSourceEngine(_FakeEngine):
            lang_in = "ko"

        extractor = DocxTermExtractor(_CjkSourceEngine(), "fr")
        assert extractor._token_multiplier == pytest.approx(0.5)
        batches = extractor._batch_units(self._short_units(35))
        assert [len(b) for b in batches] == [31, 4]

    def test_detected_cjk_content_shrinks_batches_for_a_non_cjk_pair(self):
        """A mixed en->fr document that actually contains Japanese.

        Only reachable now that mixed documents are translated rather than
        rejected. Neither declared language is CJK, so without the detected
        distribution this would take the default multiplier and build
        batches tiktoken has under-counted.
        """
        extractor = DocxTermExtractor(
            _FakeEngine(), "fr", detected_languages=["en", "ja"]
        )
        assert extractor._token_multiplier == pytest.approx(0.5)

    def test_detected_non_cjk_content_leaves_the_default_multiplier(self):
        extractor = DocxTermExtractor(
            _FakeEngine(), "fr", detected_languages=["en", "de"]
        )
        assert extractor._token_multiplier == pytest.approx(1.0)


class TestDomainAwareTermExtraction:
    def test_term_extractor_domain_prompt_injection(self):
        class _CapturingEngine(_FakeEngine):
            def __init__(self):
                self.prompts: list[str] = []

            def llm_translate(self, prompt, response_schema=None, batch_items=None):
                self.prompts.append(prompt)
                return (
                    '[{"src": "indemnification", "tgt": "indemnisation", '
                    '"src_lang": "en"}]'
                )

        engine = _CapturingEngine()
        extractor = DocxTermExtractor(engine, "fr", domain="legal")
        assert extractor.domain == "legal"

        units = [_unit(1, "The contractor agrees to indemnification obligations.")]
        pairs = extractor._extract_batch(units)
        assert pairs == [
            ExtractedGlossaryTerm(
                source="indemnification", target="indemnisation", source_language="en"
            )
        ]
        assert len(engine.prompts) == 1
        assert "### Domain Context: Legal & Regulatory" in engine.prompts[0]
        assert "legal & regulatory terminology" in engine.prompts[0]


class TestMultiLanguageExtraction:
    """Extraction is no longer restricted to the job's declared source
    language: a mixed-language document's non-declared-language passages
    are legitimate domain content too, and are now extracted and tagged
    with their own actual language (self-reported by the model) instead of
    being silently ignored. See ExtractedGlossaryTerm and
    GlossaryService.load_domain_glossary's `source_languages` filter for
    why per-term attribution -- not the batch's declared language -- is
    what keeps broadening extraction safe for the shared domain glossary.
    """

    class _CapturingEngine(_FakeEngine):
        def __init__(self):
            self.prompts: list[str] = []
            self.next_response = "[]"

        def llm_translate(self, prompt, response_schema=None, batch_items=None):
            self.prompts.append(prompt)
            return self.next_response

    def test_prompt_lists_every_supported_language_not_just_declared(self):
        """`lang_in='en'` no longer narrows the prompt to English only."""
        engine = self._CapturingEngine()
        extractor = DocxTermExtractor(engine, "de")

        extractor._extract_batch(
            [
                _unit(1, "The contractor agrees to indemnification obligations."),
                _unit(2, "Le prestataire accepte les obligations d'indemnisation."),
            ]
        )
        prompt = engine.prompts[0]
        for code in ("en", "es", "fr", "de", "it", "ja", "zh"):
            assert code in prompt
        assert "src_lang" in prompt
        assert "Extract terms ONLY from passages written in" not in prompt

    def test_prompt_is_identical_regardless_of_declared_source_language(self):
        """The prompt used to interpolate `lang_in`; now it doesn't depend
        on it at all, so a Japanese-declared job and an English-declared
        job extracting the same text get byte-identical prompts."""

        class _JaEngine(self._CapturingEngine):
            lang_in = "ja"

        en_engine = self._CapturingEngine()
        ja_engine = _JaEngine()
        unit = [_unit(1, "Some reasonably long source text here.")]
        DocxTermExtractor(en_engine, "de")._extract_batch(unit)
        DocxTermExtractor(ja_engine, "de")._extract_batch(unit)
        assert en_engine.prompts[0] == ja_engine.prompts[0]

    def test_mixed_batch_tags_each_term_with_its_own_language(self):
        """One extraction batch over a mixed EN/FR document contributes
        terms in both languages, each correctly attributed."""
        engine = self._CapturingEngine()
        engine.next_response = (
            '[{"src": "indemnification", "tgt": "Freistellung", "src_lang": "en"},'
            ' {"src": "indemnisation", "tgt": "Freistellung", "src_lang": "fr"}]'
        )
        extractor = DocxTermExtractor(engine, "de")
        pairs = extractor._extract_batch(
            [
                _unit(1, "The contractor agrees to indemnification obligations."),
                _unit(2, "Le prestataire accepte les obligations d'indemnisation."),
            ]
        )
        assert set(pairs) == {
            ExtractedGlossaryTerm(
                source="indemnification",
                target="Freistellung",
                source_language="en",
            ),
            ExtractedGlossaryTerm(
                source="indemnisation", target="Freistellung", source_language="fr"
            ),
        }

    def test_term_with_unsupported_reported_language_is_dropped(self):
        """The model must self-report one of the listed languages; a value
        that doesn't normalize (typo, unsupported language, hallucinated
        code) is dropped rather than guessed at."""
        engine = self._CapturingEngine()
        engine.next_response = (
            '[{"src": "\\u0434\\u043e\\u0433\\u043e\\u0432\\u043e\\u0440",'
            ' "tgt": "contract", "src_lang": "ru"}]'
        )
        extractor = DocxTermExtractor(engine, "en")
        assert extractor._extract_batch([_unit(1, "some text")]) == []

    def test_term_with_missing_src_lang_is_dropped(self):
        engine = self._CapturingEngine()
        engine.next_response = '[{"src": "cat", "tgt": "Katze"}]'
        extractor = DocxTermExtractor(engine, "de")
        assert extractor._extract_batch([_unit(1, "some text")]) == []
