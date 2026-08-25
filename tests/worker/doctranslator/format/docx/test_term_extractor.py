"""DOCX term extraction: CJK-aware batch sizing (B1 batch-sizing tuning)."""

from unittest.mock import MagicMock

import pytest
from src.worker.doctranslator.format.docx.term_extractor import DocxTermExtractor
from src.worker.doctranslator.format.docx.units import TranslatableUnit
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


class TestDomainAwareTermExtraction:
    def test_term_extractor_domain_prompt_injection(self):
        class _CapturingEngine(_FakeEngine):
            def __init__(self):
                self.prompts: list[str] = []

            def llm_translate(self, prompt, response_schema=None, batch_items=None):
                self.prompts.append(prompt)
                return '[{"src": "indemnification", "tgt": "indemnisation"}]'

        engine = _CapturingEngine()
        extractor = DocxTermExtractor(engine, "fr", domain="legal")
        assert extractor.domain == "legal"

        units = [_unit(1, "The contractor agrees to indemnification obligations.")]
        pairs = extractor._extract_batch(units)
        assert pairs == [("indemnification", "indemnisation")]
        assert len(engine.prompts) == 1
        assert "### Domain Context: Legal & Regulatory" in engine.prompts[0]
        assert "legal & regulatory terminology" in engine.prompts[0]
