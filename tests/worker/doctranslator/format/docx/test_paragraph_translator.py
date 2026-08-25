"""DOCX paragraph batch translation: cache granularity + truncation tolerance."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.doctranslator.format.docx.paragraph_translator import (
    DocxParagraphTranslator,
)
from src.worker.doctranslator.format.docx.paragraph_translator import (
    _partial_parse_truncated_batch,
)
from src.worker.doctranslator.format.docx.units import TranslatableUnit
from src.worker.doctranslator.translator.provider_types import LLMProvider


def _unit(unit_id: int, text: str) -> TranslatableUnit:
    return TranslatableUnit(unit_id=unit_id, paragraph=MagicMock(), label="text", text=text)


class _FakeEngine:
    """Minimal stand-in for BaseTranslator, avoiding real Gemini/Claude clients."""

    provider = LLMProvider.GEMINI_VERTEXAI
    model = "gemini-3.5-flash"
    lang_in = "en"

    def __init__(self, llm_output: str | Exception):
        self._llm_output = llm_output
        self.calls: list[str] = []

    def llm_translate(self, prompt, response_schema=None, batch_items=None):
        self.calls.append(prompt)
        if isinstance(self._llm_output, Exception):
            raise self._llm_output
        return self._llm_output

    def translate(self, text, rate_limit_params=None):
        return f"[single:{text}]"


class TestSkipNonTranslatableUnits:
    """Pre-filter parity with the PDF pipeline's _is_paragraph_skippable().

    The 2026-08-24 baseline sent bare numbers and single characters to the
    LLM inside a ~3,381-char boilerplate prompt, then logged a length_ratio
    validation failure and burned a single-unit fallback call on each.
    """

    @staticmethod
    def _translator():
        return DocxParagraphTranslator(_FakeEngine("[]"), "de")

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "7", "42", "3.14", "12%", "1,234.56", "2024-01-01", "#", "<b1>"],
    )
    def test_skips_non_translatable(self, text):
        assert self._translator()._should_skip_llm(_unit(0, text)) is True

    @pytest.mark.parametrize(
        "text", ["Hello world", "Payment terms apply", "Item 7 is overdue"]
    )
    def test_keeps_real_prose(self, text):
        assert self._translator()._should_skip_llm(_unit(0, text)) is False

    @patch(
        "src.worker.doctranslator.format.docx.paragraph_translator.get_translation_cache"
    )
    def test_skipped_units_never_reach_the_llm(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get_many.return_value = {}
        mock_get_cache.return_value = mock_cache

        engine = _FakeEngine('[{"id": 1, "output": "hallo welt"}]')
        translator = DocxParagraphTranslator(engine, "de")
        results = translator.translate_all([_unit(0, "42"), _unit(1, "hello world")])

        # The numeric unit is passed through verbatim...
        assert results[0] == "42"
        assert translator.skipped_count == 1
        # ...and exactly one LLM call was made, for the real prose only.
        assert len(engine.calls) == 1
        assert "hello world" in engine.calls[0]


class TestPartialParseTruncatedBatch:
    def test_recovers_complete_items_from_truncated_array(self):
        raw = (
            '[{"id": 1, "output": "hallo"}, {"id": 2, "output": "welt"}, '
            '{"id": 3, "output": "unfinis'
        )
        items = _partial_parse_truncated_batch(raw)
        assert items == [
            {"id": 1, "output": "hallo"},
            {"id": 2, "output": "welt"},
        ]

    def test_returns_empty_list_when_nothing_recoverable(self):
        assert _partial_parse_truncated_batch("not json at all") == []

    def test_handles_escaped_quotes_in_output(self):
        raw = '[{"id": 1, "output": "say \\"hi\\""}]'
        items = _partial_parse_truncated_batch(raw)
        assert items == [{"id": 1, "output": 'say "hi"'}]


class TestCacheGranularity:
    @patch(
        "src.worker.doctranslator.format.docx.paragraph_translator.get_translation_cache"
    )
    def test_cache_hit_units_never_enter_a_batch(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get_many.side_effect = lambda keys: {
            key: "cached-de" for key in keys if "u0" in key
        }
        mock_get_cache.return_value = mock_cache

        engine = _FakeEngine('[{"id": 1, "output": "welt"}]')
        translator = DocxParagraphTranslator(engine, "de")

        # Force a deterministic cache key per unit id so the side_effect
        # above can distinguish them without depending on real sha256 output.
        with patch.object(
            translator,
            "_unit_cache_key",
            side_effect=lambda u: f"key-u{u.unit_id}",
        ):
            units = [_unit(0, "hello"), _unit(1, "world")]
            results = translator.translate_all(units)

        assert results[0] == "cached-de"
        assert results[1] == "welt"
        # Only the cache-miss unit's text should ever reach the LLM prompt.
        assert len(engine.calls) == 1
        assert "world" in engine.calls[0]
        assert "hello" not in engine.calls[0]

    @patch(
        "src.worker.doctranslator.format.docx.paragraph_translator.get_translation_cache"
    )
    def test_all_cache_hits_skips_llm_entirely(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get_many.side_effect = lambda keys: dict.fromkeys(keys, "cached-value")
        mock_get_cache.return_value = mock_cache

        engine = _FakeEngine("should not be called")
        translator = DocxParagraphTranslator(engine, "de")
        with patch.object(
            translator, "_unit_cache_key", side_effect=lambda u: f"key-u{u.unit_id}"
        ):
            results = translator.translate_all([_unit(0, "hello")])

        assert results == {0: "cached-value"}
        assert engine.calls == []


class TestCjkAwareBatchSizing:
    """B1: DOCX batch sizing should scale with the same CJK token multiplier
    the PDF pipeline uses (translation_config.get_token_multiplier), not a
    flat paragraph cap regardless of language pair.

    tests/test.env fixes LLM_TRANSLATION_BATCH_MAX_PARAGRAPHS=40,
    LLM_TOKEN_MULTIPLIER_DEFAULT=1.0, LLM_TOKEN_MULTIPLIER_CJK=0.5, so a
    non-CJK pair should batch up to 40 short units together while a CJK
    pair should split the same units into batches of (at most) 20.
    """

    def _short_units(self, count: int) -> list[TranslatableUnit]:
        # Very short text so the paragraph-count cap (not the token cap)
        # is what determines batch boundaries.
        return [_unit(i, "hi") for i in range(count)]

    def test_non_cjk_pair_uses_default_multiplier(self):
        translator = DocxParagraphTranslator(_FakeEngine("[]"), "fr")
        assert translator._token_multiplier == pytest.approx(1.0)
        # _batch_units closes a batch once it *exceeds* max_paragraphs (a
        # pre-existing `>` rather than `>=` cutoff), so with a max of 40 the
        # first batch holds 41 units before rolling over.
        batches = translator._batch_units(self._short_units(45))
        assert [len(b) for b in batches] == [41, 4]

    def test_cjk_target_language_shrinks_batches(self):
        translator = DocxParagraphTranslator(_FakeEngine("[]"), "zh")
        assert translator._token_multiplier == pytest.approx(0.5)
        # Halved max (40 * 0.5 = 20) means noticeably smaller batches than
        # the non-CJK case above, still with the same off-by-one cutoff.
        batches = translator._batch_units(self._short_units(45))
        assert [len(b) for b in batches] == [21, 21, 3]

    def test_cjk_source_language_also_shrinks_batches(self):
        class _CjkSourceEngine(_FakeEngine):
            lang_in = "ja"

        translator = DocxParagraphTranslator(_CjkSourceEngine("[]"), "fr")
        assert translator._token_multiplier == pytest.approx(0.5)
        batches = translator._batch_units(self._short_units(25))
        assert [len(b) for b in batches] == [21, 4]


class TestTruncatedBatchFallback:
    @patch(
        "src.worker.doctranslator.format.docx.paragraph_translator.get_translation_cache"
    )
    def test_truncated_response_only_falls_back_the_unrecovered_item(
        self, mock_get_cache
    ):
        mock_cache = MagicMock()
        mock_cache.get_many.return_value = {}
        mock_get_cache.return_value = mock_cache

        truncated = (
            '[{"id": 0, "output": "hallo"}, {"id": 1, "output": "unfinis'
        )
        engine = _FakeEngine(truncated)
        translator = DocxParagraphTranslator(engine, "de")
        with patch.object(
            translator, "_unit_cache_key", side_effect=lambda u: f"key-u{u.unit_id}"
        ):
            units = [_unit(0, "hello"), _unit(1, "world")]
            result = translator._translate_batch(units)

        assert result[0] == "hallo"
        # Unit 1 wasn't recoverable from the truncated JSON -> single-unit fallback.
        assert result[1] == "[single:world]"



class TestValidationRejectionReasonLogging:
    """E4: DOCX validation rejection-reason logging."""

    def test_empty_output_logs_empty_reason(self, caplog):
        translator = DocxParagraphTranslator(_FakeEngine("[]"), "fr")
        with caplog.at_level("WARNING"):
            is_rejected = translator._validate_translation("Hello world", "   ")
        assert is_rejected is True
        assert "DOCX translation validation failed (empty)" in caplog.text

    def test_same_text_logs_same_text_reason(self, caplog):
        translator = DocxParagraphTranslator(_FakeEngine("[]"), "fr")
        long_text = "This is a long sentence that should have more than ten tokens to trigger validation."
        with caplog.at_level("WARNING"):
            is_rejected = translator._validate_translation(long_text, long_text)
        assert is_rejected is True
        assert "DOCX translation validation failed (same_text)" in caplog.text

    def test_length_ratio_too_short_logs_length_ratio_reason(self, caplog):
        translator = DocxParagraphTranslator(_FakeEngine("[]"), "fr")
        long_text = "This is a very long sentence with many tokens describing an important technical architecture."
        short_output = "Oui"
        with caplog.at_level("WARNING"):
            is_rejected = translator._validate_translation(long_text, short_output)
        assert is_rejected is True
        assert "DOCX translation validation failed (length_ratio)" in caplog.text

    def test_edit_distance_too_small_logs_edit_distance_reason(self, caplog):
        translator = DocxParagraphTranslator(_FakeEngine("[]"), "fr")
        long_text = (
            "This is a very long sentence with many tokens describing an important technical "
            "architecture for a modern cloud distributed document translation microservice system."
        )
        # Alter just 2 characters (distance = 2 < 5)
        almost_same = long_text[:-2] + "!!"
        with caplog.at_level("WARNING"):
            is_rejected = translator._validate_translation(long_text, almost_same)
        assert is_rejected is True
        assert "DOCX translation validation failed (edit_distance)" in caplog.text



class TestDomainPromptGeneration:
    """Test domain-aware prompt building in DOCX paragraph translator."""

    def test_build_prompt_with_legal_domain(self):
        from src.worker.doctranslator.format.docx.paragraph_translator import (
            _build_prompt,
        )

        units = [_unit(1, "The parties agree to the terms herein.")]
        prompt = _build_prompt(units, "Spanish", domain="legal")

        assert "Legal & Regulatory" in prompt
        assert "## Domain-Specific Guidance (Legal & Regulatory Domain)" in prompt
        assert "Strictly formal, binding, and legally rigorous." in prompt
        assert "force majeure" in prompt
        assert "The parties agree to the terms herein." in prompt

    def test_build_prompt_with_commercial_domain(self):
        from src.worker.doctranslator.format.docx.paragraph_translator import (
            _build_prompt,
        )

        units = [_unit(1, "Boost your productivity with our modern cloud solution.")]
        prompt = _build_prompt(units, "French", domain="commercial")

        assert "Commercial & Business" in prompt
        assert "## Domain-Specific Guidance (Commercial & Business Domain)" in prompt
        assert "Engaging, persuasive, confident" in prompt

    def test_paragraph_translator_inherits_domain_from_engine_or_explicit(self):
        engine = _FakeEngine("[]")
        engine.domain = "finance"
        translator = DocxParagraphTranslator(engine, "German")
        assert translator.domain == "finance"

        explicit_translator = DocxParagraphTranslator(engine, "German", domain="hr")
        assert explicit_translator.domain == "hr"

