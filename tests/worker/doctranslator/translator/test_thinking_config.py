"""Thinking-budget wiring and reasoning-token accounting.

Gemini defaults to *dynamic* thinking. In the 2026-08-24 latency baseline
that produced calls returning 13 response tokens after 123s of hidden
reasoning, and `TokenUsage` silently dropped `thoughts_token_count` so the
cost of that reasoning was never reported.
"""

from unittest.mock import patch

from src.worker.doctranslator.translator.providers.gemini import (
    GeminiVertexAITranslator,
)
from src.worker.doctranslator.translator.usage import TokenUsage


class _Usage:
    def __init__(self, prompt=10, candidates=5, cached=0, thoughts=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.cached_content_token_count = cached
        self.thoughts_token_count = thoughts


class TestThinkingTokenAccounting:
    def test_reads_thoughts_token_count(self):
        usage = TokenUsage.from_gemini_usage(_Usage(thoughts=200))
        assert usage.thinking_tokens == 200

    def test_total_tokens_semantics_unchanged(self):
        """total_tokens must stay input+output so existing cost maths holds."""
        usage = TokenUsage.from_gemini_usage(
            _Usage(prompt=100, candidates=50, thoughts=200)
        )
        assert usage.total_tokens == 150

    def test_billable_output_includes_thinking(self):
        """Vertex bills reasoning as output tokens."""
        usage = TokenUsage.from_gemini_usage(
            _Usage(prompt=100, candidates=50, thoughts=200)
        )
        assert usage.billable_output_tokens == 250

    def test_backwards_compatible_with_responses_lacking_the_field(self):
        class Old:
            prompt_token_count = 10
            candidates_token_count = 5

        assert TokenUsage.from_gemini_usage(Old()).thinking_tokens == 0

    def test_none_usage_is_safe(self):
        assert TokenUsage.from_gemini_usage(None).thinking_tokens == 0

    def test_merge_sums_thinking_tokens(self):
        merged = TokenUsage(thinking_tokens=3).merge(TokenUsage(thinking_tokens=4))
        assert merged.thinking_tokens == 7


def _make(model="gemini-3.5-flash"):
    """Build a translator without constructing a real Vertex client."""
    with patch("src.worker.doctranslator.translator.providers.gemini.genai.Client"):
        return GeminiVertexAITranslator(
            lang_in="en", lang_out="fr", model=model, temperature=0.0
        )


class TestThinkingConfigSelection:
    def test_flash_uses_standard_budget(self):
        translator = _make(model="gemini-3.5-flash")
        with patch.multiple(
            "src.worker.doctranslator.translator.providers.gemini.settings",
            LLM_THINKING_BUDGET=0,
            LLM_THINKING_BUDGET_PRO=512,
        ):
            config = translator._build_thinking_config()
        assert config is not None
        assert config.thinking_budget == 0

    def test_pro_uses_its_own_budget(self):
        """Pro-class models cannot fully disable thinking, so they get a floor."""
        translator = _make(model="gemini-2.5-pro")
        with patch.multiple(
            "src.worker.doctranslator.translator.providers.gemini.settings",
            LLM_THINKING_BUDGET=0,
            LLM_THINKING_BUDGET_PRO=512,
        ):
            config = translator._build_thinking_config()
        assert config is not None
        assert config.thinking_budget == 512

    def test_negative_budget_restores_sdk_default(self):
        """-1 is the documented escape hatch if quality regresses."""
        translator = _make(model="gemini-3.5-flash")
        with patch.multiple(
            "src.worker.doctranslator.translator.providers.gemini.settings",
            LLM_THINKING_BUDGET=-1,
            LLM_THINKING_BUDGET_PRO=512,
        ):
            assert translator._build_thinking_config() is None
