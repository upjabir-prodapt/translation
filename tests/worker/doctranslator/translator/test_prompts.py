"""Tests for translation prompt building with domain-specific guidance."""

from unittest.mock import patch

from src.worker.doctranslator.translator.prompts import build_translation_prompt
from src.worker.doctranslator.translator.providers.claude import (
    ClaudeVertexAITranslator,
)
from src.worker.doctranslator.translator.providers.gemini import (
    GeminiVertexAITranslator,
)


class TestBuildTranslationPrompt:
    def test_prompt_without_domain(self):
        prompt = build_translation_prompt("Hello world", "en", "es")
        assert "# Role" in prompt
        assert "# Task" in prompt
        assert "Translate the INPUT below from en into es." in prompt
        assert "## Domain-Specific Guidance" not in prompt
        assert "# INPUT\nHello world" in prompt

    def test_prompt_with_legal_domain(self):
        prompt = build_translation_prompt("Contract clause", "en", "de", domain="legal")
        assert "# Task" in prompt
        assert "## Domain-Specific Guidance (Legal & Regulatory Domain)" in prompt
        assert "Strictly formal, binding, and legally rigorous." in prompt
        assert "force majeure" in prompt
        assert "# INPUT\nContract clause" in prompt

    def test_prompt_with_commercial_domain(self):
        prompt = build_translation_prompt("Sales pitch", "en", "fr", domain="commercial")
        assert "## Domain-Specific Guidance (Commercial & Business Domain)" in prompt
        assert "Engaging, persuasive, confident" in prompt

    def test_prompt_with_finance_domain(self):
        prompt = build_translation_prompt("Financial statement", "en", "it", domain="finance")
        assert "## Domain-Specific Guidance (Finance & Accounting Domain)" in prompt
        assert "EBITDA" in prompt

    def test_prompt_with_hr_domain(self):
        prompt = build_translation_prompt("Employee handbook", "en", "ja", domain="hr")
        assert "## Domain-Specific Guidance (Human Resources & People Domain)" in prompt
        assert "Empathetic, clear, constructive" in prompt

    def test_prompt_with_operations_domain(self):
        prompt = build_translation_prompt("SOP runbook", "en", "es", domain="operations")
        assert "## Domain-Specific Guidance (Operations & Technical Domain)" in prompt
        assert "Standard Operating Procedures (SOPs)" in prompt


class TestProviderPrompts:
    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
    def test_gemini_prompt_uses_domain(self, mock_client):
        translator = GeminiVertexAITranslator(
            lang_in="en",
            lang_out="de",
            model="gemini-3.5-flash",
            domain="legal",
        )
        prompt_text = translator.prompt("Confidentiality clause")
        assert "## Domain-Specific Guidance (Legal & Regulatory Domain)" in prompt_text
        assert "Confidentiality clause" in prompt_text

    @patch("anthropic.AnthropicVertex")
    def test_claude_prompt_uses_domain(self, mock_vertex):
        translator = ClaudeVertexAITranslator(
            lang_in="en",
            lang_out="fr",
            model="claude-sonnet-4-6",
            domain="finance",
        )
        prompt_text = translator.prompt("Balance sheet items")
        assert "## Domain-Specific Guidance (Finance & Accounting Domain)" in prompt_text
        assert "Balance sheet items" in prompt_text
