"""Factory routing and model-chain semantics tests."""

from unittest.mock import patch

import pytest
from src.doctranslator.translator.factory import create_translator
from src.doctranslator.translator.provider_types import LLMProvider
from src.doctranslator.translator.providers import ClaudeVertexAITranslator
from src.doctranslator.translator.providers import GeminiVertexAITranslator
from src.doctranslator.translator.resolver import infer_provider


class TestProviderResolver:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("gemini-2.5-flash", LLMProvider.GEMINI_VERTEXAI),
            ("gemini_vertexai-2.5-pro", LLMProvider.GEMINI_VERTEXAI),
            ("claude-sonnet-4-6", LLMProvider.CLAUDE),
            ("claude-3-5-sonnet", LLMProvider.CLAUDE),
        ],
    )
    def test_infer_provider(self, model_id, expected):
        assert infer_provider(model_id) == expected

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            infer_provider("gpt-4o")


class TestTranslatorFactory:
    @patch("src.doctranslator.translator.providers.gemini.genai.Client")
    def test_gemini_uses_exact_model_from_chain(self, mock_client):
        translator = create_translator(
            "gemini-2.5-pro",
            lang_in="en",
            lang_out="fr",
            qps=10,
        )
        assert isinstance(translator, GeminiVertexAITranslator)
        assert translator.model == "gemini-2.5-pro"
        mock_client.assert_called_once()

    @patch("anthropic.AnthropicVertex")
    def test_claude_uses_exact_model_from_chain(self, mock_vertex):
        translator = create_translator(
            "claude-sonnet-4-6",
            lang_in="en",
            lang_out="de",
            qps=8,
        )
        assert isinstance(translator, ClaudeVertexAITranslator)
        assert translator.model == "claude-sonnet-4-6"
        mock_vertex.assert_called_once()

    @patch("src.doctranslator.translator.providers.gemini.genai.Client")
    def test_selected_model_not_replaced_by_settings_default(self, mock_client):
        chain_model = "gemini-2.5-flash-lite"
        translator = create_translator(
            chain_model,
            lang_in="en",
            lang_out="es",
            qps=5,
        )
        assert translator.model == chain_model
        assert translator.model != "settings-placeholder"
