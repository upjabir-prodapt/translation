"""Factory routing and model-chain semantics tests."""

from unittest.mock import patch

import pytest
from src.config.translation_routing import ModelRoute
from src.worker.doctranslator.translator.factory import create_translator
from src.worker.doctranslator.translator.factory import (
    create_translator_from_model_list,
)
from src.worker.doctranslator.translator.provider_types import LLMProvider
from src.worker.doctranslator.translator.providers import ClaudeVertexAITranslator
from src.worker.doctranslator.translator.providers import GeminiVertexAITranslator
from src.worker.doctranslator.translator.resolver import infer_provider


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
    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
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

    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
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

    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
    def test_region_is_passed_through_to_gemini_client(self, mock_client):
        translator = create_translator(
            "gemini-3.5-flash",
            lang_in="en",
            lang_out="de",
            qps=10,
            region="europe-west3",
        )
        assert translator.region == "europe-west3"
        _, kwargs = mock_client.call_args
        assert kwargs["location"] == "europe-west3"

    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
    def test_region_defaults_to_settings_when_not_provided(self, mock_client):
        from src.config.constants import settings

        translator = create_translator(
            "gemini-2.5-flash",
            lang_in="en",
            lang_out="de",
            qps=10,
        )
        assert translator.region == (
            settings.GEMINI_MODEL_REGION or settings.GOOGLE_CLOUD_LOCATION
        )

    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
    def test_region_defaults_to_location_when_gemini_region_empty(self, mock_client):
        from src.config.constants import settings

        with patch.object(settings, "GEMINI_MODEL_REGION", ""):
            translator = create_translator(
                "gemini-2.5-flash",
                lang_in="en",
                lang_out="de",
                qps=10,
            )
            assert translator.region == settings.GOOGLE_CLOUD_LOCATION

    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
    def test_create_translator_from_model_list_with_model_route(self, mock_client):
        model_list = [
            ModelRoute(model_id="gemini-3.5-flash", region="europe-west3"),
            ModelRoute(model_id="gemini-2.5-pro", region=None),
        ]
        translator = create_translator_from_model_list(
            model_list,
            lang_in="en",
            lang_out="de",
            qps=10,
            model_index=0,
        )
        assert translator.model == "gemini-3.5-flash"
        assert translator.region == "europe-west3"

    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
    def test_create_translator_from_model_list_backward_compat_str_list(
        self, mock_client
    ):
        translator = create_translator_from_model_list(
            ["gemini-2.5-flash"],
            lang_in="en",
            lang_out="de",
            qps=10,
        )
        assert translator.model == "gemini-2.5-flash"

    @patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
    def test_domain_is_forwarded_to_gemini_translator(self, mock_client):
        translator = create_translator(
            "gemini-3.5-flash",
            lang_in="en",
            lang_out="de",
            qps=10,
            domain="legal",
        )
        assert translator.domain == "legal"

    @patch("anthropic.AnthropicVertex")
    def test_domain_is_forwarded_to_claude_translator(self, mock_vertex):
        translator = create_translator(
            "claude-sonnet-4-6",
            lang_in="en",
            lang_out="es",
            qps=8,
            domain="commercial",
        )
        assert translator.domain == "commercial"
