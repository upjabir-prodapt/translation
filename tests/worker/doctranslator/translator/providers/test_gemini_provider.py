"""Gemini provider extract_text() and region-pinning tests."""

from types import SimpleNamespace
from unittest.mock import patch

from src.worker.doctranslator.translator.providers import GeminiVertexAITranslator
from src.worker.doctranslator.translator.schemas import BatchTranslationItem
from src.worker.doctranslator.translator.schemas import BatchTranslationResponse
from src.worker.doctranslator.translator.schemas import ExtractedTerm
from src.worker.doctranslator.translator.schemas import TermExtractionResponse
from src.worker.doctranslator.translator.schemas import TranslationResponse


@patch("src.worker.doctranslator.translator.providers.gemini.genai.Client")
def _make_translator(_mock_client, region=None):
    return GeminiVertexAITranslator(
        lang_in="en",
        lang_out="de",
        model="gemini-3.5-flash",
        region=region,
    )


class TestGeminiExtractText:
    def test_translation_response_uses_parsed(self):
        translator = _make_translator()
        response = SimpleNamespace(
            parsed=TranslationResponse(translated_text="hallo"),
            text="",
        )
        assert translator.extract_text(response) == "hallo"

    def test_batch_translation_response_uses_parsed(self):
        translator = _make_translator()
        parsed = BatchTranslationResponse(
            items=[BatchTranslationItem(id=1, output="hallo")]
        )
        response = SimpleNamespace(parsed=parsed, text="")
        out = translator.extract_text(response)
        assert '"id":1' in out.replace(" ", "")
        assert "hallo" in out

    def test_term_extraction_response_uses_parsed(self):
        translator = _make_translator()
        parsed = TermExtractionResponse(
            terms=[ExtractedTerm(src="cat", tgt="Katze", src_lang="en")]
        )
        response = SimpleNamespace(parsed=parsed, text="")
        out = translator.extract_text(response)
        assert "Katze" in out

    def test_falls_back_to_raw_text_when_unparsed(self):
        translator = _make_translator()
        response = SimpleNamespace(parsed=None, text="plain text output")
        assert translator.extract_text(response) == "plain text output"


class TestGeminiRegionPinning:
    def test_region_override_used_when_provided(self):
        translator = _make_translator(region="europe-west3")
        assert translator.region == "europe-west3"

    def test_gemini_model_region_used_when_configured(self):
        from src.config.constants import settings

        with patch.object(settings, "GEMINI_MODEL_REGION", "europe-west3"):
            translator = _make_translator(region=None)
            assert translator.region == "europe-west3"

    def test_default_region_used_when_not_provided(self):
        from src.config.constants import settings

        with patch.object(settings, "GEMINI_MODEL_REGION", ""):
            translator = _make_translator(region=None)
            assert translator.region == settings.GOOGLE_CLOUD_LOCATION
