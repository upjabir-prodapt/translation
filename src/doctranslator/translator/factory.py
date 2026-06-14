"""Factory helpers for selecting translation engines by model."""

from src.config.constants import settings
from src.doctranslator.translator.base import BaseTranslator
from src.doctranslator.translator.provider_types import LLMProvider
from src.doctranslator.translator.providers import ClaudeVertexAITranslator
from src.doctranslator.translator.providers import GeminiVertexAITranslator
from src.doctranslator.translator.rate_limiter import set_translate_rate_limiter
from src.doctranslator.translator.resolver import infer_provider


def create_translator(
    model_name: str,
    *,
    lang_in: str,
    lang_out: str,
    qps: int,
) -> BaseTranslator:
    """Create one translator for the given model_id string."""
    set_translate_rate_limiter(max(int(qps), 1))
    provider = infer_provider(model_name)
    resolved_model = model_name.strip()

    if provider == LLMProvider.GEMINI_VERTEXAI:
        return GeminiVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=resolved_model,
            temperature=settings.LLM_TEMPERATURE,
        )

    if provider == LLMProvider.CLAUDE:
        return ClaudeVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=resolved_model,
            temperature=settings.LLM_TEMPERATURE,
        )

    raise ValueError(f"No translator implementation for provider '{provider}'.")


def create_translator_from_model_list(
    model_list: list[str],
    *,
    lang_in: str,
    lang_out: str,
    qps: int,
    model_index: int = 0,
) -> BaseTranslator:
    """Create one translator from an ordered model list."""
    if not model_list:
        raise ValueError("model_list is required to create translator")
    if model_index < 0 or model_index >= len(model_list):
        raise ValueError(
            f"model_index={model_index} out of range for model_list size {len(model_list)}"
        )

    set_translate_rate_limiter(max(qps, 1))
    return create_translator(
        model_list[model_index],
        lang_in=lang_in,
        lang_out=lang_out,
        qps=qps,
    )
