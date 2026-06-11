"""Factory helpers for selecting translation engines by model."""

from src.config.constants import settings
from src.doctranslator.translator.providers import LLMProvider
from src.doctranslator.translator.translator import BaseTranslator
from src.doctranslator.translator.translator import ClaudeVertexAITranslator
from src.doctranslator.translator.translator import GeminiVertexAITranslator
from src.doctranslator.translator.translator import set_translate_rate_limiter


def _infer_provider(model_name: str) -> LLMProvider:
    """Infer the LLMProvider from a model_id string."""
    normalized = model_name.strip().lower()
    if normalized.startswith(LLMProvider.GEMINI_VERTEXAI) or normalized.startswith(
        "gemini"
    ):
        return LLMProvider.GEMINI_VERTEXAI
    if normalized.startswith(LLMProvider.CLAUDE) or normalized.startswith("claude"):
        return LLMProvider.CLAUDE
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported prefixes: 'gemini', 'claude'."
    )


def create_translator(
    model_name: str,
    *,
    lang_in: str,
    lang_out: str,
    qps: int,
) -> BaseTranslator:
    """Create one translator for the given model_id string."""
    set_translate_rate_limiter(max(int(qps), 1))
    provider = _infer_provider(model_name)

    if provider == LLMProvider.GEMINI_VERTEXAI:
        return GeminiVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=settings.GEMINI_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

    if provider == LLMProvider.CLAUDE:
        return ClaudeVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=settings.CLAUDE_MODEL,
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
