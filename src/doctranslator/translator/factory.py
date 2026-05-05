"""Factory helpers for selecting translation engines by model."""

from src.doctranslator.translator.translator import BaseTranslator
from src.doctranslator.translator.translator import GeminiVertexAITranslator
from src.doctranslator.translator.translator import set_translate_rate_limiter
from src.config.constants import settings


def _infer_provider(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized.startswith("gemini"):
        return "gemini_vertexai"
    if normalized.startswith("gpt") or "openai" in normalized:
        return "openai"
    raise ValueError(f"Unsupported model '{model_name}'")


def create_translator(
    model_name: str,
    *,
    lang_in: str,
    lang_out: str,
    qps: int,  # noqa: ARG001 (reserved for rate limiter; not all translators use it)
) -> BaseTranslator:
    """Create one translator for the given model name."""
    set_translate_rate_limiter(max(int(qps), 1))
    provider = _infer_provider(model_name)

    if provider == "gemini_vertexai":
        return GeminiVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=settings.GEMINI_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )
    
    return GeminiVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=settings.GEMINI_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )




def create_translator_from_model_list(
    model_list: list[str],
    *,
    lang_in: str,
    lang_out: str,
    qps: int,
    model_index: int = 0,
) -> BaseTranslator:
    """Create one translator from ordered model list."""
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
