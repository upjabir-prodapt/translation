"""Factory helpers for selecting translation engines by model."""

from collections.abc import Sequence

from src.config.constants import settings
from src.config.translation_routing import ModelRoute
from src.worker.doctranslator.translator.base import BaseTranslator
from src.worker.doctranslator.translator.provider_types import LLMProvider
from src.worker.doctranslator.translator.providers import ClaudeVertexAITranslator
from src.worker.doctranslator.translator.providers import GeminiVertexAITranslator
from src.worker.doctranslator.translator.rate_limiter import set_translate_rate_limiter
from src.worker.doctranslator.translator.resolver import infer_provider


def create_translator(
    model_name: str,
    *,
    lang_in: str,
    lang_out: str,
    qps: int,
    region: str | None = None,
    domain: str | None = None,
    secondary_languages: Sequence[tuple[str, float]] | None = None,
) -> BaseTranslator:
    """Create one translator for the given model_id string.

    `region`, when provided, pins the underlying Vertex AI client to that
    region instead of the process-wide default (settings.GOOGLE_CLOUD_LOCATION)
    -- used by gemini-3.5-flash's europe-west3 pinning (docs/plan.md Section 3.3).
    Ignored for Claude, which already has its own independent
    CLAUDE_VERTEX_REGION setting.
    """
    provider = infer_provider(model_name)
    resolved_model = model_name.strip()
    # Scope the QPS limiter per-provider (not a single global limiter) so
    # concurrent translations don't reset each other's shared rate budget.
    set_translate_rate_limiter(max(int(qps), 1), provider=str(provider))

    if provider == LLMProvider.GEMINI_VERTEXAI:
        return GeminiVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=resolved_model,
            temperature=settings.LLM_TEMPERATURE,
            region=region,
            domain=domain,
            secondary_languages=secondary_languages,
        )

    if provider == LLMProvider.CLAUDE:
        return ClaudeVertexAITranslator(
            lang_in=lang_in,
            lang_out=lang_out,
            model=resolved_model,
            temperature=settings.LLM_TEMPERATURE,
            domain=domain,
            secondary_languages=secondary_languages,
        )

    raise ValueError(f"No translator implementation for provider '{provider}'.")


def create_translator_from_model_list(
    model_list: list[ModelRoute] | list[str],
    *,
    lang_in: str,
    lang_out: str,
    qps: int,
    model_index: int = 0,
    domain: str | None = None,
    secondary_languages: Sequence[tuple[str, float]] | None = None,
) -> BaseTranslator:
    """Create one translator from an ordered model list.

    Accepts either `list[ModelRoute]` (the current format produced by
    `select_model_list()`) or a plain `list[str]` of model IDs, for
    backward compatibility with any caller that hasn't been migrated yet.
    """
    if not model_list:
        raise ValueError("model_list is required to create translator")
    if model_index < 0 or model_index >= len(model_list):
        raise ValueError(
            f"model_index={model_index} out of range for model_list size {len(model_list)}"
        )

    entry = model_list[model_index]
    if isinstance(entry, ModelRoute):
        model_id, region = entry.model_id, entry.region
    else:
        model_id, region = str(entry), None

    set_translate_rate_limiter(max(qps, 1))
    return create_translator(
        model_id,
        lang_in=lang_in,
        lang_out=lang_out,
        qps=qps,
        region=region,
        domain=domain,
        secondary_languages=secondary_languages,
    )
