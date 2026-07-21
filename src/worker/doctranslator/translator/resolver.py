"""Provider resolution from model ID strings."""

from __future__ import annotations

from src.worker.doctranslator.translator.provider_types import LLMProvider


def infer_provider(model_name: str) -> LLMProvider:
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
