"""Enum of supported LLM translation providers."""

from enum import StrEnum


class LLMProvider(StrEnum):
    """Canonical provider identifiers used in factory routing, OTel spans, and settings."""

    GEMINI_VERTEXAI = "gemini_vertexai"
    CLAUDE = "claude"
    QWEN_VERTEXAI = "qwen_vertexai"
