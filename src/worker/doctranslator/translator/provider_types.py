"""Canonical LLM provider identifiers."""

from enum import StrEnum


class LLMProvider(StrEnum):
    """Provider identifiers used in factory routing, OTel spans, and costing."""

    GEMINI_VERTEXAI = "gemini_vertexai"
    CLAUDE = "claude"
