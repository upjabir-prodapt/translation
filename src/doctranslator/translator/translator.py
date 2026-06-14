"""Backward-compatible re-exports for translator modules."""

from src.doctranslator.translator.base import BaseTranslator
from src.doctranslator.translator.base import remove_control_characters
from src.doctranslator.translator.providers import ClaudeVertexAITranslator
from src.doctranslator.translator.providers import GeminiVertexAITranslator
from src.doctranslator.translator.rate_limiter import set_translate_rate_limiter
from src.doctranslator.translator.schemas import BatchTranslationItem
from src.doctranslator.translator.schemas import BatchTranslationResponse
from src.doctranslator.translator.schemas import ExtractedTerm
from src.doctranslator.translator.schemas import TermExtractionResponse
from src.doctranslator.translator.schemas import TranslationResponse

__all__ = [
    "BaseTranslator",
    "BatchTranslationItem",
    "BatchTranslationResponse",
    "ClaudeVertexAITranslator",
    "ExtractedTerm",
    "GeminiVertexAITranslator",
    "TermExtractionResponse",
    "TranslationResponse",
    "remove_control_characters",
    "set_translate_rate_limiter",
]
