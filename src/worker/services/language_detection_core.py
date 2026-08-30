"""Shared language-detection primitives used by both the PDF pipeline
(`JobProcessor` in `processor_service.py`) and the DOCX/TXT pipeline
(`LanguageDetectionService`).

Extracted from `processor_service.py` (implementation_plan.md Phase C.1)
so DOCX/TXT detection can share the exact same per-unit confidence-floor
algorithm that PDF already used, instead of DOCX/TXT falling back to a
single whole-document `detect_langs()` call with no confidence floor at
all (the root cause of the EC-09 defect: short/ambiguous text like
"Information" or "OK" got a confident-looking but unreliable guess
accepted outright).
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache

from langdetect import DetectorFactory
from langdetect import LangDetectException
from langdetect import detect_langs

from src.config.translation_routing import get_language_mapper

logger = logging.getLogger(__name__)

# langdetect is non-deterministic by default (uses a random seed
# internally); pin it so repeated detection of the same text is stable.
# `processor_service.py` also sets this -- both assignments are
# idempotent (module-level, same value), so importing either module
# first has no effect on the other.
DetectorFactory.seed = 0

MIN_DETECTION_TEXT_LENGTH = 20
MIN_DETECTION_ALPHA_CHARS = 5
MIN_DETECTION_CONFIDENCE = 0.80

# langdetect emits ISO codes that don't always match this service's
# canonical set (language_mapper.json); normalize the handful of aliases
# actually seen in practice.
DETECTED_LANGUAGE_ALIASES: dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "iw": "he",
}


def is_detectable_text(text: str) -> bool:
    """Return True if `text` has enough signal for langdetect to be trusted."""
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return (
        len(text) >= MIN_DETECTION_TEXT_LENGTH
        and alpha_count >= MIN_DETECTION_ALPHA_CHARS
    )


def normalize_detected_language(language: str) -> str:
    """Map a raw langdetect code to this service's canonical alias, if any."""
    normalized = str(language).strip().lower()
    return DETECTED_LANGUAGE_ALIASES.get(normalized, normalized)


def detect_language_for_text(text: str) -> str | None:
    """Detect the language of one chunk of text, or None if not confident.

    Returns `None` (rather than a low-confidence guess) when langdetect
    itself fails, returns nothing, or reports a confidence below
    `MIN_DETECTION_CONFIDENCE` -- the confidence floor that DOCX/TXT
    detection previously lacked entirely (EC-09).
    """
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return None
    if not candidates:
        return None
    best_match = candidates[0]
    if best_match.prob < MIN_DETECTION_CONFIDENCE:
        return None
    return normalize_detected_language(best_match.lang)


def aggregate_languages(language_counter: Counter[str]) -> str | None:
    """Return the dominant language from a char-weighted Counter, or None if empty."""
    if not language_counter:
        return None
    detected_language, _ = language_counter.most_common(1)[0]
    return detected_language


@lru_cache(maxsize=1)
def get_supported_languages() -> frozenset[str]:
    """Canonical language codes this service actually supports for translation.

    Derived from `language_mapper.json` (single source of truth, per
    implementation_plan.md Phase C.1.2) rather than a hardcoded literal
    tuple, so adding a language to the mapper automatically updates the
    "skip unsupported language" behaviour without a second edit.
    """
    return frozenset(get_language_mapper().values())
