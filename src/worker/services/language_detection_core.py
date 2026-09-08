"""Shared language-detection primitives used by both the PDF pipeline
(`JobProcessor` in `processor_service.py`) and the DOCX/TXT pipeline
(`LanguageDetectionService`).

Extracted from `processor_service.py` (implementation_plan.md Phase C.1)
so DOCX/TXT detection can share the exact same per-unit confidence-floor
algorithm that PDF already used, instead of DOCX/TXT falling back to a
single whole-document detection call with no confidence floor at all
(the root cause of the EC-09 defect: short/ambiguous text like
"Information" or "OK" got a confident-looking but unreliable guess
accepted outright).

Backed by `lingua-language-detector` (previously `langdetect`): lingua's
`with_minimum_relative_distance()` floor rejects ambiguous short/list-like
text (bare place-name lists, ALL-CAPS headers) far more reliably than
langdetect's absolute-probability floor did, which used to false-positive
"mixed language" on entirely monolingual marketing documents.
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache

from lingua import Language
from lingua import LanguageDetector
from lingua import LanguageDetectorBuilder

from src.config.constants import settings
from src.config.translation_routing import get_language_mapper
from src.worker.doctranslator.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)

# Bounds how many *unexpected* detector errors get a WARNING per process --
# see detect_language_for_text(). Detection runs concurrently across a
# ThreadPoolExecutor (DOCX term extraction/translation batches), hence the
# thread-safe counter rather than a plain module-level int.
_DETECTOR_ERROR_LOG_LIMIT = 5
_detector_error_log_count = AtomicInteger(0)


@lru_cache(maxsize=1)
def _build_detector(min_relative_distance: float) -> LanguageDetector:
    """Build the process-wide lingua detector for one confidence threshold.

    This module already requires a full worker `.env` to import: it pulls
    in `get_language_mapper()` below, which imports `settings` at module
    scope. There is nothing to protect by deferring the `settings` import
    used here, so it is a plain top-level import like the rest of this
    file. (`scripts/lang_detect_benchmark.py`, which does need to run
    without a worker `.env`, does not import this module at all -- it
    reimplements the pieces it compares against inline for exactly this
    reason; see its own docstring.)

    `from_all_languages()` is deliberate. Restricting the detector to the
    seven translatable languages would be smaller and more accurate, but
    it would also make it impossible to *recognise* that a document is
    Russian or Arabic -- which is exactly what the pipeline's
    supported-language coverage gate and the per-unit
    SKIP_UNSUPPORTED_LANGUAGE_UNITS check both need to know.

    `with_preloaded_language_models()` is intentionally NOT used. In
    lingua-language-detector 2.x (the Rust binding) it is a no-op:
    measured on this codebase, `build()` returns in <10ms and leaves RSS
    at ~12 MB with or without it, and the models are memory-mapped
    lazily on first use regardless. Steady-state RSS after touching every
    script is ~210 MB against the worker's 16Gi limit, so there is
    nothing to preload away from. Calling it would only imply a guarantee
    the library does not make.

    Keyed on `min_relative_distance` itself (not called with no arguments)
    so the cache self-invalidates: with `maxsize=1`, a settings change is
    a different argument value, which evicts the stale detector and builds
    a fresh one on the very next call -- no `.cache_clear()` needed, in
    production or in tests.
    """
    return (
        LanguageDetectorBuilder.from_all_languages()
        .with_minimum_relative_distance(min_relative_distance)
        .build()
    )


def _get_detector() -> LanguageDetector:
    """Return the process-wide lingua detector for the current setting.

    Thin wrapper so call sites and test patches (`patch.object(core,
    "_get_detector", ...)`) are unaffected by `_build_detector()` now being
    memoized on its argument rather than on no arguments at all.
    """
    return _build_detector(float(settings.LANGUAGE_DETECTION_MIN_RELATIVE_DISTANCE))


def is_detectable_text(text: str) -> bool:
    """Return True if `text` has enough signal for the detector to be trusted.

    Reads `settings.MIN_DETECTION_TEXT_LENGTH` /
    `settings.MIN_DETECTION_ALPHA_CHARS` fresh on every call rather than
    binding them at import time -- unlike `JobProcessor.MAX_DETECTION_CHARS`
    (bound at class-definition time, so a settings change after import never
    takes effect), a settings override here is picked up immediately.
    """
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return (
        len(text) >= settings.MIN_DETECTION_TEXT_LENGTH
        and alpha_count >= settings.MIN_DETECTION_ALPHA_CHARS
    )


def normalize_detected_language(language: str) -> str:
    """Map a raw detector code to its `language_mapper.json` canonical
    code, if known; otherwise pass it through unchanged.

    Previously used a small hardcoded alias dict (`zh-cn`/`zh-tw` -> `zh`,
    `iw` -> `he`) that duplicated aliases already present in
    `language_mapper.json` (`zh-cn`/`zh-tw` both map to `zh` there too) --
    a second, independent source of truth with nothing keeping the two in
    sync. Looks up `get_language_mapper()` directly instead.

    Unlike `normalize_language()` (which raises on a miss -- appropriate
    for a routing input the API already validated), this must never raise:
    a detected code legitimately falls outside the mapper constantly
    (Russian, Arabic, ...) and has to pass through unchanged so it still
    appears in the language distribution as "detected but unsupported"
    rather than being dropped or mis-mapped.
    """
    normalized = str(language).strip().lower()
    return get_language_mapper().get(normalized, normalized)


def detect_language_for_text(text: str) -> str | None:
    """Detect the language of one chunk of text, or None if not confident.

    Returns `None` (rather than a low-confidence guess) in two distinct
    cases, logged at two distinct levels:

    - lingua **declines** to name a language because the top two candidates
      are within `LANGUAGE_DETECTION_MIN_RELATIVE_DISTANCE` of each other --
      the confidence floor that DOCX/TXT detection previously lacked
      entirely (EC-09). Expected and frequent on short/ambiguous text, so
      this stays at debug.
    - lingua **raises unexpectedly** (a real detector-side fault, not a
      confidence judgement). This is logged at WARNING -- capped at
      `_DETECTOR_ERROR_LOG_LIMIT` per process so a systemic failure (e.g. a
      threading issue under concurrent DOCX extraction) is visible in
      production logs instead of only ever appearing at debug, which is
      exactly the class of failure that used to surface as a confusing
      "cannot translate" coverage-gate rejection with no signal pointing at
      the detector itself.
    """
    try:
        language: Language | None = _get_detector().detect_language_of(text)
    except Exception:
        seen = _detector_error_log_count.inc()
        if seen <= _DETECTOR_ERROR_LOG_LIMIT:
            logger.warning(
                "lingua detect_language_of raised unexpectedly (%d/%d logged "
                "this process) -- treating this block as undetectable rather "
                "than failing it outright.",
                seen,
                _DETECTOR_ERROR_LOG_LIMIT,
                exc_info=True,
            )
        else:
            logger.debug("lingua detect_language_of raised", exc_info=True)
        return None
    if language is None:
        return None
    return normalize_detected_language(language.iso_code_639_1.name)


def aggregate_languages(language_counter: Counter[str]) -> str | None:
    """Return the dominant language from a char-weighted Counter, or None if empty."""
    if not language_counter:
        return None
    detected_language, _ = language_counter.most_common(1)[0]
    return detected_language


def language_shares(language_counter: Counter[str]) -> dict[str, float]:
    """Convert a char-weighted Counter into each language's fraction of the total.

    Returns an empty dict for an empty (or non-positive) counter rather
    than raising, so callers can log a distribution without first having
    to special-case the no-evidence document.
    """
    total = sum(language_counter.values())
    if total <= 0:
        return {}
    return {code: count / total for code, count in language_counter.items()}


def significant_languages(language_counter: Counter[str]) -> Counter[str]:
    """Drop long-tail detection artefacts from a char-weighted distribution.

    Detection runs per text block, and across a long document a handful of
    blocks will always land on some unrelated language -- the HR Policy
    Manual fixture produces 30 distinct "languages", 28 of which are a
    single block each. Judging supported-language coverage on the raw
    distribution would let one such block drag an entirely English
    document below the threshold, so anything holding less than
    `LANGUAGE_DETECTION_NOISE_SHARE` of the detected characters is
    discarded first.

    If *every* language falls below the floor (a genuinely fragmented
    document, or one so short that no single language reaches the share),
    the full distribution is returned unchanged. Returning an empty
    Counter there would silently turn "many languages, none dominant"
    into "no evidence at all", which is a different verdict with a
    different user-facing message.
    """
    total = sum(language_counter.values())
    if total <= 0:
        return Counter()

    noise_share = float(settings.LANGUAGE_DETECTION_NOISE_SHARE)
    kept = Counter(
        {
            code: count
            for code, count in language_counter.items()
            if count / total >= noise_share
        }
    )
    return kept or Counter(language_counter)


def build_unclassifiable_text_message(
    *,
    subject: str,
    text_noun: str = "readable text",
    include_prefix: bool = True,
) -> str:
    """Shared wording for "text was present but nothing could be classified".

    Both the PDF (`processor_service.py`) and DOCX/TXT
    (`language_detection_service.py`) detection paths split "no candidate
    text at all" from "candidate text present, but every block fell below
    lingua's confidence floor" into two different user-facing messages --
    conflating them would misdescribe a text-rich but ambiguous document as
    empty. This covers only the second case: the first is deliberately
    NOT unified here, since the PDF path's wording for it must stay
    byte-identical to `PDFValidator`'s API-side "no extractable text
    layer" rejection (a different, format-specific concern), while
    DOCX/TXT's is a generic message.

    `subject`/`text_noun`/`include_prefix` let each caller keep its
    existing exact wording (so this refactor changes no user-facing text)
    while sharing the actual message *shape* in one place.
    """
    prefix = "Unable to detect a source language: " if include_prefix else ""
    return (
        f"{prefix}{subject} contains {text_noun}, but no passage was long "
        "or distinctive enough to identify its language with confidence. "
        "Please supply a document with more continuous prose."
    )


def supported_language_share(language_counter: Counter[str]) -> float:
    """Fraction of ALL detected characters we can actually translate.

    This is the single number the pipeline's coverage gate is built on. It
    deliberately says nothing about which language was *declared* -- that
    comparison (dominant vs. declared) is a separate check the caller makes
    itself against `significant_languages()`. The only question this
    answers is whether enough of the document is in a language present in
    `language_mapper.json`; the rest is passed through untranslated by the
    per-unit SKIP_UNSUPPORTED_LANGUAGE_UNITS check.

    Deliberately computed on the **raw**, non-noise-filtered distribution:
    `significant_languages()` exists to stop one-block detection artefacts
    from dragging a coverage number around, but that same floor would also
    let genuinely untranslatable content evade the gate entirely if it
    happens to be split across several sub-floor blocks (e.g. ten different
    languages at 9% each `sum -- 90%` unsupported, none individually above
    `LANGUAGE_DETECTION_NOISE_SHARE`). Coverage must count every detected
    character against the budget; only the *dominant-language* decision
    (and the CJK/glossary language list) is allowed to use the filtered
    view.

    Returns 0.0 for an empty distribution. Callers must distinguish "no
    evidence" from "evidence, all unsupported" *before* reading this
    value -- both come back as 0.0 and they are not the same failure.
    """
    total = sum(language_counter.values())
    if total <= 0:
        return 0.0
    supported = get_supported_languages()
    return (
        sum(count for code, count in language_counter.items() if code in supported)
        / total
    )


@lru_cache(maxsize=1)
def get_supported_languages() -> frozenset[str]:
    """Canonical language codes this service actually supports for translation.

    Derived from `language_mapper.json` (single source of truth, per
    implementation_plan.md Phase C.1.2) rather than a hardcoded literal
    tuple, so adding a language to the mapper automatically updates the
    "skip unsupported language" behaviour without a second edit.
    """
    return frozenset(get_language_mapper().values())
