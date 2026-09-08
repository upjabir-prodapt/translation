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

Detector
--------
Detection is backed by `lingua` (see `_get_detector`), which replaced
`langdetect`. The two differ in ways that matter to everything below:

* `lingua` normalizes its confidence values across all the languages it
  was built with, so `MIN_DETECTION_CONFIDENCE` is a real threshold
  rather than a formality. `langdetect` returned ~0.99999 for almost any
  input, including a three-word run of city names, which is why the
  floor could not filter noise on its own and the shape-based rules
  below had to exist.
* `lingua` lowercases internally, so casing no longer changes a verdict
  (`case_normalize_for_detection` is now a no-op in effect, and is kept
  only so the pipeline is not sensitive to that detail of the backend).
* `lingua` is deterministic; `langdetect` needed `DetectorFactory.seed`
  pinned to stop the same text resolving differently between runs.

Mixed-language decision
-----------------------
The original guard rejected a document when the *count* of distinct
languages on a single page exceeded a fixed limit. That was unusable in
practice: a wholly monolingual English brochure (all-caps headings plus
a network map full of city names) registered four "languages" on one
page. A 26-character run of proper nouns counted exactly as much toward
the limit as 2,400 characters of real prose.

Detection is now two layers, both defined here so every format shares
them:

1. Per-unit signal quality (`is_high_signal_unit`,
   `case_normalize_for_detection`) discards text that carries no
   reliable language signal -- too short, too few words, or a list of
   proper nouns -- so noise never enters the distribution in the first
   place. `lingua` scores that noise low enough for the confidence floor
   to catch most of it ("Richmond Richmond Richmond" -> en 0.14,
   "Busan Tokyo Osaka Yokohama" -> ms 0.22), so this layer is now
   defence in depth rather than the only thing standing between a
   monolingual brochure and a mixed-language rejection.

2. A char-weighted *share* decision (`resolve_dominant_language`)
   replaces the distinct-language count. Languages holding a negligible
   share are treated as incidental and excluded; a document is rejected
   as genuinely mixed only when no single language dominates the text
   that remains. This is scale-invariant (a 2-page brochure and a
   2,000-page manual are judged the same way) and evaluated once per
   document rather than per page, so one map page or one page of
   addresses can no longer fail an otherwise monolingual job.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

from lingua import LanguageDetector
from lingua import LanguageDetectorBuilder

from src.config.constants import settings
from src.config.translation_routing import get_language_mapper

logger = logging.getLogger(__name__)

# Relaxed floor. Only gates which text is *considered* at all, and is
# also the floor used by the whole-document fallback pass; counting a
# unit toward the language distribution requires `is_high_signal_unit`.
MIN_DETECTION_TEXT_LENGTH = 20
MIN_DETECTION_ALPHA_CHARS = 5
# lingua spreads its confidence across every language it knows, so this
# is a meaningful floor: clear prose in a supported language scores
# >=0.93, while proper-noun runs and stray-word fragments score under
# 0.25 and are dropped.
MIN_DETECTION_CONFIDENCE = 0.80

# --- Layer 1: per-unit signal quality --------------------------------
# Length and shape rules, applied before the detector is consulted at
# all. Retained from the langdetect era (where they were load-bearing,
# because the confidence floor could not filter anything) as a cheap
# first pass that keeps short display text out of the distribution.
MIN_UNIT_TEXT_LENGTH = settings.LANGUAGE_DETECTION_MIN_UNIT_CHARS
MIN_UNIT_WORD_COUNT = settings.LANGUAGE_DETECTION_MIN_UNIT_WORDS

# Above this share of cased letters being uppercase, the unit is
# lowercased before detection. lingua is case-insensitive, so this no
# longer changes any verdict; it is kept so the pipeline does not depend
# on that property of the backend.
UPPERCASE_NORMALIZATION_RATIO = 0.6

# Above this share of words being capitalized, the unit is treated as a
# list of proper nouns (city labels, person names, job titles) which
# carries no language signal. Deliberately high: languages that
# capitalize nouns in ordinary prose (German) sit well under it.
MAX_CAPITALIZED_WORD_RATIO = 0.7

# --- Layer 2: mixed-language decision --------------------------------
# A language holding less than this share of detected characters is
# incidental -- a quoted phrase, an address, a brand name, or residual
# detector noise -- and is excluded from the dominance denominator so it
# cannot drag an otherwise monolingual document below the bar.
NOISE_LANGUAGE_SHARE = settings.LANGUAGE_DETECTION_NOISE_SHARE

# The dominant language must hold at least this share of the
# non-incidental characters, else the document is genuinely mixed and is
# rejected rather than translated from a single wrong source language.
MIN_DOMINANT_LANGUAGE_SHARE = settings.LANGUAGE_DETECTION_MIN_DOMINANT_SHARE

# Below this many detected characters there is not enough evidence to
# call a document "mixed"; accept the dominant language instead of
# rejecting a short document on the strength of one or two units.
MIN_MIXED_DECISION_CHARS = settings.LANGUAGE_DETECTION_MIN_MIXED_DECISION_CHARS

# Minimum concatenated length before the whole-document fallback pass
# will trust a single verdict. Deliberately *not* configurable: it is the
# EC-09 safety floor that stops "Information Total OK 2026" resolving to
# a confident-looking wrong language, not a tuning knob.
MIN_FALLBACK_TEXT_CHARS = 500

# lingua emits clean ISO 639-1 codes (`zh`, `he`), so these aliases are
# inert for the current backend. Kept because the same normalization is
# applied to language codes arriving from elsewhere -- language_mapper.json
# still carries a `zh-cn` key -- and because dropping it would silently
# change what a caller passing a legacy code resolves to.
DETECTED_LANGUAGE_ALIASES: dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "iw": "he",
}


@lru_cache(maxsize=1)
def _get_detector() -> LanguageDetector:
    """Return the process-wide lingua detector, building it once.

    Built from *all* of lingua's languages rather than only the ones in
    `get_supported_languages()`: callers rely on detection being able to
    name an unsupported language so they can skip that text
    (`paragraph_translator.py`, `il_translator_llm_only.py`,
    `pipeline_orchestrator.py`). Restricting the detector would force
    every Dutch or Portuguese block into the nearest supported language
    instead.

    Language models load lazily on first use and are then cached inside
    the detector, which is why this is a singleton -- one shared
    detector across concurrent jobs rather than one per `JobProcessor`.
    `LanguageDetector` is immutable and safe to share between threads.
    """
    return LanguageDetectorBuilder.from_all_languages().build()


class MixedLanguageError(ValueError):
    """Raised when no single language dominates the document's text.

    Subclasses `ValueError` so existing callers and the API exception
    handler keep treating it as a validation failure, while
    `TranslationAttemptRunner` can classify it as non-retryable -- no
    amount of re-running the model chain makes a genuinely
    mixed-language document translatable from one source language.
    """

    def __init__(self, message: str, language_shares: dict[str, float] | None = None):
        super().__init__(message)
        self.language_shares = language_shares or {}


def is_detectable_text(text: str) -> bool:
    """Return True if `text` has enough signal for detection to be trusted."""
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return (
        len(text) >= MIN_DETECTION_TEXT_LENGTH
        and alpha_count >= MIN_DETECTION_ALPHA_CHARS
    )


def case_normalize_for_detection(text: str) -> str:
    """Lowercase `text` when it is predominantly uppercase.

    A no-op as far as the verdict goes under lingua, which lowercases
    input itself: "SUPPORTING YOUR BUSINESS WITH OUR BACKBONE NETWORK
    TODAY" and its lowercase form both score en 0.899591. It mattered
    under langdetect, whose profiles were lowercase n-grams, so all-caps
    English headings -- ubiquitous in brochures, slide decks and
    marketing PDFs -- matched foreign profiles at near-total confidence.
    Kept so the pipeline stays correct if the backend changes again;
    text that is already mixed-case is returned untouched.
    """
    cased = [ch for ch in text if ch.isalpha()]
    if not cased:
        return text
    uppercase_ratio = sum(1 for ch in cased if ch.isupper()) / len(cased)
    if uppercase_ratio >= UPPERCASE_NORMALIZATION_RATIO:
        return text.lower()
    return text


def is_proper_noun_list(text: str) -> bool:
    """Return True if `text` looks like a run of proper nouns.

    Page furniture such as "Busan Tokyo Osaka Yokohama Nagoya Kyoto
    Kobe" or "Executive Manager Katsuya Oe (Vice President, Enterprise
    Sales - Asia)" is long enough to clear every length floor, yet
    carries no language signal. Names are overwhelmingly capitalized,
    which is what this detects.
    """
    words = [word for word in text.split() if any(ch.isalpha() for ch in word)]
    if not words:
        return False
    capitalized = sum(1 for word in words if word[:1].isupper())
    return capitalized / len(words) >= MAX_CAPITALIZED_WORD_RATIO


def is_high_signal_unit(text: str) -> bool:
    """Return True if `text` may contribute to the language distribution.

    Stricter than `is_detectable_text`: a unit only votes on the
    document's language when it is long enough, has enough words, and is
    not a proper-noun list. Applied *after* case normalization, so
    all-caps prose is judged on its words rather than its casing.
    """
    if len(text) < MIN_UNIT_TEXT_LENGTH:
        return False
    words = [word for word in text.split() if any(ch.isalpha() for ch in word)]
    if len(words) < MIN_UNIT_WORD_COUNT:
        return False
    return not is_proper_noun_list(text)


def normalize_detected_language(language: str) -> str:
    """Map a raw detected code to this service's canonical alias, if any."""
    normalized = str(language).strip().lower()
    return DETECTED_LANGUAGE_ALIASES.get(normalized, normalized)


def detect_language_for_text(text: str) -> str | None:
    """Detect the language of one chunk of text, or None if not confident.

    Returns `None` (rather than a low-confidence guess) when the
    detector fails, returns nothing, or reports a confidence below
    `MIN_DETECTION_CONFIDENCE` -- the confidence floor that DOCX/TXT
    detection previously lacked entirely (EC-09). The text is
    case-normalized first (see `case_normalize_for_detection`).

    Note that lingua returns a full, zero-valued candidate list for text
    it cannot read at all rather than an empty one, so the floor -- not
    the emptiness check -- is what rejects those.
    """
    try:
        candidates = _get_detector().compute_language_confidence_values(
            case_normalize_for_detection(text)
        )
    except Exception:
        # Detection is advisory: a backend failure on one text block must
        # not abort a job that would otherwise translate fine, so this
        # degrades to "no language" exactly as an unconfident result does.
        logger.warning("Language detection failed for a text block", exc_info=True)
        return None
    if not candidates:
        return None
    best_match = candidates[0]
    if best_match.value < MIN_DETECTION_CONFIDENCE:
        return None
    return normalize_detected_language(best_match.language.iso_code_639_1.name)


def count_unit_languages(
    units: Iterable[str], *, max_chars: int | None = None
) -> tuple[Counter[str], int]:
    """Char-weight the languages of `units`, keeping only high-signal ones.

    Returns `(language_counter, considered_chars)`. `considered_chars`
    counts every unit examined (not just those that produced a
    language) so callers can enforce a sampling budget consistently with
    the previous behaviour.
    """
    language_counter: Counter[str] = Counter()
    considered_chars = 0
    for text in units:
        considered_chars += len(text)
        if is_high_signal_unit(text):
            detected = detect_language_for_text(text)
            if detected is not None:
                language_counter[detected] += len(text)
        if max_chars is not None and considered_chars >= max_chars:
            break
    return language_counter, considered_chars


def detect_language_of_joined_text(units: Iterable[str]) -> Counter[str]:
    """Fallback pass: detect one language across all `units` concatenated.

    Documents made entirely of short units -- slide decks, bullet lists,
    form labels -- can leave the high-signal pass empty even though the
    document as a whole is unambiguous. Concatenating gives the detector
    the long text it is most reliable on, so this recovers them without
    weakening the per-unit floors.

    Deliberately requires substantially more text than a single unit
    does: a handful of words like "Information Total OK 2026" must still
    be reported as undetectable rather than resolved to a
    confident-looking wrong guess (EC-09).
    """
    joined = " ".join(units).strip()
    if len(joined) < MIN_FALLBACK_TEXT_CHARS:
        return Counter()
    detected = detect_language_for_text(joined)
    if detected is None:
        return Counter()
    logger.info(
        "Language detection fell back to whole-document text (%s chars) -> %s",
        len(joined),
        detected,
    )
    return Counter({detected: len(joined)})


def language_shares(language_counter: Counter[str]) -> dict[str, float]:
    """Return each language's share of detected characters, highest first."""
    total = sum(language_counter.values())
    if not total:
        return {}
    return {
        language: chars / total for language, chars in language_counter.most_common()
    }


def significant_languages(language_counter: Counter[str]) -> Counter[str]:
    """Drop languages holding a negligible share of detected characters.

    Excluding incidental languages from the dominance denominator is
    what lets a monolingual document survive residual detector noise: a
    brochure at 89% English plus four sub-5% artefacts is 100% English
    once the artefacts are removed, instead of being judged on an 89%
    that a slightly noisier document would fall below. Genuine minority
    content sits far above this floor and is retained.
    """
    total = sum(language_counter.values())
    if not total:
        return Counter()
    significant = Counter(
        {
            language: chars
            for language, chars in language_counter.items()
            if chars / total >= NOISE_LANGUAGE_SHARE
        }
    )
    # Pathological case: enough languages that every one is individually
    # below the floor. Judge on the full distribution rather than an
    # empty one.
    return significant or language_counter


def aggregate_languages(language_counter: Counter[str]) -> str | None:
    """Return the dominant language from a char-weighted Counter, or None if empty."""
    if not language_counter:
        return None
    detected_language, _ = language_counter.most_common(1)[0]
    return detected_language


def _format_shares(language_counter: Counter[str]) -> str:
    return ", ".join(
        f"{language} {share:.0%}"
        for language, share in language_shares(language_counter).items()
    )


def resolve_dominant_language(
    language_counter: Counter[str], *, source_label: str
) -> str:
    """Return the language to translate from, or raise if there isn't one.

    Raises `MixedLanguageError` when the document is genuinely
    multilingual -- no language holds `MIN_DOMINANT_LANGUAGE_SHARE` of
    the non-incidental characters -- and `ValueError` when nothing
    detectable was found at all.
    """
    if not language_counter:
        raise ValueError(
            f"Unable to detect a source language: no sufficiently long, "
            f"confidently detectable text was found in {source_label}. "
            "Please choose the source language explicitly."
        )

    significant = significant_languages(language_counter)
    dominant = aggregate_languages(significant)
    significant_total = sum(significant.values())
    dominant_share = significant[dominant] / significant_total

    if dominant_share >= MIN_DOMINANT_LANGUAGE_SHARE:
        return dominant

    # Not enough text sampled to distinguish a genuinely mixed document
    # from one or two unlucky units -- prefer translating over failing.
    if significant_total < MIN_MIXED_DECISION_CHARS:
        logger.info(
            "Dominant language %s holds only %.0f%% of %s, but only %s chars "
            "were detected -- accepting rather than rejecting as mixed-language.",
            dominant,
            dominant_share * 100,
            source_label,
            significant_total,
        )
        return dominant

    raise MixedLanguageError(
        f"This document is mixed-language ({_format_shares(significant)}): no "
        f"single language covers at least {MIN_DOMINANT_LANGUAGE_SHARE:.0%} of "
        f"the text in {source_label}, so it cannot be reliably translated from "
        "one source language. Please split the document by language, or specify "
        "the source language explicitly to translate it as-is.",
        language_shares=language_shares(significant),
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
