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

Mixed-language decision
-----------------------
The original guard rejected a document when the *count* of distinct
languages on a single page exceeded a fixed limit. That was unusable in
practice: a wholly monolingual English brochure (all-caps headings plus
a network map full of city names) registers four "languages" on one
page, because `langdetect` is ~0.99999 confident that "THE EXTRAORDINARY
EVERYDAY." is Spanish and "Richmond Richmond Richmond" is German. A
26-character run of proper nouns counted exactly as much toward the
limit as 2,400 characters of real prose.

Detection is now two layers, both defined here so every format shares
them:

1. Per-unit signal quality (`is_high_signal_unit`,
   `case_normalize_for_detection`) discards the text where `langdetect`
   is known to be unreliable -- too short, too few words, all-caps (its
   n-gram profiles are lowercase), or a list of proper nouns -- so noise
   never enters the distribution in the first place.

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

from langdetect import DetectorFactory
from langdetect import LangDetectException
from langdetect import detect_langs

from src.config.constants import settings
from src.config.translation_routing import get_language_mapper

logger = logging.getLogger(__name__)

# langdetect is non-deterministic by default (uses a random seed
# internally); pin it so repeated detection of the same text is stable.
# `processor_service.py` also sets this -- both assignments are
# idempotent (module-level, same value), so importing either module
# first has no effect on the other.
DetectorFactory.seed = 0

# Relaxed floor. Only gates which text is *considered* at all, and is
# also the floor used by the whole-document fallback pass; counting a
# unit toward the language distribution requires `is_high_signal_unit`.
MIN_DETECTION_TEXT_LENGTH = 20
MIN_DETECTION_ALPHA_CHARS = 5
MIN_DETECTION_CONFIDENCE = 0.80

# --- Layer 1: per-unit signal quality --------------------------------
# langdetect is overconfident on short text -- it returns p=0.99999 for
# "Richmond Richmond Richmond" (de) -- so the confidence floor above
# cannot filter these on its own. Length and shape can.
MIN_UNIT_TEXT_LENGTH = settings.LANGUAGE_DETECTION_MIN_UNIT_CHARS
MIN_UNIT_WORD_COUNT = settings.LANGUAGE_DETECTION_MIN_UNIT_WORDS

# Above this share of cased letters being uppercase, the unit is treated
# as all-caps display text and lowercased before detection: langdetect's
# n-gram profiles are built from lowercase text, so ALL-CAPS English
# matches foreign profiles at near-certainty ("SUPPORTING YOUR BUSINESS
# WITH OUR BACKBONE" -> de 0.99999; lowercased -> en 0.99999).
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

# langdetect emits ISO codes that don't always match this service's
# canonical set (language_mapper.json); normalize the handful of aliases
# actually seen in practice.
DETECTED_LANGUAGE_ALIASES: dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "iw": "he",
}


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
    """Return True if `text` has enough signal for langdetect to be trusted."""
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return (
        len(text) >= MIN_DETECTION_TEXT_LENGTH
        and alpha_count >= MIN_DETECTION_ALPHA_CHARS
    )


def case_normalize_for_detection(text: str) -> str:
    """Lowercase `text` when it is predominantly uppercase.

    langdetect's language profiles are lowercase n-grams, so all-caps
    headings -- ubiquitous in brochures, slide decks and marketing PDFs
    -- are matched against the wrong profiles with near-total
    confidence. Lowercasing restores the correct match; text that is
    already mixed-case is returned untouched.
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
    carries no language signal -- langdetect reports Tagalog and
    Catalan respectively, at high confidence. Names are overwhelmingly
    capitalized, which is what this detects.
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
    """Map a raw langdetect code to this service's canonical alias, if any."""
    normalized = str(language).strip().lower()
    return DETECTED_LANGUAGE_ALIASES.get(normalized, normalized)


def detect_language_for_text(text: str) -> str | None:
    """Detect the language of one chunk of text, or None if not confident.

    Returns `None` (rather than a low-confidence guess) when langdetect
    itself fails, returns nothing, or reports a confidence below
    `MIN_DETECTION_CONFIDENCE` -- the confidence floor that DOCX/TXT
    detection previously lacked entirely (EC-09). The text is
    case-normalized first (see `case_normalize_for_detection`).
    """
    try:
        candidates = detect_langs(case_normalize_for_detection(text))
    except LangDetectException:
        return None
    if not candidates:
        return None
    best_match = candidates[0]
    if best_match.prob < MIN_DETECTION_CONFIDENCE:
        return None
    return normalize_detected_language(best_match.lang)


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
    document as a whole is unambiguous. Concatenating is exactly the
    case langdetect handles best (long text), so this recovers them
    without weakening the per-unit floors.

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
