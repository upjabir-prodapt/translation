"""Single gate deciding whether a (source_term, target_term) pair may enter a glossary.

Every path that can add a term to a glossary routes through
:func:`evaluate_term_pair` -- the PDF term extractor, the DOCX term extractor,
the merge that persists learned terms, and the read path that loads a domain
glossary back. There is deliberately exactly one implementation: the UAT round
analysed in TRANSLATION_FIX_PLAN.md failed because two extractors carried two
different (and both too weak) guards, so a class of junk that one rejected the
other happily wrote.

Applying the gate on *read* as well as on *write* is not redundant. Glossary
data already in storage contains thousands of bad entries; filtering on read
makes that pollution inert the moment this code deploys.

**This module holds rules, not vocabulary.** Every word list it consults is
loaded or derived at runtime by :mod:`src.config.linguistic_data`: closed-class
words from ``glossary_vocabulary.json`` keyed by ISO code, month names from
Babel's CLDR data, prompt-echo phrases from the actual prompt templates, and
business vocabulary from the curated glossary in hand. Every threshold comes
from ``settings``. Adding a language, editing a prompt or onboarding a domain
therefore needs no change here.

Rejection reasons, and the UAT defect each one prevents:

``identity``
    ``source == target``. The single biggest cause of the "residual
    source-language words" complaints: the extractor emitted ``der -> der``,
    ``los -> los``, ``integrity -> integrity``, and the translation prompt then
    carried a standing instruction to leave that word alone. A term that must
    genuinely survive translation unchanged belongs in a ``preserve_as_is``
    list, which is authored by hand and never learned.

``function_word``
    Closed-class words in any supported language, and fixed discourse
    connectives such as ``a pesar de ello``.

``too_short`` / ``too_long`` / ``too_many_words`` / ``heading``
    Structural: terminology is a short noun phrase, not a token or a clause.

``placeholder`` / ``prompt_marker`` / ``prompt_echo``
    Contamination from our own pipeline -- DLP tokens, injection-guard
    delimiters, and vocabulary the model lifted out of its own instructions.

``person_name`` / ``date_literal`` / ``instance_value``
    Document *instances* rather than terminology. They bloat the glossary,
    carry personal data into a shared asset, and never generalise.

``length_ratio``
    Target wildly disproportionate to source, which signals a truncated or
    hallucinated translation. Script-aware, so CJK is not punished for density.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum

from src.config import linguistic_data
from src.config.constants import settings
from src.config.linguistic_data import normalize_term

__all__ = [
    "HygieneContext",
    "TermVerdict",
    "Trust",
    "evaluate_term_pair",
    "is_acceptable_term_pair",
    "normalize_term",
    "sanitize_term_pairs",
]

# ---------------------------------------------------------------------------
# Structural shapes. These are language-independent by construction, which is
# why they live in code while vocabulary does not.
# ---------------------------------------------------------------------------

#: DLP tokens in every mangled shape observed in shipped glossaries
#: (``__DLP_TOKEN_0001__``, ``DLP_TOKEN_0028``), plus formula/rich-text
#: placeholders and format specifiers.
_PLACEHOLDER_PATTERN = re.compile(
    r"""
      _{0,2}DLP[_\s-]?TOKEN[_\s-]?\d+_{0,2}
    | \{\{?\s*[\w.]+\s*\}?\}
    | </?[a-z]+\d+>
    | %[sdfx]\b
    | \[\[.*?\]\]
    | %%.*?%%
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Injection-guard delimiters and jailbreak boilerplate.
_PROMPT_MARKER_PATTERN = re.compile(
    r"TRANSLATE_CONTENT_(?:START|END)|ignore\s+(?:all\s+)?previous"
    r"|system\s+prompt|as\s+an\s+AI",
    re.IGNORECASE,
)

#: Two or three capitalised words -- the shape of a personal name. Vocabulary
#: decides whether it actually is one; see :func:`_looks_like_person_name`.
_PERSON_NAME_PATTERN = re.compile(r"^[A-Z][a-z’']+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z’']+$")

#: Identifier-shaped instance values: invoice, ticket and version references.
_INSTANCE_VALUE_PATTERN = re.compile(
    r"""
      ^[A-Z]{2,}[-_/]?\d{2,}[A-Z0-9-]*$
    | ^\d+[A-Za-z]?$
    | ^v?\d+(?:\.\d+)+[a-z]?$
    | ^[0-9a-f]{8,}$
    """,
    re.VERBOSE,
)

#: Segmented uppercase codes (``LON-PE-01``). The digit requirement keeps
#: genuine hyphenated terminology (``SD-WAN``, ``P&L``) out of it.
_SEGMENTED_CODE_PATTERN = re.compile(r"^[A-Z0-9]{1,8}(?:[-_/][A-Z0-9]{1,8}){1,4}$")
_HAS_DIGIT_PATTERN = re.compile(r"\d")

#: A version literal anywhere makes the term an instance, not a concept.
_EMBEDDED_VERSION_PATTERN = re.compile(r"\bv\d+(?:\.\d+)+\b", re.IGNORECASE)

#: Clause and heading punctuation. The ``(?<!\d)`` guard keeps ordinals out:
#: "95. Perzentil" is a German ordinal, not a sentence break.
_HEADING_PATTERN = re.compile(
    r"(?<!\d)[.;:]\s+\S|[—–]\s|\s[—–]|→|\band\b.*\band\b", re.IGNORECASE
)

#: A date literal: an all-numeric date, or a month name adjacent to a number.
#: The month vocabulary comes from CLDR, so this covers every supported locale
#: rather than the five a hand-typed alternation happened to list.
_NUMERIC_DATE_PATTERN = re.compile(
    r"^\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\s*$|^\s*\d{1,4}\s*年"
)
_DATE_ADJACENCY_PATTERN = re.compile(
    r"(\d{1,4})\s*(?:st|nd|rd|th)?[\s./-]*([^\W\d_]{3,})"
    r"|([^\W\d_]{3,})[\s./-]*(\d{1,4})",
    re.UNICODE,
)

#: An all-caps token of 2-6 characters is an acronym. Acronyms are real
#: terminology (``IVR`` -> ``SVI``) and are exempt from the single-token length
#: floor and the length-ratio test, both of which assume a word-for-word pair.
#: They are *not* exempt from the identity rule.
_ACRONYM_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9&/.-]{1,5}$")

_CJK_PATTERN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
_LETTER_PATTERN = re.compile(r"[^\W\d_]", re.UNICODE)
_PUNCT_TO_SPACE_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_TOKEN_PATTERN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


class Trust(StrEnum):
    """How much the gate should second-guess where an entry came from.

    The rules split cleanly along provenance, and collapsing that into one
    boolean was wrong in both directions.

    ``LEARNED`` (default)
        A model wrote this. Every rule applies. The vocabulary heuristics exist
        precisely for this case -- to stop the extractor teaching itself that
        ``der`` means ``der`` and that ``Elon Musk`` is terminology.

    ``CURATED``
        A human authored this translation pair. Structural rules and the
        identity rule still apply -- an identity pair is the defect the whole
        gate exists to stop, and a human typing one has still made a mistake --
        but the vocabulary heuristics do not. They cannot tell ``The Hague ->
        La Haya`` from ``Elon Musk -> イーロン・マスク``, and overruling a
        reviewer on a judgement call they made deliberately is not the gate's
        job.

    ``PRESERVE``
        A human authored a do-not-translate entry. Identity is the *intent*,
        so only the structural rules apply. This is also why the vocabulary
        heuristics must be off: ``SoW``, ``PoP`` and ``QoS`` are mixed case so
        they read as "too short", ``DID`` collides with an English verb, and
        ``On Demand`` has the shape of a name.
    """

    LEARNED = "learned"
    CURATED = "curated"
    PRESERVE = "preserve"

    @classmethod
    def from_origin(cls, origin: str | None) -> Trust:
        """Map a stored ``origin`` field to a trust level.

        Unknown or missing origin means LEARNED. Defaulting the other way would
        let a legacy file -- which mixes hand-authored and machine-written
        entries in one undifferentiated structure -- claim trust it has not
        earned.
        """
        try:
            return cls(str(origin or "").strip().lower())
        except ValueError:
            return cls.LEARNED


@dataclass(frozen=True)
class TermVerdict:
    """Outcome of the gate. `reason` is None exactly when `accepted` is True."""

    accepted: bool
    reason: str | None = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.accepted


ACCEPTED = TermVerdict(True)


@dataclass(frozen=True)
class HygieneContext:
    """Per-call knowledge that sharpens the heuristics.

    `domain_vocabulary` is the set of words appearing in terminology already
    trusted for this domain. It is what tells ``Legal Entity`` and ``Account
    Executive`` apart from ``Elon Musk``: all three are two capitalised words,
    and only vocabulary distinguishes them.

    Deriving it from the curated glossary rather than a hand-maintained list
    means onboarding a new domain improves the heuristic automatically, and a
    domain nobody anticipated is not silently mis-served.
    """

    domain_vocabulary: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_terms(cls, terms: Iterable[str]) -> HygieneContext:
        """Build vocabulary from trusted terminology.

        Pass terms you already trust -- a curated glossary, a ``preserve_as_is``
        list. Never pass the candidates being judged: a set containing the term
        under test would whitelist it.
        """
        vocabulary: set[str] = set()
        for term in terms:
            vocabulary.update(
                normalize_term(token) for token in _TOKEN_PATTERN.findall(str(term))
            )
        vocabulary.discard("")
        return cls(domain_vocabulary=frozenset(vocabulary))

    def known_words(self) -> frozenset[str]:
        """Seed vocabulary plus whatever this domain's glossary adds.

        A union, not a choice. The seed is general business vocabulary
        ("account", "executive", "liability"); the derived set is whatever this
        domain happens to use. Treating the derived set as a *replacement*
        meant supplying any vocabulary at all silently dropped the general
        floor, and ``Account Executive`` started reading as a personal name.
        """
        return linguistic_data.domain_vocabulary_seed() | self.domain_vocabulary


_EMPTY_CONTEXT = HygieneContext()


def _is_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text))


def _effective_length(text: str) -> float:
    """Character count with CJK weighted by its information density."""
    cjk = len(_CJK_PATTERN.findall(text))
    return (len(text) - cjk) + cjk * float(settings.GLOSSARY_CJK_DENSITY_FACTOR)


def _collapse_punctuation(text: str) -> str:
    """``domain-specific nouns`` -> ``domain specific nouns``.

    Prompt-echo phrases are derived by tokenising prompt text, which drops
    hyphens; candidate terms keep them. Comparing both in this form makes the
    two representations meet.
    """
    return normalize_term(_PUNCT_TO_SPACE_PATTERN.sub(" ", text))


def _looks_like_person_name(term: str, context: HygieneContext) -> bool:
    """Two capitalised words carrying no organisational or domain vocabulary.

    Necessarily a heuristic. A name built entirely from business words
    ("Mark Field") is not caught here; the identity rule catches it instead,
    since names pass through translation unchanged.
    """
    if not _PERSON_NAME_PATTERN.match(term.strip()):
        return False
    tokens = {normalize_term(t.strip(".,")) for t in term.split()}
    if tokens & linguistic_data.organisation_suffixes():
        return False
    return not (tokens & context.known_words())


def _is_date_literal(source: str) -> bool:
    """A month name adjacent to a number, in any supported locale."""
    if _NUMERIC_DATE_PATTERN.search(source):
        return True
    months = linguistic_data.month_names()
    if not months:
        return False
    for match in _DATE_ADJACENCY_PATTERN.finditer(source):
        word = match.group(2) or match.group(3) or ""
        if normalize_term(word) in months:
            return True
    return False


def evaluate_term_pair(
    source_term: str,
    target_term: str,
    *,
    trust: Trust = Trust.LEARNED,
    context: HygieneContext | None = None,
) -> TermVerdict:
    """Decide whether this pair may enter a glossary.

    `trust` says where the entry came from; see :class:`Trust` for what each
    level relaxes and why. Structural checks -- empty, DLP tokens, prompt
    markers, absurd length, heading shape -- apply at every level, because they
    catch corruption rather than word choice.
    """
    ctx = context or _EMPTY_CONTEXT
    source = str(source_term or "").strip()
    target = str(target_term or "").strip()

    if not source or not target:
        return TermVerdict(False, "empty")

    # Contamination first: the reason is more useful in the logs than a
    # downstream length verdict would be.
    for value in (source, target):
        if _PLACEHOLDER_PATTERN.search(value):
            return TermVerdict(False, "placeholder")
        if _PROMPT_MARKER_PATTERN.search(value):
            return TermVerdict(False, "prompt_marker")

    norm_source = normalize_term(source)
    norm_target = normalize_term(target)

    if _collapse_punctuation(source) in linguistic_data.prompt_echo_terms():
        return TermVerdict(False, "prompt_echo")

    if not _LETTER_PATTERN.search(source):
        return TermVerdict(False, "no_letters")

    max_chars = int(settings.GLOSSARY_MAX_TERM_CHARS)
    if len(source) > max_chars or len(target) > max_chars:
        return TermVerdict(False, "too_long")

    word_count = len(source.split())
    if word_count > int(settings.GLOSSARY_MAX_TERM_WORDS):
        return TermVerdict(False, "too_many_words")

    # Two or more commas is an enumeration, not a term. One is left alone so
    # company forms such as "Data Live Co., Ltd." survive.
    if _HEADING_PATTERN.search(source) or source.count(",") >= 2:
        return TermVerdict(False, "heading")

    if _EMBEDDED_VERSION_PATTERN.search(source):
        return TermVerdict(False, "instance_value")

    # Identity is a defect for any *pair*, however it was authored: it is the
    # instruction that produced the residual source-language words. Only a
    # do-not-translate entry means it deliberately.
    if trust is not Trust.PRESERVE and norm_source == norm_target:
        return TermVerdict(False, "identity")

    # Everything below is a vocabulary heuristic, and a human has already made
    # these calls.
    if trust is not Trust.LEARNED:
        return ACCEPTED

    is_acronym = bool(_ACRONYM_PATTERN.match(source))

    min_chars = (
        int(settings.GLOSSARY_MIN_TOKEN_CHARS_CJK)
        if _is_cjk(source)
        else int(settings.GLOSSARY_MIN_TOKEN_CHARS)
    )
    if not is_acronym and word_count == 1 and len(source) < min_chars:
        return TermVerdict(False, "too_short")

    closed_class = linguistic_data.function_words()
    if (
        norm_source in closed_class
        or norm_source in linguistic_data.connective_phrases()
    ):
        return TermVerdict(False, "function_word")
    # A multi-word term built entirely of function words ("prima del") is just
    # as useless as a single one.
    tokens = [t.strip(".,;:()") for t in norm_source.split()]
    tokens = [t for t in tokens if t]
    if tokens and closed_class and all(t in closed_class for t in tokens):
        return TermVerdict(False, "function_word")

    if norm_source in linguistic_data.generic_non_terms():
        return TermVerdict(False, "generic_non_term")

    if _is_date_literal(source):
        return TermVerdict(False, "date_literal")

    if _INSTANCE_VALUE_PATTERN.match(source) or (
        _SEGMENTED_CODE_PATTERN.match(source) and _HAS_DIGIT_PATTERN.search(source)
    ):
        return TermVerdict(False, "instance_value")

    if _looks_like_person_name(source, ctx):
        return TermVerdict(False, "person_name")

    # Gated on the *longer* side so a one-character "translation" of a long
    # term is caught; guarding on the shorter side would divide that case out.
    # Acronyms are exempt: expanding one is legitimately lopsided.
    source_len = _effective_length(source)
    target_len = _effective_length(target)
    ratio_floor = max(2.0, float(settings.GLOSSARY_MIN_TOKEN_CHARS) * 2)
    if not is_acronym and max(source_len, target_len) >= ratio_floor:
        ratio = max(source_len, target_len) / max(1.0, min(source_len, target_len))
        if ratio > float(settings.GLOSSARY_MAX_LENGTH_RATIO):
            return TermVerdict(False, "length_ratio")

    return ACCEPTED


def is_acceptable_term_pair(
    source_term: str,
    target_term: str,
    *,
    trust: Trust = Trust.LEARNED,
    context: HygieneContext | None = None,
) -> bool:
    """Boolean form of :func:`evaluate_term_pair`, for call sites that only branch."""
    return evaluate_term_pair(
        source_term, target_term, trust=trust, context=context
    ).accepted


def sanitize_term_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    trust: Trust = Trust.LEARNED,
    context: HygieneContext | None = None,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Filter an iterable of ``(source, target)`` pairs.

    Returns the surviving pairs (de-duplicated on the normalized source, first
    occurrence wins) and a ``{reason: count}`` tally suitable for a log line.
    Callers log the tally rather than each rejection: a polluted batch would
    otherwise emit hundreds of lines.
    """
    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    rejected: dict[str, int] = {}

    for source_term, target_term in pairs:
        verdict = evaluate_term_pair(
            source_term, target_term, trust=trust, context=context
        )
        if not verdict.accepted:
            reason = verdict.reason or "unknown"
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        key = normalize_term(source_term)
        if key in seen:
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            continue
        seen.add(key)
        kept.append((str(source_term).strip(), str(target_term).strip()))

    return kept, rejected
