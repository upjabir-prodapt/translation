"""Runtime sources for the vocabulary the glossary hygiene gate needs.

The gate used to carry ~840 hand-typed words compiled into Python across six
hardcoded languages. That failed in three ways that had nothing to do with the
rules being wrong:

* a language outside the typed set got **no** coverage at all, silently --
  including `pt` and `nl`, which the UAT round explicitly asked Colt to add;
* the prompt-echo list was a copy of the extractor prompt's own wording, so it
  stopped working the moment anyone edited the prompt;
* the "is this a person's name or business vocabulary?" list had to be extended
  by hand for every new domain.

Everything here is therefore either loaded from data or derived from something
that already exists in the system:

``function words, connectives, generic terms``
    Loaded from ``glossary_vocabulary.json``, keyed by ISO code. The set of
    languages that *matter* comes from ``language_mapper.json``, so adding a
    language to the product surfaces a missing-coverage warning instead of
    quietly degrading.

``month names``
    Derived from Babel's CLDR data for every supported locale, in wide and
    abbreviated forms. No hand-typed month list to fall behind.

``prompt echo``
    Derived by reading the actual prompt templates at import. If someone
    rewrites the extractor prompt, the terms this rejects change with it.

``domain vocabulary``
    Derived from the curated glossary at call time (see
    :class:`~src.config.glossary_hygiene.HygieneContext`). As Colt's glossary
    grows, the name heuristic improves on its own.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

VOCABULARY_PATH = Path(__file__).with_name("glossary_vocabulary.json")

_WHITESPACE_PATTERN = re.compile(r"\s+")
_WORD_PATTERN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def normalize_term(term: str) -> str:
    """Casefold, NFKC-normalise and collapse whitespace.

    NFKC matters for Japanese: the shipped glossaries mix half-width katakana
    (``ｹ-ﾌﾞﾙ``) with full-width, so a naive comparison sees two different terms
    and an identity pair slips through as a non-identity one.
    """
    folded = unicodedata.normalize("NFKC", str(term)).casefold()
    return _WHITESPACE_PATTERN.sub(" ", folded).strip()


@functools.lru_cache(maxsize=1)
def _raw_vocabulary() -> dict:
    try:
        return json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(
            "Could not read %s; the glossary hygiene gate will run with its "
            "structural rules only (identity, placeholders, length, shape). "
            "Those still reject the majority of junk, but closed-class words "
            "will get through.",
            VOCABULARY_PATH,
        )
        return {}


@functools.lru_cache(maxsize=1)
def supported_languages() -> frozenset[str]:
    """Canonical codes the product accepts, straight from language_mapper.json.

    Imported lazily: `translation_routing` reads a JSON asset of its own, and
    importing it at module scope would make this module's import order matter.
    """
    try:
        from src.config.translation_routing import get_language_mapper

        return frozenset(get_language_mapper().values())
    except Exception:
        logger.warning("Could not read the language mapper", exc_info=True)
        return frozenset()


@functools.lru_cache(maxsize=1)
def function_words() -> frozenset[str]:
    """Closed-class words across every language present in the data file.

    Flattened across languages on purpose. One glossary file serves several
    language pairs, and a surface form that is a function word in *any*
    supported language is too dangerous to match document-wide -- ``die`` is a
    German article and an English verb, ``sono`` an Italian copula.
    """
    data = _raw_vocabulary().get("function_words", {})
    words: set[str] = set()
    for terms in data.values():
        words.update(normalize_term(t) for t in terms)
    words.discard("")
    return frozenset(words)


@functools.lru_cache(maxsize=1)
def languages_missing_coverage() -> tuple[str, ...]:
    """Supported languages with no closed-class vocabulary.

    Surfaced by :func:`log_coverage` at startup rather than discovered when a
    reviewer reports untranslated Portuguese articles in a UAT round.
    """
    covered = set(_raw_vocabulary().get("function_words", {}))
    return tuple(sorted(supported_languages() - covered))


@functools.lru_cache(maxsize=1)
def connective_phrases() -> frozenset[str]:
    return frozenset(
        normalize_term(t) for t in _raw_vocabulary().get("connective_phrases", [])
    )


@functools.lru_cache(maxsize=1)
def generic_non_terms() -> frozenset[str]:
    return frozenset(
        normalize_term(t) for t in _raw_vocabulary().get("generic_non_terms", [])
    )


@functools.lru_cache(maxsize=1)
def organisation_suffixes() -> frozenset[str]:
    return frozenset(
        normalize_term(t) for t in _raw_vocabulary().get("organisation_suffixes", [])
    )


@functools.lru_cache(maxsize=1)
def domain_vocabulary_seed() -> frozenset[str]:
    """Fallback business vocabulary for the person-name heuristic.

    Used only when a caller supplies no glossary-derived vocabulary. The
    derived set is always preferable: it cannot fall behind a new domain.
    """
    return frozenset(
        normalize_term(t) for t in _raw_vocabulary().get("domain_vocabulary_seed", [])
    )


@functools.lru_cache(maxsize=1)
def month_names() -> frozenset[str]:
    """Month names in every supported locale, from Babel's CLDR data.

    Replaces a hand-typed alternation that covered five languages and would
    have needed another edit for each new one.
    """
    names: set[str] = set()
    try:
        from babel.dates import get_month_names
    except Exception:
        logger.warning(
            "Babel unavailable; month-literal detection is disabled", exc_info=True
        )
        return frozenset()

    for code in supported_languages() | {"en"}:
        for width in ("wide", "abbreviated"):
            try:
                names.update(
                    normalize_term(name)
                    for name in get_month_names(width, locale=code).values()
                )
            except Exception:  # noqa: BLE001 - an unknown locale is not fatal
                logger.debug("No CLDR month data for %r (%s)", code, width)
    names.discard("")
    return frozenset(names)


@functools.lru_cache(maxsize=1)
def prompt_echo_terms() -> frozenset[str]:
    """Distinctive phrases from our own prompts, read from the prompt sources.

    The extractor prompt tells the model to return "named entities" and
    "domain-specific nouns"; the shipped ``legal.json`` duly contains
    ``named entities`` and ``domain-specific nouns`` as glossary terms -- the
    model extracted terminology from its own instructions.

    Deriving these from the template text means the rejection list tracks the
    prompt instead of drifting away from it. Only multi-word phrases are taken:
    single words from the prompt are ordinary English and would over-reject.
    """
    templates: list[str] = []
    try:
        from src.worker.doctranslator.format.pdf.document_il.midend import (
            automatic_term_extractor,
        )

        templates.append(automatic_term_extractor.LLM_PROMPT_TEMPLATE)
    except Exception:
        logger.debug("PDF extractor prompt unavailable", exc_info=True)
    try:
        from src.worker.doctranslator.format.docx import term_extractor

        templates.append(term_extractor._PROMPT_TEMPLATE)
    except Exception:
        logger.debug("DOCX extractor prompt unavailable", exc_info=True)

    phrases: set[str] = set()
    for template in templates:
        # Strip the substitution slots so "{target_language}" does not become a
        # phrase, then take 2-4 word windows of ordinary words.
        cleaned = re.sub(r"\{[^}]*\}", " ", template)
        for line in cleaned.splitlines():
            words = [normalize_term(w) for w in _WORD_PATTERN.findall(line)]
            for size in (2, 3):
                for i in range(len(words) - size + 1):
                    phrases.add(" ".join(words[i : i + size]))
    return frozenset(p for p in phrases if p)


def log_coverage() -> None:
    """Emit one startup line describing what the gate actually knows."""
    missing = languages_missing_coverage()
    logger.info(
        "Glossary hygiene vocabulary: %d function words across %d language(s), "
        "%d month names, %d prompt phrases, source=%s",
        len(function_words()),
        len(_raw_vocabulary().get("function_words", {})),
        len(month_names()),
        len(prompt_echo_terms()),
        _raw_vocabulary().get("source", "unknown"),
    )
    if missing:
        logger.warning(
            "No closed-class vocabulary for supported language(s): %s. Terms in "
            "those languages are still checked structurally (identity, "
            "placeholders, length, shape) but their articles and prepositions "
            "will not be recognised. Add them to %s.",
            ", ".join(missing),
            VOCABULARY_PATH.name,
        )


def reset_caches() -> None:
    """Drop every cached lookup. For tests and for hot-reloading the data file.

    Tolerates an accessor that has been patched out: a test that swaps one for
    a plain callable should not have to know which of these are memoised.
    """
    for fn in (
        _raw_vocabulary,
        supported_languages,
        function_words,
        languages_missing_coverage,
        connective_phrases,
        generic_non_terms,
        organisation_suffixes,
        domain_vocabulary_seed,
        month_names,
        prompt_echo_terms,
    ):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
