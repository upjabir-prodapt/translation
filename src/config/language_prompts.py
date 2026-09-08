"""Prompt block describing the secondary languages present in a document.

Language detection reports one dominant language per document and the
pipeline translates the whole document from that one language. On a
document that is 97% French with an English cover page, the English lines
are handed to the model as if they were French, and the model translates
them from the wrong source language.

This module renders a block, injected into every translation prompt, that
names the minority languages detection found and tells the model to judge
each segment for itself and translate it from whichever of those languages
it is actually written in.

It is deliberately a prompt-level fix: nothing here re-routes segments,
re-runs detection per segment, or splits the document. The model is simply
told what else is in the document and asked to act on the text in front of
it. That keeps a single translation pass and one source-language decision,
while removing the wrong-source-language failure on the minority lines.

Shares are whole-document character shares, so they describe the document,
not the specific batch a given prompt carries -- which is why the block
tells the model to trust the text over the list.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping

from src.config.translation_routing import get_language_display_name

# A language below this share of detected characters is left out of the
# prompt. Deliberately far below `LANGUAGE_DETECTION_NOISE_SHARE` (5%),
# which governs whether a language is ignored for the *mixed-language
# rejection*: the languages this block exists to rescue are exactly the
# ones that guard treats as incidental. On a real 9-page French document
# the genuine English cover page was 1.8% and genuine German contract
# terms 0.8%, both well under the noise share.
DEFAULT_MIN_SECONDARY_SHARE = 0.005

# Cap on how many languages the block names. Detection noise on a long
# document produces a long tail of sub-1% artefacts; listing all of them
# would spend prompt budget telling the model to expect languages that
# are not really there.
DEFAULT_MAX_SECONDARY_LANGUAGES = 4


def describe_language(code: str) -> str:
    """Render a language code for a prompt, as "English (en)".

    `get_language_display_name` title-cases codes it has no name for
    ("nl" -> "Nl"), which would read as a typo in a prompt, so an unnamed
    code is emitted bare -- models read ISO codes reliably.
    """
    code = str(code).strip()
    display = get_language_display_name(code)
    if display.strip().lower() == code.lower():
        return code
    return f"{display} ({code})"


def select_secondary_languages(
    distribution: Mapping[str, int | float] | None,
    *,
    primary_language: str,
    target_language: str,
    min_share: float = DEFAULT_MIN_SECONDARY_SHARE,
    max_languages: int = DEFAULT_MAX_SECONDARY_LANGUAGES,
) -> list[tuple[str, float]]:
    """Pick the minority languages worth naming, as `(code, share)` pairs.

    `distribution` is the char-weighted `Counter` from language detection.
    Returns the languages holding at least `min_share`, largest first,
    excluding two codes that would produce a contradictory instruction:

    * `primary_language` -- the language the document is already being
      translated from, so naming it as a *secondary* language would tell
      the model to treat the normal case as an exception.
    * `target_language` -- text already in the target language needs no
      translation, and "translate this from French into French" is an
      instruction with no useful reading.
    """
    if not distribution:
        return []
    total = sum(distribution.values())
    if total <= 0:
        return []

    excluded = {
        str(primary_language).strip().lower(),
        str(target_language).strip().lower(),
    }
    candidates = [
        (str(code).strip().lower(), count / total)
        for code, count in distribution.items()
        if str(code).strip().lower() not in excluded
    ]
    significant = [
        (code, share) for code, share in candidates if share >= min_share and code
    ]
    significant.sort(key=lambda item: item[1], reverse=True)
    return significant[:max_languages]


def format_share(share: float) -> str:
    """Render a share for the prompt, without collapsing small ones to 0%.

    The interesting shares here are small by definition -- a 0.8% German
    footnote is exactly the case this block exists for -- and "0% of the
    text" would read as "none", contradicting the list it appears in.

    Includes its own qualifier ("about" / "under") so the caller does not
    have to, which is what keeps sub-1% shares from rendering as the
    self-contradictory "about under 1%".
    """
    percent = share * 100
    if percent >= 10:
        return f"about {percent:.0f}%"
    if percent >= 1:
        return f"about {percent:.1f}%"
    return "under 1%"


def build_secondary_language_block(
    secondary_languages: Iterable[tuple[str, float]],
    *,
    primary_language: str,
    target_language: str,
) -> str:
    """Render the mixed-language instruction block, or "" if there is nothing to say.

    Returning the empty string for an empty language list keeps this
    callable unconditionally from every prompt builder: a monolingual
    document adds nothing to its prompt.
    """
    listed = list(secondary_languages)
    if not listed:
        return ""

    primary = describe_language(primary_language)
    target = describe_language(target_language)
    bullets = "\n".join(
        f"- {describe_language(code)} -- {format_share(share)} of the detected text"
        for code, share in listed
    )

    return (
        "## Mixed-Language Source\n"
        f"Automatic language detection found this document is predominantly "
        f"{primary}, but that it also contains text in these languages:\n\n"
        f"{bullets}\n\n"
        "Some segments below may therefore be written in one of those "
        f"languages rather than in {primary}. For every segment you translate:\n\n"
        "1. Judge from the text itself which language it is actually written in.\n"
        "2. If it is written in one of the languages listed above, translate it "
        f"from that language into {target}.\n"
        f"3. Otherwise treat it as {primary} and translate it into {target} as normal.\n\n"
        "Those percentages describe the whole document, not the specific "
        "segments in this request, and detection can be wrong -- so treat the "
        "list as a hint about what to expect and trust the text in front of "
        f"you over the list. Every segment must still come back in {target}: "
        "never leave one untranslated, and never copy it through unchanged, "
        f"because it was not written in {primary}.\n"
    )
