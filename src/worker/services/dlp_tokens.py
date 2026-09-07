"""Shared DLP placeholder-token format helpers (mask -> LLM -> unmask).

The masking side (`dlp_service.py`) replaces detected PII with
`__DLP_TOKEN_NNNN__` placeholders, the document is translated with those
placeholders in place, and the unmasking side (PDF `dlp_adapter.py`, DOCX
`docx_translator.py`) restores the original values.

Restoration used to be an exact `str.replace()` per token. That is too
brittle in practice, and it fails most often on German:

* German inflects and compounds, so a PII finding frequently covers only
  part of a word ("Mustermann" inside "Mustermanns", "Müller" inside
  "Müller-Schmidt"). The placeholder is then emitted welded to the
  remaining letters -- `__DLP_TOKEN_0001__s` -- which the model reads as a
  single German word rather than as an opaque token.
* Once welded, the model routinely "tidies" the token while translating
  the word it appears to be part of: it changes case, swaps `_` for `-` or
  a space, or drops a leading/trailing underscore pair. The token no
  longer matches exactly, so restoration silently misses it.

The consequence differed per format and both outcomes were wrong: the PDF
pipeline stripped the unmatched token (deleting the PII value from the
delivered document), while the DOCX pipeline had no strip step at all and
shipped the literal `__DLP_TOKEN_0001__` glued to the neighbouring word --
the reported symptom.

`restore_tokens()` therefore matches tokens by their *numeric id* through
`LENIENT_DLP_TOKEN_RE`, tolerating the formatting drift above, in a single
regex pass (so a token welded to surrounding text is still recovered).
`strip_leaked_tokens_from_text()` is the shared last-resort cleanup for
anything that still could not be resolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field

# Canonical emitted form, e.g. __DLP_TOKEN_0001__. Used when *writing*
# tokens and for strict detection of a well-formed leftover token.
DLP_TOKEN_RE = re.compile(r"__DLP_TOKEN_\d{4,}__")

# Recovery form used when *reading* tokens back out of an LLM response.
# Deliberately tolerant of the drift described in the module docstring:
# any case, 0-4 underscores on either side, and `_`, `-` or a space as the
# internal separators. The `DLP`/`TOKEN` words plus a digit run are still
# required, so ordinary prose cannot match by accident.
LENIENT_DLP_TOKEN_RE = re.compile(
    r"_{0,4}[ \t]*DLP[ \t_-]*TOKEN[ \t_-]*(\d{1,10})[ \t]*_{0,4}",
    re.IGNORECASE,
)


def format_token(token_id: int) -> str:
    """Render the canonical placeholder for `token_id`."""
    return f"__DLP_TOKEN_{token_id:04d}__"


@dataclass(slots=True)
class TokenMap:
    """Lookup for restoring masked values.

    `by_id` holds tokens in the canonical `__DLP_TOKEN_NNNN__` family, keyed
    by numeric id rather than by the literal string so a token whose
    formatting the model altered is still resolvable. `literal` holds any
    other token shape a caller supplied (tests and callers that pre-compute
    their own DLP result are free to use a different placeholder format);
    those can only ever be matched exactly.
    """

    by_id: dict[int, str] = field(default_factory=dict)
    literal: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.by_id or self.literal)


def build_token_map(token_rows: list[dict]) -> TokenMap:
    """Build a `TokenMap` from persisted DLP token rows."""
    token_map = TokenMap()
    for row in token_rows or []:
        token = row.get("token")
        original = row.get("original_value")
        if not token or original is None:
            continue
        token = str(token)
        match = LENIENT_DLP_TOKEN_RE.fullmatch(token)
        if match:
            token_map.by_id[int(match.group(1))] = str(original)
        else:
            token_map.literal[token] = str(original)
    return token_map


def restore_tokens(text: str, token_map: TokenMap) -> tuple[str, int]:
    """Replace every recoverable token in `text` with its original value.

    Returns `(restored_text, replaced_count)`. Tokens whose id is unknown are
    left untouched for the caller's leak check to report.
    """
    if not text or not token_map:
        return text, 0

    replaced = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal replaced
        original = token_map.by_id.get(int(match.group(1)))
        if original is None:
            return match.group(0)
        replaced += 1
        return original

    if token_map.by_id:
        text = LENIENT_DLP_TOKEN_RE.sub(_sub, text)

    for token, original in token_map.literal.items():
        occurrences = text.count(token)
        if occurrences:
            text = text.replace(token, original)
            replaced += occurrences

    return text, replaced


def find_leaked_tokens(text: str) -> list[str]:
    """Return every DLP token still present in `text` after restoration."""
    if not text:
        return []
    return [match.group(0) for match in LENIENT_DLP_TOKEN_RE.finditer(text)]


def strip_leaked_tokens_from_text(text: str) -> tuple[str, list[str]]:
    """Remove any unresolved DLP tokens. Returns `(cleaned_text, leaked)`.

    A non-empty `leaked` list means restoration failed for those tokens and
    the original sensitive values could not be put back -- callers must
    treat it as a pipeline fault, not a cosmetic cleanup.
    """
    leaked = find_leaked_tokens(text)
    if not leaked:
        return text, []
    return LENIENT_DLP_TOKEN_RE.sub("", text), leaked


@dataclass(slots=True)
class MaskApplier:
    """Re-applies an existing DLP masking to text derived from the IL.

    The PDF pipeline masks `PdfParagraph.unicode`, but the text actually sent
    to the LLM is rebuilt from the paragraph's *characters* whenever the
    paragraph has more than one composition (rich-text spans, formulas), and
    the single-paragraph fallback translator rebuilds it unconditionally.
    Those rebuilt strings never carried the masking, so raw PII reached the
    model for most paragraphs.

    Rather than trying to keep two representations in sync, this replays the
    substitutions the masking pass already decided on: every original value
    recorded in the job's token rows is mapped back to its placeholder. A
    value that DLP found in one paragraph is therefore also masked wherever
    else it appears, which is strictly safer and still round-trips exactly,
    since unmasking resolves each placeholder to that same value.
    """

    pattern: re.Pattern[str] | None
    token_by_original: dict[str, str]
    info_type_by_token: dict[str, str]

    def __bool__(self) -> bool:
        return self.pattern is not None

    def apply(self, text: str) -> str:
        """Return `text` with every known original value replaced by its token."""
        if not text or self.pattern is None:
            return text
        return self.pattern.sub(
            lambda match: self.token_by_original[match.group(0)], text
        )

    def missing_info_types(self, reference: str, masked: str) -> list[str]:
        """Info types masked in `reference` but absent from `masked`.

        The masking pass already decided which values in this paragraph are
        sensitive and recorded them as tokens in the reference text. If a
        token it assigned is missing from the text about to be sent to the
        model, that value could not be re-masked -- typically because a
        rich-text or formula placeholder was inserted in the middle of it --
        and it is about to leave the process in the clear.
        """
        if not reference:
            return []
        present = set(DLP_TOKEN_RE.findall(masked or ""))
        return sorted(
            {
                self.info_type_by_token.get(token, "UNKNOWN")
                for token in DLP_TOKEN_RE.findall(reference)
                if token not in present
            }
        )


def build_mask_applier(token_rows: list[dict]) -> MaskApplier:
    """Compile a `MaskApplier` from a job's persisted DLP token rows.

    Originals are matched longest-first so a value contained inside a longer
    one cannot mask it first, and word-character edges are anchored so a
    short value ("45") cannot match inside an unrelated word ("1452").
    """
    token_by_original: dict[str, str] = {}
    info_type_by_token: dict[str, str] = {}
    for row in token_rows or []:
        token = row.get("token")
        original = row.get("original_value")
        if not token or not original:
            continue
        # First writer wins: any token for this value restores to this value.
        token_by_original.setdefault(str(original), str(token))
        info_type_by_token[str(token)] = str(row.get("info_type") or "UNKNOWN")

    if not token_by_original:
        return MaskApplier(None, {}, {})

    alternatives = []
    for original in sorted(token_by_original, key=len, reverse=True):
        prefix = r"(?<!\w)" if original[:1].isalnum() or original[:1] == "_" else ""
        suffix = r"(?!\w)" if original[-1:].isalnum() or original[-1:] == "_" else ""
        alternatives.append(f"{prefix}{re.escape(original)}{suffix}")

    return MaskApplier(
        re.compile("|".join(alternatives)),
        token_by_original,
        info_type_by_token,
    )
