"""Post-translation protected-token verification (implementation_plan.md D.6).

UAT Edge Case Pack EC-01/EC-02/EC-14: source documents often contain
tokens that must survive translation completely unchanged -- digits,
currency amounts, URLs, email addresses, ticket/case IDs, CIDR blocks,
and CLI flags/switches. An LLM translation can occasionally "translate"
or subtly corrupt these (e.g. localizing a decimal separator, dropping a
URL scheme, renumbering a ticket ID) without the existing quality-judge
heuristics (alignment/omission/hallucination, all prose-level) ever
noticing, since they operate on sentence structure, not exact-token
survival.

This is a **lightweight, best-effort quality signal, not a hard gate**:
per D.6.1, a mismatch is logged as a quality warning and recorded in the
attempt's quality report, but never fails the job or blocks the
translated document from being delivered -- regex-based token extraction
is inherently approximate (a currency amount can legitimately be
reformatted for the target locale, e.g. "$1,234.56" -> "1 234,56 $"), so
treating every mismatch as fatal would create far more false-positive
job failures than it prevents real defects.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from dataclasses import field

logger = logging.getLogger(__name__)


# Each pattern extracts one category of "protected" token that should
# appear verbatim (case-sensitive) in the translated output. Order does
# not matter -- every pattern is applied independently to the full text.
_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "url": re.compile(r"https?://[^\s<>\"')]+"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    # $1,234.56 / €1.234,56 / £99 / 1234.56 USD -- currency symbol/code
    # adjacent to a number. Deliberately does NOT try to normalize
    # locale-specific thousands/decimal separators (see module docstring).
    "currency": re.compile(
        r"(?:[$€£¥]\s?\d[\d,.]*\d|\b\d[\d,.]*\d\s?(?:USD|EUR|GBP|JPY))\b"
    ),
    # CIDR block, e.g. 10.0.0.0/8 or 2001:db8::/32. Checked before the
    # generic digit pattern so a CIDR's component numbers are not also
    # double-reported as bare digit-sequence mismatches.
    "cidr": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}\b|\b[0-9a-fA-F:]+::?[0-9a-fA-F:]*/\d{1,3}\b"
    ),
    # Ticket/case IDs: PROJ-1234, INC0012345, #4567 -- an all-caps or
    # digit-led prefix followed by a numeric suffix.
    "ticket_id": re.compile(r"\b(?:[A-Z]{2,}-\d+|#\d{3,})\b"),
    # CLI flags/switches: --verbose, -v, /help (Windows-style).
    "cli_flag": re.compile(r"(?<!\S)(?:--[a-zA-Z][\w-]*|-[a-zA-Z]\b)"),
    # Bare digit sequences of 2+ digits, standalone (not already part of a
    # currency/CIDR/ticket-ID token, which are extracted first so this is
    # intentionally the most permissive/last-resort category). Requires a
    # non-digit boundary so it does not fragment the categories above.
    "digits": re.compile(r"(?<![\d.])\d{2,}(?:\.\d+)?(?![\d.])"),
}


@dataclass(slots=True)
class TokenVerificationResult:
    """Outcome of checking one (source, translated) text pair."""

    checked_token_count: int = 0
    missing_by_category: dict[str, list[str]] = field(default_factory=dict)

    @property
    def missing_count(self) -> int:
        return sum(len(v) for v in self.missing_by_category.values())

    @property
    def passed(self) -> bool:
        """True if every protected token found in the source also appears
        in the translated output. Purely informational -- see module
        docstring; never used to fail a job."""
        return self.missing_count == 0

    def to_dict(self) -> dict:
        return {
            "checked_token_count": self.checked_token_count,
            "missing_count": self.missing_count,
            "missing_by_category": self.missing_by_category,
            "passed": self.passed,
        }


def verify_protected_tokens(
    source_text: str, translated_text: str
) -> TokenVerificationResult:
    """Extract protected tokens from `source_text` and confirm each still
    appears verbatim in `translated_text`.

    Never raises and never fails a job (D.6.1) -- callers should log the
    result as a quality warning when `result.passed` is False and attach
    `result.to_dict()` to the attempt's quality report, but must not treat
    a mismatch as a translation failure.
    """
    result = TokenVerificationResult()
    if not source_text or not translated_text:
        return result

    for category, pattern in _TOKEN_PATTERNS.items():
        # Deduplicate while preserving first-seen order, so a token
        # repeated many times in the source (e.g. a URL in every
        # paragraph) is reported once, not once per occurrence.
        seen: set[str] = set()
        tokens: list[str] = []
        for match in pattern.finditer(source_text):
            token = match.group(0)
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        if not tokens:
            continue
        result.checked_token_count += len(tokens)
        missing = [token for token in tokens if token not in translated_text]
        if missing:
            result.missing_by_category[category] = missing

    return result


def log_token_verification_warning(
    result: TokenVerificationResult, *, job_id: str, attempt_index: int
) -> None:
    """Emit a structured warning log line for a failed verification.

    No-ops when `result.passed` is True. Deliberately a plain WARNING log
    (not an exception, not a job-status change) -- see module docstring.
    """
    if result.passed:
        return
    logger.warning(
        f"Protected-token verification found {result.missing_count} missing "
        f"token(s) out of {result.checked_token_count} checked for job="
        f"{job_id} attempt={attempt_index}: {result.missing_by_category}"
    )
