"""Shared quality guards for LLM translation output in the PDF pipeline.

Both PDF translators (`ILTranslator`, the single-paragraph path, and
`ILTranslatorLLMOnly`, the batch path) must refuse to write obviously
broken model output into a paragraph. The batch path grew these checks
first; the single path had none, which is how a leaked system prompt
(~2k chars) ended up as a paragraph's `unicode`, failed to typeset into
its ~75x9pt box, and was silently dropped from the output PDF with
`Unable to export paragraphs that have not yet been formatted`.

This module holds the two checks both paths share -- prompt-leak
detection and the output/input token-ratio band -- so there is exactly
one implementation and one set of thresholds.
"""

from __future__ import annotations

from collections.abc import Callable

from src.worker.doctranslator.translator.prompt_safety import looks_like_prompt_leak

# A faithful translation may legitimately expand or contract, but not by
# an order of magnitude. Anything outside this band is a formatting
# failure, a refusal, or leaked instructions -- never a usable
# translation.
TOKEN_RATIO_MIN = 0.3
TOKEN_RATIO_MAX = 3.0

# The single-paragraph translator is also the *fallback* the batch
# translator hands rejected paragraphs to. Applying the ratio rule there
# unconditionally would mean a short paragraph with a legitimately
# expansive translation ("GmbH" -> "limited liability company") is rejected
# twice and left untranslated in the output -- a visible quality defect
# traded for no safety gain. So on that path the ratio only rejects when
# the input is long enough for the ratio to mean anything, or when the
# output is absurdly long in absolute terms (which is what a leaked
# ~2k-char system prompt looks like).
MIN_INPUT_TOKENS_FOR_RATIO_CHECK = 10
ABSURD_OUTPUT_TOKENS = 64

PROMPT_LEAK_REASON = (
    "Translation result echoes a system-prompt marker, possible "
    "injection attempt or leak, fallback."
)


def is_token_ratio_out_of_band(
    input_token_count: int,
    output_token_count: int,
) -> bool:
    """Return True when the output/input token ratio is implausible.

    A zero input token count means the tokenizer failed (``calc_token_count``
    swallows exceptions and returns 0); in that case there is nothing to
    compare against, so the check abstains rather than dividing by zero.
    """
    if input_token_count <= 0:
        return False
    ratio = output_token_count / input_token_count
    return not (TOKEN_RATIO_MIN < ratio < TOKEN_RATIO_MAX)


def reject_reason_for_translation(
    input_text: str,
    output_text: str,
    calc_token_count: Callable[[str], int],
) -> str | None:
    """Return a human-readable rejection reason, or None if output is usable.

    Only covers the checks that are meaningful on *both* PDF translation
    paths. The batch translator layers its own same-text and
    edit-distance heuristics on top of this.
    """
    if not isinstance(output_text, str) or not output_text:
        return "Translation result is empty."

    if looks_like_prompt_leak(output_text):
        return PROMPT_LEAK_REASON

    input_token_count = calc_token_count(input_text)
    output_token_count = calc_token_count(output_text)
    ratio_is_meaningful = (
        input_token_count >= MIN_INPUT_TOKENS_FOR_RATIO_CHECK
        or output_token_count >= ABSURD_OUTPUT_TOKENS
    )
    if ratio_is_meaningful and is_token_ratio_out_of_band(
        input_token_count, output_token_count
    ):
        return (
            "Translation result is too long or too short. "
            f"Input: {input_token_count}, Output: {output_token_count}"
        )

    return None
