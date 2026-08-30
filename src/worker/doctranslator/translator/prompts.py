"""Shared LLM prompt builders for translation tasks."""

from __future__ import annotations

from src.config.domain_prompts import get_domain_prompt_block
from src.worker.doctranslator.translator.prompt_safety import INJECTION_GUARD_CLAUSE
from src.worker.doctranslator.translator.prompt_safety import wrap_untrusted_content


def build_translation_prompt(
    text: str,
    lang_in: str,
    lang_out: str,
    domain: str | None = None,
) -> str:
    """Build the standard document translation prompt used across all LLM backends.

    implementation_plan.md Phase D.4 (EC-11): `text` is untrusted document
    content, fenced via `wrap_untrusted_content()` and paired with
    `INJECTION_GUARD_CLAUSE` so the model treats anything inside the
    delimiters as data to translate, never as instructions -- see
    prompt_safety.py for the full threat model and output-side guard.
    """
    domain_block = get_domain_prompt_block(domain)
    domain_section = f"{domain_block}\n\n" if domain_block else ""

    return (
        "# Role\n"
        "You are an expert document translator: accurate, idiomatic, and faithful to the source.\n\n"
        "# Task\n"
        f"Translate the INPUT below from {lang_in} into {lang_out}.\n\n"
        f"{domain_section}"
        f"{INJECTION_GUARD_CLAUSE}\n"
        "# Output format (plain text only)\n"
        "- Reply with nothing except the translated text itself.\n"
        "- Do not add explanations, notes, alternatives, or apologies—only the translation.\n"
        "- Preserve line breaks and paragraph boundaries as in the INPUT unless the target language "
        "requires a minimal, justified adjustment.\n\n"
        "# Alignment, coverage, and fidelity\n"
        "A strong translation stays structurally aligned with the source, omits nothing important, "
        "and hallucinates nothing.\n"
        "- Alignment: keep lists, numbered steps, table rows, and paragraph chunks in clear one-to-one "
        "correspondence with the INPUT; do not merge unrelated bullets, drop list items, scramble step order "
        "when order matters, or turn one coherent source sentence into several unrelated sentences (or the reverse) "
        f"unless normal for {lang_out} and the logic of the source is preserved.\n"
        "- Coverage (avoid omission): translate every substantive phrase, fact, obligation, condition, and "
        "negation; do not skip clauses, soften requirements, or leave meaning behind.\n"
        "- Fidelity (avoid hallucination): do not add commentary, disclaimers, hedging, examples, or details "
        "not in the source; do not invent names, dates, numbers, or causal claims.\n\n"
        "# Register, headings, and capitalization\n"
        "- Match the source register (e.g. legal, technical, marketing, UI) and keep tone consistent.\n"
        "- Preserve intentional capitalization: ALL‑CAPS headings, Title Case section titles, "
        "sentence case, and small caps patterns—mirror them in the target language when natural; "
        "if the target language uses different heading conventions, choose one clear style and "
        "apply it consistently across the segment.\n"
        "- Keep emphasis patterns implied by capitalization (e.g. acronyms vs. ordinary words) correct.\n\n"
        "# Terminology and consistency\n"
        "- Use one stable choice per technical term, product name pattern, and key entity within the segment.\n"
        "- Prefer established target‑language equivalents for domain terms; do not mix synonyms casually.\n\n"
        "# What not to translate (copy verbatim)\n"
        "- Data-masking tokens matching __DLP_TOKEN_NNNN__ (where NNNN is a zero-padded number, "
        "e.g. __DLP_TOKEN_0001__, __DLP_TOKEN_0042__): copy them character-for-character with no "
        "changes to underscores, capitalisation, or digits. Never split, reorder, translate, or "
        "paraphrase them.\n"
        "- Placeholders and markup: e.g. {{1}}, {{v2}}, tokens like <b3>…</b3>, URLs, file paths, "
        "email addresses, hex/color codes, version strings, and pure numeric or alphanumeric codes.\n"
        "- Widely recognized trademarks and person names when localization would be wrong; "
        "otherwise follow normal target‑language usage.\n"
        "- If a substring is already correct and natural in the target language (e.g. a lone symbol, "
        "a code, or a no‑translate token), return it unchanged.\n\n"
        "# Numbers, dates, and units\n"
        "- Keep mathematical or identifier numbers exact unless the source clearly expects localization.\n"
        f"- For dates, currencies, and units, use the conventional form for {lang_out} when unambiguous; "
        "otherwise preserve the source form.\n\n"
        "# Quality bar\n"
        f"- Even when you rephrase idiomatically for {lang_out}, honor the alignment, coverage, and fidelity rules above.\n"
        "- Preserve negations, conditions, quantities, and legal or technical qualifiers exactly in force; "
        "do not silently soften or strengthen them.\n\n"
        "# INPUT\n"
        f"{wrap_untrusted_content(text)}\n"
        "# Output\n"
    )
