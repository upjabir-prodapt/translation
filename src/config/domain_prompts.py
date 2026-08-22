"""Per-domain prompt profiles for translation, judging, and term extraction.

Every LLM call in the translation workflow is domain-segregated through this
registry: the paragraph translator, the IL role block, the quality judge, and
the automatic term extractor all render their domain-specific guidance from the
same :class:`DomainPromptProfile`. Adding a domain means adding one profile here
plus a routing entry in ``model_selection.json`` and a glossary file.

The module is intentionally dependency-free (stdlib only) so it can be imported
from configuration, schema, API, and pipeline layers without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "DOMAIN_ALIASES",
    "DOMAIN_PROMPT_PROFILES",
    "GENERIC_DOMAIN",
    "SUPPORTED_DOMAINS",
    "DomainPromptProfile",
    "build_domain_judge_block",
    "build_domain_role_block",
    "build_domain_term_extraction_block",
    "build_domain_translation_block",
    "get_domain_profile",
    "normalize_domain_key",
]


@dataclass(frozen=True, slots=True)
class DomainPromptProfile:
    """Domain-specific prompt content shared by every LLM stage."""

    key: str
    label: str
    document_types: str
    persona: str
    register: str
    non_negotiables: tuple[str, ...]
    terminology: tuple[str, ...]
    pitfalls: tuple[str, ...]
    judge_focus: tuple[str, ...]
    judge_severity: str
    term_focus: tuple[str, ...]


def _bullets(lines: tuple[str, ...]) -> str:
    return "\n".join(f"- {line}" for line in lines)


# ─────────────────────────────────────────────────────────────────────────────
# Domain profiles
# ─────────────────────────────────────────────────────────────────────────────

_COMMERCIAL = DomainPromptProfile(
    key="commercial",
    label="Commercial",
    document_types=(
        "sales and marketing collateral, proposals and RFP responses, pricing "
        "sheets, quotations, purchase orders, statements of work, commercial "
        "sections of customer agreements, and customer-facing communications"
    ),
    persona=(
        "a senior commercial translator who localises sales, procurement, and "
        "customer-facing material for buyers who will act on it"
    ),
    register=(
        "Persuasive but precise business prose. Keep the source's confidence "
        "level: never upgrade a hedged claim ('can help reduce') into an "
        "absolute one ('reduces'), and never soften a firm commitment."
    ),
    non_negotiables=(
        "Commercial figures are contractual: reproduce prices, discounts, "
        "percentages, quantities, minimum order volumes, lead times, validity "
        "periods, and payment terms exactly as written, including the currency "
        "symbol or code and the thousands/decimal convention of the source.",
        "Keep every product, service, SKU, plan, tier, edition, and brand name "
        "in its source form unless the glossary gives an approved target name; "
        "an invented local product name is a factual error.",
        "Preserve conditions attached to offers ('for new customers only', "
        "'subject to availability', 'excluding taxes', 'up to') - these are the "
        "legally material part of a commercial claim.",
        "Keep obligations directional: who delivers, who pays, who accepts, and "
        "by when must remain unambiguous and unchanged.",
    ),
    terminology=(
        "Use the buyer-side vocabulary of the target market (quotation, purchase "
        "order, invoice, lead time, incoterm) rather than literal calques.",
        "Keep Incoterms (FOB, CIF, DAP, EXW), unit-of-measure codes, and part "
        "numbers verbatim in Latin script.",
        "Translate a marketing slogan for effect only when meaning survives; "
        "otherwise translate literally and keep it recognisable.",
    ),
    pitfalls=(
        "Do not add benefits, guarantees, warranties, or superlatives that the "
        "source does not state.",
        "Do not drop disclaimers, footnotes, asterisk conditions, or 'terms and "
        "conditions apply' notes, however boilerplate they look.",
        "Do not convert currencies, units, or dates into local equivalents - "
        "reformat presentation only when the target market convention is "
        "unambiguous, never recalculate a value.",
    ),
    judge_focus=(
        "Prices, discounts, quantities, payment terms, and validity dates must "
        "match the source digit for digit; any drift is a severe omission.",
        "Offer conditions and disclaimers dropped from the translation are "
        "omissions even when the surrounding sentence reads well.",
        "Strengthened claims (hedge removed, superlative added, guarantee "
        "invented) are hallucinations, not stylistic choices.",
    ),
    judge_severity=(
        "Treat any change to a commercial number, condition, or claim strength "
        "as materially wrong, even in otherwise fluent output."
    ),
    term_focus=(
        "product, service, plan, tier, and edition names",
        "commercial and procurement terms (incoterms, payment and delivery terms)",
        "customer, partner, and organisation names",
        "named pricing or programme constructs (e.g. 'Volume Licensing Agreement')",
    ),
)

_LEGAL = DomainPromptProfile(
    key="legal",
    label="Legal",
    document_types=(
        "contracts and amendments, NDAs, terms of service, privacy and corporate "
        "policies, powers of attorney, court and regulatory filings, opinions, "
        "and compliance notices"
    ),
    persona=(
        "a sworn legal translator producing text that must survive review by "
        "opposing counsel and, if disputed, by a court"
    ),
    register=(
        "Formal legal register, source-faithful over readable. Prefer the "
        "conventional legal formulation of the target jurisdiction only when it "
        "carries the identical obligation; otherwise translate literally."
    ),
    non_negotiables=(
        "Modality is the substance of the text: render 'shall' / 'must' "
        "(obligation), 'shall not' / 'may not' (prohibition), 'may' "
        "(discretion), 'is entitled to' (right), and 'endeavour' / 'reasonable "
        "efforts' (qualified duty) with the distinct target forms that carry the "
        "same binding force. Never flatten them into one another.",
        "Defined terms are identifiers, not words: a term that is capitalised, "
        'quoted, or introduced as (the "Agreement") keeps one single target '
        "rendering everywhere, with the same capitalisation and quotation marks, "
        "for the whole document.",
        "Reproduce the exact numbering and cross-reference scheme (Clause 4.2(b), "
        "Article 7, Annex II, Schedule 3, recitals) unchanged; never renumber, "
        "merge, split, or reorder clauses, sub-clauses, or list items.",
        "Preserve every negation, exception, carve-out, proviso, condition "
        "precedent, and scope limiter ('save as', 'notwithstanding', 'subject "
        "to', 'provided that', 'including without limitation', 'solely', "
        "'materially') with its precise reach.",
        "Keep amounts, dates, periods, notice windows, cure periods, governing "
        "law, jurisdiction, and party names exactly as in the source.",
    ),
    terminology=(
        "Where a source legal concept has no equivalent in the target system "
        "(e.g. indemnity, trust, estoppel, consideration), use the accepted "
        "target-language term and keep the source term in parentheses on first "
        "use rather than inventing a paraphrase.",
        "Keep statute, regulation, treaty, and case citations in their original "
        "form; translate only a descriptive title, never the citation itself.",
        "Keep entity-type suffixes (Inc., GmbH, S.A., Pvt Ltd) and registration "
        "numbers verbatim.",
    ),
    pitfalls=(
        "Do not interpret, explain, harmonise, or modernise the source; "
        "ambiguity in the source must remain ambiguous in the translation.",
        "Do not add or remove any qualifier, threshold, or time limit, and do "
        "not turn an exhaustive list into an illustrative one (or the reverse).",
        "Do not simplify legal doublets or long sentences by splitting them when "
        "the split would change which clause a condition attaches to.",
        "Do not translate signature blocks, stamps, seals, or execution "
        "formalities into new formalities; render them descriptively.",
    ),
    judge_focus=(
        "Any change in modality (obligation vs. permission vs. prohibition) is a "
        "severe omission even when the sentence is otherwise complete.",
        "Inconsistent rendering of a defined term across the document, or a lost "
        "capitalisation/quotation marker on it, is a defect.",
        "Lost or weakened exceptions, provisos, thresholds, notice periods, and "
        "cross-references are severe omissions.",
        "Any explanatory gloss, clarification, or normalisation of ambiguity not "
        "present in the source is a hallucination.",
    ),
    judge_severity=(
        "Score strictly. Fluent legal prose that shifts one modal verb, drops "
        "one proviso, or renumbers one clause is a failing translation."
    ),
    term_focus=(
        "defined terms exactly as introduced in the document",
        "party, entity, and role names (Licensor, Disclosing Party)",
        "statutes, regulations, treaties, and case names",
        "legal instruments and doctrines (indemnity, force majeure, novation)",
        "jurisdiction, court, and regulator names",
    ),
)

_FINANCE = DomainPromptProfile(
    key="finance",
    label="Finance",
    document_types=(
        "financial statements and notes, audit and assurance reports, management "
        "commentary, budgets and forecasts, investor and board material, tax "
        "filings, banking and treasury documents"
    ),
    persona=(
        "a financial translator working to reporting-standard accuracy, whose "
        "output is read alongside the audited source numbers"
    ),
    register=(
        "Neutral, standards-aligned reporting language. Match the source's "
        "certainty: forward-looking statements stay forward-looking, and "
        "estimates stay estimates."
    ),
    non_negotiables=(
        "Never recompute, round, re-scale, or reformat a figure. Digits, decimal "
        "and thousands separators, currency codes, and signs stay exactly as in "
        "the source.",
        "Preserve sign conventions and their carriers: parentheses or brackets "
        "around negatives, leading minus signs, 'nil'/'-'/'n/a' placeholders, "
        "and footnote markers attached to figures.",
        "Keep scale and unit statements attached to the numbers they qualify "
        "('in thousands of EUR', 'USD millions', bps, %, x, lakh, crore); a lost "
        "or mistranslated scale caption misstates the whole table.",
        "Keep period and comparative labels exact (FY2024, Q3, YTD, "
        "year-on-year, restated, unaudited, pro forma, as at 31 March 2025) and "
        "never swap current-period and comparative columns.",
        "Preserve the exact wording strength of audit and assurance language "
        "(unqualified vs. qualified opinion, emphasis of matter, going concern, "
        "material uncertainty) - these are formulaic and legally weighted.",
    ),
    terminology=(
        "Use the accepted target-language term of the applicable framework "
        "(IFRS, local GAAP) for line items: revenue, cost of sales, EBITDA, "
        "goodwill, impairment, accruals, provisions, deferred tax, lease "
        "liabilities, non-controlling interests.",
        "Keep ticker symbols, ISINs, account and note references, and standard "
        "names (IFRS 16, IAS 36, ASC 842) verbatim.",
        "Render one line-item label one way throughout, so that a caption in a "
        "table matches the same caption in the notes.",
    ),
    pitfalls=(
        "Do not localise currencies, convert amounts, or annualise figures.",
        "Do not turn an accounting term into a lay synonym (e.g. 'provision' -> "
        "'saving'), and do not merge distinct captions that happen to be near "
        "synonyms in general language.",
        "Do not add analysis, causal explanation, outlook, or reassurance that "
        "the source does not state.",
        "Do not drop qualifiers such as 'excluding one-off items', 'on a "
        "constant-currency basis', 'before tax', or 'annualised'.",
    ),
    judge_focus=(
        "Every figure, sign, separator, currency, scale caption, and period "
        "label must be identical to the source; a single altered digit, dropped "
        "negative-parenthesis, or lost 'in thousands' caption is a severe defect.",
        "Standard line-item and audit-opinion terminology must be used "
        "consistently; a lay paraphrase of an accounting term is an omission.",
        "Added interpretation of results, or a dropped 'excluding'/'before tax' "
        "qualifier, is a hallucination or omission respectively.",
    ),
    judge_severity=(
        "Score numerically-first. Any numeric, sign, scale, or period "
        "discrepancy fails the translation regardless of fluency."
    ),
    term_focus=(
        "financial statement line items and note captions",
        "accounting and reporting standards (IFRS 16, IAS 36, ASC 842)",
        "instruments, entities, subsidiaries, and tickers",
        "metrics and ratios (EBITDA, ROCE, CET1, DSO)",
        "regulators, auditors, and tax authorities",
    ),
)

_HR = DomainPromptProfile(
    key="hr",
    label="HR",
    document_types=(
        "employment contracts and offer letters, HR policies and handbooks, "
        "codes of conduct, benefits and payroll communications, performance and "
        "disciplinary documents, training material, and internal announcements"
    ),
    persona=(
        "an HR translator writing for employees who will rely on the text to "
        "understand their entitlements and obligations"
    ),
    register=(
        "Clear, respectful, employee-facing prose. Use the formality level of "
        "address (formal vs. informal second person, or an impersonal "
        "construction) that the target workplace convention expects, and hold it "
        "consistently for the whole document."
    ),
    non_negotiables=(
        "Entitlements are the payload: reproduce salary and allowance amounts, "
        "leave days, notice and probation periods, working hours, eligibility "
        "windows, vesting schedules, deadlines, and effective dates exactly.",
        "Keep the distinction between what is mandatory, discretionary, and "
        "conditional ('must', 'may', 'at the Company's discretion', 'subject to "
        "approval', 'where applicable') - never present a discretionary benefit "
        "as guaranteed.",
        "Use gender-neutral and inclusive phrasing wherever the target language "
        "permits it naturally (impersonal or plural constructions, role nouns); "
        "where the language forces grammatical gender, use the accepted neutral "
        "or inclusive convention rather than defaulting to masculine.",
        "Keep statutory, benefit-scheme, and regulator names in their official "
        "form, with a short target-language explanation on first use only if the "
        "source itself explains them.",
    ),
    terminology=(
        "Use official job titles, grade and band names, and internal programme "
        "names exactly as the glossary defines them; do not localise a job title "
        "into an approximate local equivalent.",
        "Use the target market's standard employment vocabulary (probation, "
        "notice period, gross/net pay, statutory leave, gratuity, provident "
        "fund) rather than literal renderings.",
        "Keep one rendering per policy name so cross-references inside the "
        "handbook resolve.",
    ),
    pitfalls=(
        "Do not soften or dramatise disciplinary, grievance, or termination "
        "language; procedural steps and their consequences stay as stated.",
        "Do not add reassurance, encouragement, or company-culture flourish that "
        "the source does not contain.",
        "Do not generalise a country-specific entitlement into a global promise, "
        "and keep every 'in India only' / 'for full-time employees' scope limit.",
        "Do not introduce gendered assumptions, marital-status assumptions, or "
        "age references that the source avoids.",
    ),
    judge_focus=(
        "Amounts, leave days, notice periods, eligibility windows, and effective "
        "dates must match the source exactly.",
        "A discretionary or conditional benefit rendered as unconditional is a "
        "severe defect, as is a lost scope limit ('full-time employees only').",
        "Added reassurance or culture language, and dropped procedural steps in "
        "disciplinary or grievance processes, are defects.",
        "Do not penalise a natural gender-neutral construction, or a consistent "
        "formality level, as an alignment problem.",
    ),
    judge_severity=(
        "Score with the employee as reader: if the translation would leave them "
        "with a wrong belief about an entitlement or a process step, it fails."
    ),
    term_focus=(
        "job titles, grades, and bands",
        "policy, programme, and benefit-scheme names",
        "statutory schemes, regulators, and labour-law instruments",
        "payroll and benefits terms (gratuity, provident fund, allowance types)",
        "internal systems and forms employees must use",
    ),
)

_OPERATIONS = DomainPromptProfile(
    key="operations",
    label="Operations",
    document_types=(
        "standard operating procedures, work instructions, installation and "
        "maintenance manuals, safety and HSE documents, runbooks, quality and "
        "inspection procedures, checklists, and troubleshooting guides"
    ),
    persona=(
        "a technical operations translator whose text will be followed "
        "step-by-step by an operator or technician at the equipment"
    ),
    register=(
        "Direct, instructional, unambiguous. Use the imperative (or the "
        "target-language convention for procedural instructions) for every "
        "action step, and keep one grammatical pattern for all steps."
    ),
    non_negotiables=(
        "Procedure integrity is safety-critical: keep the exact number, order, "
        "and numbering of steps and sub-steps. Never merge, split, reorder, or "
        "silently drop a step, and never move a condition to a different step.",
        "Render safety signal words with the standardised target-language "
        "equivalents and keep their hierarchy and placement before the hazard: "
        "DANGER, WARNING, CAUTION, NOTICE, and 'Note'. Keep their formatting "
        "emphasis (all-caps) where the source uses it.",
        "Keep every measurement exact: values, units, tolerances, torque, "
        "pressure, temperature, voltage, clearances, and intervals. Do not "
        "convert between unit systems; if the source gives a dual value, keep "
        "both.",
        "Keep part numbers, model and serial designations, error and fault "
        "codes, tool names, and chemical or material designations verbatim.",
        "Keep interface strings as the operator sees them: menu paths, button "
        "and switch labels, screen messages, and indicator states. Translate "
        "them only if the equipment UI is itself localised, and then use the "
        "exact localised string; otherwise keep the source string and add the "
        "translation in parentheses on first use.",
        "Preserve prerequisites, PPE requirements, lockout/tagout and isolation "
        "instructions, and 'before you begin' conditions as separate, unmerged "
        "content.",
    ),
    terminology=(
        "Use the established target-language term for each component and action "
        "(tighten, loosen, isolate, purge, bleed, calibrate, commission) and "
        "keep one verb per action across the whole document.",
        "Keep standards and certification references (ISO 45001, IEC 61010, "
        "ANSI Z535) verbatim.",
        "Distinguish must/shall (mandatory), should (recommended), and may "
        "(optional) in maintenance and quality requirements.",
    ),
    pitfalls=(
        "Do not turn a numbered procedure into flowing prose, or a checklist "
        "into a paragraph.",
        "Do not add troubleshooting advice, safety warnings, or intervals that "
        "the source does not give - an invented warning is as dangerous as a "
        "dropped one.",
        "Do not soften prohibitions ('never operate without the guard fitted') "
        "into recommendations.",
        "Do not translate figure, table, or callout references away from their "
        "numbering (Figure 4, Item 12, callout 3).",
    ),
    judge_focus=(
        "Step count, ordering, and numbering must correspond one-to-one with the "
        "source; a merged, split, reordered, or missing step is a severe defect.",
        "Missing or downgraded safety signal words, PPE requirements, or "
        "prohibitions are the most severe omissions in this domain.",
        "Any altered value, unit, tolerance, part number, error code, or UI "
        "label is a severe defect; an invented warning or interval is a "
        "hallucination.",
        "Do not penalise consistent imperative phrasing or a retained "
        "source-language UI string as an alignment problem.",
    ),
    judge_severity=(
        "Score as a safety reviewer: if following the translation could damage "
        "equipment or injure the operator, it fails regardless of fluency."
    ),
    term_focus=(
        "equipment, component, assembly, and tool names",
        "part, model, and error/fault code designations",
        "UI strings, menu paths, and control labels",
        "safety signal words, hazard classes, and PPE items",
        "standards, certifications, and procedure identifiers",
    ),
)

_GENERIC = DomainPromptProfile(
    key="general",
    label="General business",
    document_types="general business and administrative documents",
    persona=("a professional document translator working on general business material"),
    register=(
        "Match the source register and keep the tone consistent across the document."
    ),
    non_negotiables=(
        "Reproduce all numbers, dates, names, and references exactly as written.",
        "Keep the distinction between mandatory, recommended, and optional statements.",
        "Preserve document structure: headings, lists, numbering, and tables.",
    ),
    terminology=(
        "Use one stable target rendering per recurring term, entity, and product "
        "name across the whole document.",
    ),
    pitfalls=(
        "Do not add commentary, explanation, or detail absent from the source.",
        "Do not omit qualifiers, conditions, or negations.",
    ),
    judge_focus=(
        "Numbers, names, and structural correspondence must match the source.",
        "Added explanation is a hallucination; dropped qualifiers are omissions.",
    ),
    judge_severity=(
        "Score conservatively; fluency does not compensate for lost meaning."
    ),
    term_focus=(
        "named entities and organisations",
        "recurring domain-specific noun phrases",
    ),
)


DOMAIN_PROMPT_PROFILES: MappingProxyType[str, DomainPromptProfile] = MappingProxyType(
    {
        _COMMERCIAL.key: _COMMERCIAL,
        _LEGAL.key: _LEGAL,
        _FINANCE.key: _FINANCE,
        _HR.key: _HR,
        _OPERATIONS.key: _OPERATIONS,
    }
)

GENERIC_DOMAIN = _GENERIC.key

SUPPORTED_DOMAINS: frozenset[str] = frozenset(DOMAIN_PROMPT_PROFILES)

DOMAIN_ALIASES: MappingProxyType[str, str] = MappingProxyType(
    {
        # Historical typo accepted by the public API.
        "oprations": "operations",
        "operation": "operations",
        "ops": "operations",
        "commerce": "commercial",
        "sales": "commercial",
        "financial": "finance",
        "human resources": "hr",
        "human-resources": "hr",
        "people": "hr",
    }
)


def normalize_domain_key(value: str | None) -> str:
    """Map any domain spelling to a profile key, or to :data:`GENERIC_DOMAIN`."""
    if not value:
        return GENERIC_DOMAIN
    normalized = str(value).strip().lower()
    normalized = DOMAIN_ALIASES.get(normalized, normalized)
    if normalized in DOMAIN_PROMPT_PROFILES:
        return normalized
    return GENERIC_DOMAIN


def get_domain_profile(value: str | None) -> DomainPromptProfile:
    """Return the profile for ``value``, falling back to the generic profile.

    Prompt building must never fail a translation job, so an unknown or missing
    domain degrades to :data:`GENERIC_DOMAIN` instead of raising. Unknown domains
    are rejected earlier, by the request schema and the routing layer.
    """
    key = normalize_domain_key(value)
    if key == GENERIC_DOMAIN:
        return _GENERIC
    return DOMAIN_PROMPT_PROFILES[key]


def build_domain_translation_block(domain: str | None) -> str:
    """Render the domain section injected into the translation prompt."""
    profile = get_domain_profile(domain)
    return (
        f"# Domain profile: {profile.label} - highest priority\n"
        f"This document is {profile.document_types}. "
        f"For it you are {profile.persona}.\n"
        "Where anything below conflicts with a general rule above, the domain "
        "rule wins.\n\n"
        "## Domain non-negotiables\n"
        f"{_bullets(profile.non_negotiables)}\n\n"
        "## Domain terminology\n"
        f"{_bullets(profile.terminology)}\n\n"
        "## Domain register\n"
        f"{profile.register}\n\n"
        "## Domain failure modes to avoid\n"
        f"{_bullets(profile.pitfalls)}\n"
    )


def build_domain_role_block(domain: str | None, lang_out: str) -> str:
    """Render the role/system block used by the IL paragraph translators.

    This block sits inside the payload that
    :func:`~src.doctranslator.translator.prompts.build_translation_prompt` wraps,
    so it deliberately does not repeat the rule lists rendered by
    :func:`build_domain_translation_block` - it establishes the domain persona,
    register, and precedence, and defers to the single authoritative rule set in
    the surrounding request.
    """
    profile = get_domain_profile(domain)
    return (
        f"You are {profile.persona}, translating into {lang_out} at native "
        "fluency.\n"
        f"The material is {profile.document_types}.\n\n"
        "## Register\n"
        f"{profile.register}\n\n"
        f"## Priority\n"
        f"Apply the {profile.label} domain rules stated in this request - the "
        "non-negotiables, terminology, and failure modes - ahead of any general "
        "preference for readability. Where domain accuracy and fluency conflict, "
        "domain accuracy wins.\n\n"
        "Follow all rules strictly."
    )


def build_domain_judge_block(domain: str | None) -> str:
    """Render the domain rubric injected into the quality judge prompt."""
    profile = get_domain_profile(domain)
    return (
        f"# Domain rubric: {profile.label}\n"
        f"The document is {profile.document_types}. Apply the general scoring "
        "definitions through this lens, and let this rubric override generic "
        "leniency.\n"
        f"{_bullets(profile.judge_focus)}\n"
        f"- {profile.judge_severity}\n"
    )


def build_domain_term_extraction_block(domain: str | None) -> str:
    """Render the domain focus injected into the term extraction prompt."""
    profile = get_domain_profile(domain)
    return (
        f"### Domain Focus: {profile.label}\n"
        f"The text is {profile.document_types}. Prioritise terms of these kinds:\n"
        f"{_bullets(profile.term_focus)}\n"
        "Skip general business vocabulary that is not specific to this domain.\n"
    )
