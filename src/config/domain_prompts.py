"""Domain-specific translation prompt profiles for formality, tone, and terminology."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.translation_routing import normalize_domain


@dataclass(frozen=True, slots=True)
class DomainPromptProfile:
    """Prompt instructions tailored to a specific business domain."""

    domain: str
    display_name: str
    formality: str
    tone: str
    register_guidelines: tuple[str, ...]
    terminology_guidelines: tuple[str, ...]

    def render_prompt_block(self) -> str:
        """Render markdown section for LLM prompt injection."""
        lines: list[str] = [
            f"## Domain-Specific Guidance ({self.display_name} Domain)",
            f"- Formality: {self.formality}",
            f"- Tone & Style: {self.tone}",
            "- Register & Conventions:",
        ]
        for guideline in self.register_guidelines:
            lines.append(f"  * {guideline}")
        lines.append("- Terminology & Phrasing:")
        for guideline in self.terminology_guidelines:
            lines.append(f"  * {guideline}")
        return "\n".join(lines)


DOMAIN_PROMPT_PROFILES: dict[str, DomainPromptProfile] = {
    "commercial": DomainPromptProfile(
        domain="commercial",
        display_name="Commercial & Business",
        formality="Professional, polished, and business-appropriate.",
        tone="Engaging, persuasive, confident, and customer-oriented without exaggeration or hyperbole.",
        register_guidelines=(
            "Maintain an active, client-facing voice suitable for proposals, marketing materials, and commercial agreements.",
            "Ensure calls-to-action (CTAs) and value propositions remain crisp, compelling, and natural in the target language.",
            "Adapt idioms and metaphors to culturally appropriate business equivalents rather than literal translations.",
        ),
        terminology_guidelines=(
            "Preserve product names, brand trademarks, and slogans verbatim unless an official target-language localization exists.",
            "Use standard corporate and commercial terminology (e.g., value proposition, market share, stakeholder, deliverables).",
            "Keep pricing models, discount terms, and service tiers clear and unambiguous.",
        ),
    ),
    "legal": DomainPromptProfile(
        domain="legal",
        display_name="Legal & Regulatory",
        formality="Strictly formal, binding, and legally rigorous.",
        tone="Objective, unambiguous, authoritative, and neutral.",
        register_guidelines=(
            "Preserve precise legal meaning and enforceability; avoid colloquial phrasing or stylistic embellishment.",
            "Translate modal verbs with absolute precision: distinguish mandatory obligations ('shall', 'must'), discretionary rights ('may'), and conditions precedent ('provided that').",
            "Maintain sentence structure and clause relationships faithfully to prevent legal ambiguity or altered liability scope.",
        ),
        terminology_guidelines=(
            "Adhere strictly to standard statutory and contractual terms of art (e.g., force majeure, indemnification, jurisdiction, severance, governing law).",
            "Never paraphrase, substitute synonyms, or simplify defined terms (capitalized terms of art).",
            "Retain international legal acronyms (e.g., GDPR, NDA, SLA, ISDA, ICC) in standard format.",
        ),
    ),
    "finance": DomainPromptProfile(
        domain="finance",
        display_name="Finance & Accounting",
        formality="Formal, analytical, and structured.",
        tone="Precise, factual, objective, and conservative.",
        register_guidelines=(
            "Ensure mathematical and tabular clarity; numbers, financial formulas, and accounting relations must remain exact.",
            "Do not round, approximate, or rephrase monetary amounts, percentage rates, basis points, or fiscal periods.",
            "Maintain standard financial reporting conventions (e.g., balance sheet captions, cash flow line items).",
        ),
        terminology_guidelines=(
            "Use authoritative accounting and financial terminology aligned with IFRS/GAAP standards (e.g., EBITDA, receivables, amortization, asset depreciation, liquidity).",
            "Preserve currency codes, symbols, and formatting according to target-language financial conventions (e.g., EUR 1,000.00 vs 1.000,00 €).",
            "Keep fiscal metrics, ratios, and audit terms consistent across all sections.",
        ),
    ),
    "hr": DomainPromptProfile(
        domain="hr",
        display_name="Human Resources & People",
        formality="Professional, respectful, accessible, and compliant.",
        tone="Empathetic, clear, constructive, and people-centric while remaining firm on policy and organizational standards.",
        register_guidelines=(
            "Use inclusive, gender-neutral, and culturally sensitive language appropriate to workplace communications.",
            "Handle sensitive personnel matters (e.g., disciplinary procedures, grievances, redundancy, termination) with dignity, clarity, and legal precision.",
            "Ensure employee handbooks, performance evaluations, and internal policies sound supportive and transparent.",
        ),
        terminology_guidelines=(
            "Use jurisdiction-appropriate labor and HR terminology (e.g., notice period, performance review, maternity/paternity leave, collective bargaining, severance).",
            "Preserve organizational acronyms and job title conventions (e.g., HRIS, KPI, OKR, FTE).",
            "Maintain consistent terminology for employee benefits, compensation tiers, and workplace safety protocols.",
        ),
    ),
    "operations": DomainPromptProfile(
        domain="operations",
        display_name="Operations & Technical",
        formality="Standard professional, direct, and technical.",
        tone="Instructional, unambiguous, concise, and action-oriented.",
        register_guidelines=(
            "Use clear imperative or infinitive verb forms for step-by-step procedures, runbooks, and Standard Operating Procedures (SOPs).",
            "Preserve logical workflows, conditional steps (e.g., 'If X fails, execute Y'), and chronological order without ambiguity.",
            "Avoid wordiness; prioritize clarity, brevity, and actionable technical instructions.",
        ),
        terminology_guidelines=(
            "Use exact technical, telecommunications, networking, and IT operations terminology (e.g., latency, SLA, failover, throughput, provisioning, incident management).",
            "Preserve system commands, parameter names, error codes, and configuration keys verbatim.",
            "Maintain consistent naming for infrastructure components, hardware units, and operational milestones.",
        ),
    ),
}


def get_domain_prompt_profile(domain: str | None) -> DomainPromptProfile | None:
    """Retrieve the DomainPromptProfile for a domain name, or None if unknown/empty."""
    if not domain or not str(domain).strip():
        return None
    try:
        normalized = normalize_domain(domain)
        return DOMAIN_PROMPT_PROFILES.get(normalized)
    except ValueError:
        return None


def get_domain_prompt_block(domain: str | None) -> str:
    """Render a domain guidance block for prompt injection, or empty string if domain is None/unknown."""
    profile = get_domain_prompt_profile(domain)
    if profile is None:
        return ""
    return profile.render_prompt_block()


def get_domain_role_block(
    domain: str | None,
    lang_out: str,
    custom_system_prompt: str | None = None,
) -> str:
    """Build the role/system block for LLM prompts, incorporating domain guidance.

    If custom_system_prompt is provided, it is honored as the primary role instruction,
    with domain-specific guidance appended if available.
    """
    domain_block = get_domain_prompt_block(domain)

    if custom_system_prompt and custom_system_prompt.strip():
        role_base = custom_system_prompt.strip()
        if "Follow all rules strictly." not in role_base:
            if not role_base.endswith("\n"):
                role_base += "\n"
            role_base += "Follow all rules strictly."
    else:
        profile = get_domain_prompt_profile(domain)
        domain_desc = f" in {profile.display_name}" if profile else ""
        role_base = (
            f"You are a professional {lang_out} native translator who specializes{domain_desc} "
            f"and fluently translates text into {lang_out}.\n\n"
            "Follow all rules strictly."
        )

    if domain_block:
        return f"{role_base}\n\n{domain_block}"
    return role_base
