"""Unit tests for domain-specific translation prompt profiles."""

import pytest
from src.config.domain_prompts import DOMAIN_PROMPT_PROFILES
from src.config.domain_prompts import get_domain_prompt_block
from src.config.domain_prompts import get_domain_prompt_profile
from src.config.domain_prompts import get_domain_role_block


class TestDomainPromptProfiles:
    @pytest.mark.parametrize(
        "domain,expected_label",
        [
            ("commercial", "Commercial & Business"),
            ("legal", "Legal & Regulatory"),
            ("finance", "Finance & Accounting"),
            ("hr", "Human Resources & People"),
            ("operations", "Operations & Technical"),
        ],
    )
    def test_all_five_domains_configured(self, domain, expected_label):
        profile = get_domain_prompt_profile(domain)
        assert profile is not None
        assert profile.domain == domain
        assert profile.display_name == expected_label
        assert len(profile.formality) > 0
        assert len(profile.tone) > 0
        assert len(profile.register_guidelines) >= 3
        assert len(profile.terminology_guidelines) >= 3

    def test_domain_case_and_whitespace_insensitivity(self):
        assert get_domain_prompt_profile("  LEGAL ") == DOMAIN_PROMPT_PROFILES["legal"]
        assert get_domain_prompt_profile("Commercial") == DOMAIN_PROMPT_PROFILES["commercial"]
        assert get_domain_prompt_profile("FiNaNcE") == DOMAIN_PROMPT_PROFILES["finance"]

    def test_unknown_or_empty_domain_returns_none(self):
        assert get_domain_prompt_profile(None) is None
        assert get_domain_prompt_profile("") is None
        assert get_domain_prompt_profile("   ") is None
        assert get_domain_prompt_profile("astrology") is None


class TestDomainPromptBlockRendering:
    def test_render_prompt_block_contains_all_sections(self):
        profile = DOMAIN_PROMPT_PROFILES["legal"]
        block = profile.render_prompt_block()
        assert "## Domain-Specific Guidance (Legal & Regulatory Domain)" in block
        assert "- Formality: Strictly formal, binding, and legally rigorous." in block
        assert "- Tone & Style: Objective, unambiguous, authoritative, and neutral." in block
        assert "- Register & Conventions:" in block
        assert "- Terminology & Phrasing:" in block
        assert "force majeure" in block

    def test_get_domain_prompt_block_valid_domain(self):
        block = get_domain_prompt_block("finance")
        assert "## Domain-Specific Guidance (Finance & Accounting Domain)" in block
        assert "EBITDA" in block
        assert "GAAP" in block

    def test_get_domain_prompt_block_empty_for_unknown_or_none(self):
        assert get_domain_prompt_block(None) == ""
        assert get_domain_prompt_block("") == ""
        assert get_domain_prompt_block("invalid_domain") == ""


class TestDomainRoleBlockGeneration:
    def test_get_domain_role_block_standard_with_domain(self):
        role_block = get_domain_role_block("operations", "German")
        assert "You are a professional German native translator who specializes in Operations & Technical" in role_block
        assert "fluently translates text into German." in role_block
        assert "Follow all rules strictly." in role_block
        assert "## Domain-Specific Guidance (Operations & Technical Domain)" in role_block
        assert "Standard Operating Procedures (SOPs)" in role_block

    def test_get_domain_role_block_without_domain(self):
        role_block = get_domain_role_block(None, "Spanish")
        assert "You are a professional Spanish native translator who specializes and fluently translates text into Spanish." in role_block
        assert "Follow all rules strictly." in role_block
        assert "## Domain-Specific Guidance" not in role_block

    def test_get_domain_role_block_with_custom_system_prompt_preserves_custom_and_appends_domain(self):
        custom = "Custom identity prompt for testing."
        role_block = get_domain_role_block("hr", "French", custom_system_prompt=custom)
        assert "Custom identity prompt for testing." in role_block
        assert "Follow all rules strictly." in role_block
        assert "## Domain-Specific Guidance (Human Resources & People Domain)" in role_block
        assert "Human Resources" in role_block
