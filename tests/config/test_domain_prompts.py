"""Domain prompt registry and per-domain prompt segregation tests."""

import pytest
from src.config.domain_prompts import DOMAIN_PROMPT_PROFILES
from src.config.domain_prompts import GENERIC_DOMAIN
from src.config.domain_prompts import SUPPORTED_DOMAINS
from src.config.domain_prompts import build_domain_judge_block
from src.config.domain_prompts import build_domain_role_block
from src.config.domain_prompts import build_domain_term_extraction_block
from src.config.domain_prompts import build_domain_translation_block
from src.config.domain_prompts import get_domain_profile
from src.config.domain_prompts import normalize_domain_key
from src.doctranslator.translator.prompts import build_translation_prompt

ALL_DOMAINS = sorted(SUPPORTED_DOMAINS)


class TestRegistry:
    def test_expected_domains(self):
        assert SUPPORTED_DOMAINS == {
            "commercial",
            "legal",
            "finance",
            "hr",
            "operations",
        }

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_profile_key_matches_registry_key(self, domain):
        assert DOMAIN_PROMPT_PROFILES[domain].key == domain

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_profile_content_is_populated(self, domain):
        profile = DOMAIN_PROMPT_PROFILES[domain]
        assert profile.label
        assert profile.persona
        assert profile.register
        assert profile.judge_severity
        assert len(profile.non_negotiables) >= 3
        assert profile.terminology
        assert profile.pitfalls
        assert profile.judge_focus
        assert profile.term_focus

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Legal", "legal"),
            ("  FINANCE ", "finance"),
            ("oprations", "operations"),
            ("ops", "operations"),
            ("human resources", "hr"),
            ("sales", "commercial"),
        ],
    )
    def test_alias_and_case_normalization(self, value, expected):
        assert normalize_domain_key(value) == expected

    @pytest.mark.parametrize("value", ["", None, "marketing", "unknown-domain"])
    def test_unknown_domain_degrades_to_generic(self, value):
        assert normalize_domain_key(value) == GENERIC_DOMAIN
        assert get_domain_profile(value).key == GENERIC_DOMAIN


class TestBlockRendering:
    @pytest.mark.parametrize(
        "builder",
        [
            build_domain_translation_block,
            build_domain_judge_block,
            build_domain_term_extraction_block,
        ],
    )
    def test_blocks_are_distinct_per_domain(self, builder):
        rendered = {domain: builder(domain) for domain in ALL_DOMAINS}
        assert len(set(rendered.values())) == len(ALL_DOMAINS)
        for text in rendered.values():
            assert text.strip()

    def test_role_blocks_are_distinct_per_domain(self):
        rendered = {
            domain: build_domain_role_block(domain, "French") for domain in ALL_DOMAINS
        }
        assert len(set(rendered.values())) == len(ALL_DOMAINS)

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_role_block_names_target_language_and_closes_with_rule(self, domain):
        block = build_domain_role_block(domain, "German")
        assert "German" in block
        # The IL role block contract: il_translator only appends this sentence
        # when absent, so the profile must already end with it.
        assert block.rstrip().endswith("Follow all rules strictly.")

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_domain_label_appears_in_every_block(self, domain):
        label = DOMAIN_PROMPT_PROFILES[domain].label
        assert label in build_domain_translation_block(domain)
        assert label in build_domain_role_block(domain, "Spanish")
        assert label in build_domain_judge_block(domain)
        assert label in build_domain_term_extraction_block(domain)

    def test_domain_specific_guidance_is_targeted(self):
        assert "modality" in build_domain_judge_block("legal").lower()
        assert "shall" in build_domain_translation_block("legal")
        assert "IFRS" in build_domain_translation_block("finance")
        assert "DANGER" in build_domain_translation_block("operations")
        assert "gender-neutral" in build_domain_translation_block("hr")
        assert "Incoterms" in build_domain_translation_block("commercial")

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_role_block_does_not_duplicate_the_rule_lists(self, domain):
        # The role block travels inside the payload that the translation prompt
        # wraps; repeating the rule lists would double the prompt cost of every
        # batch call for no added instruction.
        role_block = build_domain_role_block(domain, "French")
        profile = DOMAIN_PROMPT_PROFILES[domain]
        for rule in profile.non_negotiables + profile.pitfalls:
            assert rule not in role_block


class TestTranslationPrompt:
    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_prompt_embeds_domain_block(self, domain):
        prompt = build_translation_prompt("hello", "en", "fr", domain=domain)
        assert build_domain_translation_block(domain) in prompt
        assert "hello" in prompt
        assert "__DLP_TOKEN_NNNN__" in prompt

    def test_prompts_differ_across_domains(self):
        prompts = {
            domain: build_translation_prompt("hello", "en", "fr", domain=domain)
            for domain in ALL_DOMAINS
        }
        assert len(set(prompts.values())) == len(ALL_DOMAINS)

    def test_domain_block_precedes_input(self):
        prompt = build_translation_prompt("payload", "en", "de", domain="legal")
        assert prompt.index("# Domain profile: Legal") < prompt.index("# INPUT")

    def test_missing_domain_falls_back_to_generic(self):
        prompt = build_translation_prompt("hello", "en", "fr")
        assert build_domain_translation_block(None) in prompt
