"""Tests for domain role block generation in PDF IL translators."""

from unittest.mock import MagicMock

from src.worker.doctranslator.format.pdf.document_il.midend.il_translator import (
    ILTranslator,
)
from src.worker.doctranslator.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
)


def _make_mock_config(domain: str | None = None, custom_system_prompt: str | None = None):
    config = MagicMock()
    config.lang_in = "en"
    config.lang_out = "es"
    config.domain = domain
    config.custom_system_prompt = custom_system_prompt
    config.shared_context_cross_split_part.get_glossaries_for_translation.return_value = []
    config.auto_extract_glossary = False
    return config


class TestIlTranslatorDomainRoleBlocks:
    def test_il_translator_role_block_with_legal_domain(self):
        config = _make_mock_config(domain="legal")
        translator = ILTranslator.__new__(ILTranslator)
        translator.translation_config = config

        role_block = translator._build_role_block()
        assert "Legal & Regulatory" in role_block
        assert "## Domain-Specific Guidance (Legal & Regulatory Domain)" in role_block
        assert "Strictly formal, binding, and legally rigorous." in role_block
        assert "Follow all rules strictly." in role_block

    def test_il_translator_role_block_with_finance_domain(self):
        config = _make_mock_config(domain="finance")
        translator = ILTranslator.__new__(ILTranslator)
        translator.translation_config = config

        role_block = translator._build_role_block()
        assert "Finance & Accounting" in role_block
        assert "EBITDA" in role_block

    def test_il_translator_role_block_no_domain(self):
        config = _make_mock_config(domain=None)
        translator = ILTranslator.__new__(ILTranslator)
        translator.translation_config = config

        role_block = translator._build_role_block()
        assert "## Domain-Specific Guidance" not in role_block
        assert "Follow all rules strictly." in role_block

    def test_il_translator_llm_only_role_block_with_domain(self):
        config = _make_mock_config(domain="operations")
        translator = ILTranslatorLLMOnly.__new__(ILTranslatorLLMOnly)
        translator.translation_config = config

        role_block = translator._build_llm_role_block()
        assert "Operations & Technical" in role_block
        assert "Standard Operating Procedures (SOPs)" in role_block
        assert "Follow all rules strictly." in role_block

    def test_custom_system_prompt_is_preserved_with_domain_appended(self):
        config = _make_mock_config(domain="hr", custom_system_prompt="My Custom Enterprise Translator.")
        translator = ILTranslatorLLMOnly.__new__(ILTranslatorLLMOnly)
        translator.translation_config = config

        role_block = translator._build_llm_role_block()
        assert "My Custom Enterprise Translator." in role_block
        assert "Follow all rules strictly." in role_block
        assert "## Domain-Specific Guidance (Human Resources & People Domain)" in role_block
