"""The detected minority languages must survive the trip to every prompt.

`language_prompts.py` is unit-tested on its own; what these tests cover is
the plumbing between the orchestrator's language distribution and the four
places a translation prompt is built. That plumbing is the fragile part: a
dropped kwarg anywhere along the way leaves the feature silently inert,
with every prompt rendering exactly as it did before and no error raised.
"""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.language_prompts import build_secondary_language_block
from src.worker.doctranslator.format.docx.paragraph_translator import _build_prompt
from src.worker.doctranslator.translator.base import BaseTranslator
from src.worker.doctranslator.translator.prompts import build_translation_prompt
from src.worker.doctranslator.translator.translation_cache import build_cache_key

SECONDARY = [("en", 0.0183), ("de", 0.0076)]
MARKER = "## Mixed-Language Source"


class _ConcreteTranslator(BaseTranslator):
    """Minimal concrete subclass: BaseTranslator is abstract, and the
    behaviour under test lives entirely in its __init__."""

    name = "stub"
    provider = "stub"

    def prompt(self, text):
        return text

    def invoke(self, contents, response_schema=None):
        raise NotImplementedError

    def extract_text(self, response, response_schema=None):
        raise NotImplementedError

    def extract_usage(self, response):
        raise NotImplementedError

    def apply_usage(self, usage):
        raise NotImplementedError


def _block() -> str:
    return build_secondary_language_block(
        SECONDARY, primary_language="fr", target_language="es"
    )


class TestRawTextPrompt:
    """`build_translation_prompt` -- the `translate()` fallback path."""

    def test_block_is_rendered(self):
        prompt = build_translation_prompt(
            "Bonjour", "fr", "es", secondary_languages=SECONDARY
        )
        assert MARKER in prompt

    def test_block_sits_with_the_task_line_it_overrides(self):
        """The Task line is the only place this prompt names a source
        language, so the exception to it has to be adjacent -- not buried
        below the terminology and formatting sections."""
        prompt = build_translation_prompt(
            "Bonjour", "fr", "es", secondary_languages=SECONDARY
        )
        assert prompt.index("# Task") < prompt.index(MARKER)
        assert prompt.index(MARKER) < prompt.index("# Output format")

    def test_monolingual_prompt_is_byte_identical(self):
        assert build_translation_prompt(
            "Bonjour", "fr", "es", secondary_languages=[]
        ) == build_translation_prompt("Bonjour", "fr", "es")


class TestPdfTemplates:
    """Both PDF templates -- the real high-volume paths.

    Neither template contained a source language at all before this
    change, which is why a minority-language paragraph had nothing to
    tell the model it was not written in the dominant language.
    """

    def test_batch_template_renders_the_block_before_its_rules(self):
        from src.worker.doctranslator.format.pdf.document_il.midend.il_translator_llm_only import (
            PROMPT_TEMPLATE,
        )

        rendered = PROMPT_TEMPLATE.substitute(
            role_block="ROLE",
            secondary_language_block=_block(),
            glossary_usage_rules_block="",
            contextual_hints_block="",
            json_input_str="[]",
            glossary_tables_block="",
            security_notice="SEC",
            lang_out="es",
        )
        assert MARKER in rendered
        assert rendered.index(MARKER) < rendered.index("## Structure Rules")

    def test_single_paragraph_template_renders_the_block_before_its_rules(self):
        from src.worker.doctranslator.format.pdf.document_il.midend.il_translator import (
            PROMPT_TEMPLATE,
        )

        rendered = PROMPT_TEMPLATE.substitute(
            role_block="ROLE",
            secondary_language_block=_block(),
            glossary_block="",
            context_block="",
            security_notice="SEC",
            lang_out="es",
            text_to_translate="x",
        )
        assert MARKER in rendered
        assert rendered.index(MARKER) < rendered.index("## Rules")

    @pytest.mark.parametrize(
        "module_path",
        [
            "src.worker.doctranslator.format.pdf.document_il.midend.il_translator",
            "src.worker.doctranslator.format.pdf.document_il.midend.il_translator_llm_only",
        ],
    )
    def test_builder_reads_the_field_off_translation_config(self, module_path):
        """The templates are fed from `translation_config.secondary_languages`;
        this is the join that `processor_service` populates."""
        import importlib

        module = importlib.import_module(module_path)
        cls = (
            module.ILTranslatorLLMOnly
            if "llm_only" in module_path
            else module.ILTranslator
        )
        instance = cls.__new__(cls)
        instance.translation_config = SimpleNamespace(
            secondary_languages=SECONDARY, lang_in="fr", lang_out="es"
        )
        assert MARKER in instance._build_secondary_language_block()

    def test_monolingual_config_renders_nothing(self):
        from src.worker.doctranslator.format.pdf.document_il.midend.il_translator_llm_only import (
            ILTranslatorLLMOnly,
        )

        instance = ILTranslatorLLMOnly.__new__(ILTranslatorLLMOnly)
        instance.translation_config = SimpleNamespace(
            secondary_languages=[], lang_in="fr", lang_out="es"
        )
        assert instance._build_secondary_language_block() == ""


class TestDocxPrompt:
    def test_block_is_rendered(self):
        unit = SimpleNamespace(unit_id=0, text="Bonjour", label="text")
        prompt = _build_prompt(
            [unit], "es", domain=None, lang_in="fr", secondary_languages=SECONDARY
        )
        assert MARKER in prompt
        assert prompt.index(MARKER) < prompt.index("## Structure Rules")

    def test_translator_reads_languages_off_the_engine(self):
        """DOCX carries these on the translator rather than through
        `translate_docx`, mirroring how `domain` already reaches it."""
        from src.worker.doctranslator.format.docx.paragraph_translator import (
            DocxParagraphTranslator,
        )

        engine = MagicMock()
        engine.domain = None
        engine.lang_in = "fr"
        engine.secondary_languages = SECONDARY

        translator = DocxParagraphTranslator(engine, "es")
        assert translator.secondary_languages == SECONDARY
        assert translator.lang_in == "fr"


class TestTranslatorPlumbing:
    def test_factory_passes_languages_to_the_translator(self):
        from src.worker.doctranslator.translator import factory

        with patch.object(factory, "GeminiVertexAITranslator") as gemini:
            factory.create_translator(
                "gemini-2.5-flash",
                lang_in="fr",
                lang_out="es",
                qps=1,
                secondary_languages=SECONDARY,
            )
        assert gemini.call_args.kwargs["secondary_languages"] == SECONDARY

    def test_base_translator_stores_and_defaults(self):
        translator = _ConcreteTranslator("fr", "es", secondary_languages=SECONDARY)
        assert translator.secondary_languages == SECONDARY

        # Every existing caller omits the argument; they must keep working
        # and must not share a mutable default between instances.
        default = _ConcreteTranslator("fr", "es")
        assert default.secondary_languages == []
        default.secondary_languages.append(("it", 0.1))
        assert _ConcreteTranslator("fr", "es").secondary_languages == []


class TestCacheKey:
    """A prompt input that is not in the cache key silently serves the
    wrong translation: a run that had the mixed-language instruction would
    reuse an entry produced without it, for the same text."""

    BASE = {
        "provider": "gemini",
        "model": "m",
        "lang_in": "fr",
        "lang_out": "es",
        "text": "x",
    }

    def test_mixed_and_monolingual_runs_do_not_share_an_entry(self):
        assert build_cache_key(**self.BASE) != build_cache_key(
            **self.BASE, secondary_languages=SECONDARY
        )

    def test_different_secondary_languages_do_not_share_an_entry(self):
        assert build_cache_key(
            **self.BASE, secondary_languages=SECONDARY
        ) != build_cache_key(**self.BASE, secondary_languages=[("it", 0.02)])

    def test_empty_list_leaves_existing_keys_untouched(self):
        """Monolingual jobs are the overwhelming majority; their cache
        entries must survive this change."""
        assert build_cache_key(**self.BASE) == build_cache_key(
            **self.BASE, secondary_languages=[]
        )

    def test_key_is_order_independent(self):
        assert build_cache_key(
            **self.BASE, secondary_languages=[("en", 0.02), ("de", 0.01)]
        ) == build_cache_key(
            **self.BASE, secondary_languages=[("de", 0.01), ("en", 0.02)]
        )


class TestProcessorServiceThreading:
    def test_config_value_reaches_translator_and_translation_config(self):
        """`_build_translation_config` must pass the languages twice --
        once to the translator (raw-text prompt) and once to
        TranslationConfig (both PDF templates) -- exactly as `domain` is."""
        from src.worker.services import processor_service

        processor = processor_service.JobProcessor(progress_tracker=MagicMock())
        config = {
            "job_id": "job-1",
            "input_file": "in.pdf",
            "lang_in": "fr",
            "lang_out": "es",
            "model_list": ["gemini-2.5-flash"],
            "secondary_languages": SECONDARY,
        }

        with (
            patch.object(processor, "_get_doc_layout_model", return_value=MagicMock()),
            patch.object(
                processor_service, "create_translator", return_value=MagicMock()
            ) as make_translator,
            patch.object(processor_service, "TranslationConfig") as translation_config,
            patch("pathlib.Path.mkdir"),
        ):
            processor._build_translation_config(config, Path("out"))

        assert make_translator.call_args.kwargs["secondary_languages"] == SECONDARY
        assert translation_config.call_args.kwargs["secondary_languages"] == SECONDARY

    def test_absent_key_yields_an_empty_list(self):
        """Callers that predate this key (and the DOCX/TXT paths' own
        configs) must not crash or inherit a stale value."""
        from src.worker.services import processor_service

        processor = processor_service.JobProcessor(progress_tracker=MagicMock())
        config = {
            "job_id": "job-1",
            "input_file": "in.pdf",
            "lang_in": "fr",
            "lang_out": "es",
            "model_list": ["gemini-2.5-flash"],
        }

        with (
            patch.object(processor, "_get_doc_layout_model", return_value=MagicMock()),
            patch.object(
                processor_service, "create_translator", return_value=MagicMock()
            ) as make_translator,
            patch.object(processor_service, "TranslationConfig"),
            patch("pathlib.Path.mkdir"),
        ):
            processor._build_translation_config(config, Path("out"))

        assert make_translator.call_args.kwargs["secondary_languages"] == []


class TestOrchestratorSelection:
    def _orchestrator(self):
        from src.worker.services.pipeline_orchestrator import PipelineOrchestrator

        return PipelineOrchestrator.__new__(PipelineOrchestrator)

    def test_selects_minority_languages(self):
        selected = self._orchestrator()._secondary_prompt_languages(
            job_id="job-1",
            language_distribution=Counter({"fr": 11554, "en": 217, "de": 90}),
            source_lang="fr",
            target_lang="es",
        )
        assert [code for code, _share in selected] == ["en", "de"]

    def test_feature_flag_disables_it(self):
        from src.worker.services import pipeline_orchestrator

        with patch.object(
            pipeline_orchestrator.settings,
            "MIXED_LANGUAGE_PROMPT_HINT_ENABLED",
            False,
        ):
            selected = self._orchestrator()._secondary_prompt_languages(
                job_id="job-1",
                language_distribution=Counter({"fr": 11554, "en": 217}),
                source_lang="fr",
                target_lang="es",
            )
        assert selected == []

    def test_monolingual_document_selects_nothing(self):
        selected = self._orchestrator()._secondary_prompt_languages(
            job_id="job-1",
            language_distribution=Counter({"fr": 11554}),
            source_lang="fr",
            target_lang="es",
        )
        assert selected == []
