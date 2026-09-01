"""Regression tests for prompt wrapping in `BaseTranslator`.

Production defect: `_run_translation_batch` unconditionally applied
`self.prompt()` to its input, but every `llm_translate()` caller already
passes a fully-built prompt. The model therefore received
`build_translation_prompt(<our own prompt>)` and dutifully translated our
instructions into the target language. The ~2k-char result could not be
typeset into the paragraph's box, the composition stayed empty, and the
paragraph was dropped from the output PDF with
`Unable to export paragraphs that have not yet been formatted`.

These tests pin the contract:
  * `llm_translate(prompt)`  -> model sees exactly `prompt`
  * `translate(text)`        -> model sees `build_translation_prompt(text)`
"""

from __future__ import annotations

from typing import Any

import pytest
from src.worker.doctranslator.translator.base import BaseTranslator
from src.worker.doctranslator.translator.prompts import build_translation_prompt
from src.worker.doctranslator.translator.schemas import TranslationResponse
from src.worker.doctranslator.translator.usage import TokenUsage

CONTENT_START = "<<<TRANSLATE_CONTENT_START>>>"
CONTENT_END = "<<<TRANSLATE_CONTENT_END>>>"


class RecordingTranslator(BaseTranslator):
    """Minimal concrete translator that records what `invoke()` receives."""

    name = "recording"
    provider = "recording"

    def __init__(self, lang_in: str = "en", lang_out: str = "de", domain=None):
        super().__init__(lang_in, lang_out, domain=domain)
        self.model = "recording-model"
        self.temperature = 0.0
        self.seen_contents: list[str] = []
        self.seen_schemas: list[Any] = []
        self.response_text = "TRANSLATED"

    def prompt(self, text: str) -> str:
        return build_translation_prompt(
            text, self.lang_in, self.lang_out, domain=self.domain
        )

    def invoke(self, contents: str, response_schema: Any | None = None) -> Any:
        self.seen_contents.append(contents)
        self.seen_schemas.append(response_schema)
        return self.response_text

    def extract_text(self, response: Any, response_schema: Any | None = None) -> str:
        return str(response)

    def extract_usage(self, response: Any) -> TokenUsage:
        return TokenUsage()

    def apply_usage(self, usage: TokenUsage) -> None:
        return None


@pytest.fixture
def translator(monkeypatch) -> RecordingTranslator:
    # Never touch Redis from a unit test, and never let a cache hit short
    # circuit the assertions below.
    import src.worker.doctranslator.translator.base as base_module

    class _NullCache:
        def get(self, _key):
            return None

        def set(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(base_module, "get_translation_cache", lambda: _NullCache())
    return RecordingTranslator()


PREBUILT_PROMPT = (
    "# Role\nYou are an expert document translator.\n\n"
    "# INPUT\n"
    f"{CONTENT_START}\nVerwandtschaftsverhältnis\n{CONTENT_END}\n"
    "# Output\n"
)


class TestLlmTranslateDoesNotRewrap:
    def test_sends_prebuilt_prompt_verbatim(self, translator: RecordingTranslator):
        translator.llm_translate(PREBUILT_PROMPT)

        assert translator.seen_contents == [PREBUILT_PROMPT]

    def test_content_delimiters_appear_exactly_once(
        self, translator: RecordingTranslator
    ):
        translator.llm_translate(PREBUILT_PROMPT)

        sent = translator.seen_contents[0]
        assert sent.count(CONTENT_START) == 1
        assert sent.count(CONTENT_END) == 1

    def test_does_not_inject_the_translation_task_header(
        self, translator: RecordingTranslator
    ):
        """The double-wrap signature was our own '# Task' block appearing
        around a prompt that already had one."""
        translator.llm_translate("bare caller prompt")

        assert "Translate the INPUT below from" not in translator.seen_contents[0]

    def test_returns_model_output(self, translator: RecordingTranslator):
        assert translator.llm_translate(PREBUILT_PROMPT) == "TRANSLATED"

    def test_none_text_short_circuits(self, translator: RecordingTranslator):
        assert translator.llm_translate(None) is None
        assert translator.seen_contents == []

    def test_defaults_to_translation_response_schema(
        self, translator: RecordingTranslator
    ):
        """Guards the single-paragraph PDF path: with `response_schema=None`
        the schema must still default to `TranslationResponse`, so Gemini's
        `extract_text` can read `parsed.translated_text`."""
        translator.llm_translate(PREBUILT_PROMPT)

        assert translator.seen_schemas == [TranslationResponse]

    def test_explicit_schema_is_forwarded(self, translator: RecordingTranslator):
        from src.worker.doctranslator.translator.schemas import BatchTranslationResponse

        translator.llm_translate(
            PREBUILT_PROMPT, response_schema=BatchTranslationResponse
        )

        assert translator.seen_schemas == [BatchTranslationResponse]


class TestTranslateStillWrapsRawText:
    def test_raw_text_gets_the_standard_wrapper(self, translator: RecordingTranslator):
        translator.translate("hello")

        sent = translator.seen_contents[0]
        assert sent == build_translation_prompt("hello", "en", "de")
        assert "Translate the INPUT below from en into de." in sent
        assert "hello" in sent

    def test_wrapper_applied_exactly_once(self, translator: RecordingTranslator):
        """The delimiters occur twice in a correctly single-wrapped prompt
        (once inside `INJECTION_GUARD_CLAUSE`, once fencing the INPUT).
        Double-wrapping would double that count."""
        translator.translate("hello")

        sent = translator.seen_contents[0]
        assert sent.count(CONTENT_START) == 2
        assert sent.count(CONTENT_END) == 2
        assert sent.count("# INPUT\n") == 1
        assert sent.count("You are an expert document translator") == 1


class TestCacheKeyUsesIncomingText:
    def test_cache_key_is_keyed_on_text_not_prompt(
        self, translator: RecordingTranslator
    ):
        """`_cache_key_for` must hash the caller's text, not the wrapped
        prompt, so `translate()` and `llm_translate()` stay comparable and the
        key does not silently change when the wrapper template is edited."""
        from src.worker.doctranslator.translator.translation_cache import (
            build_cache_key,
        )

        assert translator._cache_key_for("hello") == build_cache_key(
            provider="recording",
            model="recording-model",
            lang_in="en",
            lang_out="de",
            text="hello",
            domain=None,
        )
