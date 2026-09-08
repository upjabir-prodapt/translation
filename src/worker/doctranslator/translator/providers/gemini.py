"""Gemini Vertex AI translator provider."""

from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types as genai_types

from src.config.constants import settings
from src.config.retry import llm_retry
from src.worker.doctranslator.translator.base import BaseTranslator
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_MODEL
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_NAME
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_PROMPT_CHARS
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_PROMPT_HASH
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_PROMPT_PREVIEW
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_PROVIDER
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_TEMPERATURE
from src.worker.doctranslator.translator.instrumentation import instrumented_llm_call
from src.worker.doctranslator.translator.instrumentation import prompt_fingerprint
from src.worker.doctranslator.translator.prompts import build_translation_prompt
from src.worker.doctranslator.translator.provider_types import LLMProvider
from src.worker.doctranslator.translator.schemas import BatchTranslationResponse
from src.worker.doctranslator.translator.schemas import TermExtractionResponse
from src.worker.doctranslator.translator.schemas import TranslationResponse
from src.worker.doctranslator.translator.usage import TokenUsage
from src.worker.doctranslator.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)


def _usage_metadata_summary(response) -> str:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return "usage=unavailable"
    pt = getattr(usage, "prompt_token_count", None)
    ct = getattr(usage, "candidates_token_count", None)
    tt = getattr(usage, "total_token_count", None)
    ch = getattr(usage, "cached_content_token_count", None)
    return (
        f"prompt_tokens={pt} completion_tokens={ct} total_tokens={tt} "
        f"cached_tokens={ch}"
    )


class GeminiVertexAITranslator(BaseTranslator):
    """Translator backed by Google GenAI SDK (Vertex AI with service account)."""

    name = "gemini"
    provider = LLMProvider.GEMINI_VERTEXAI

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        temperature=0.0,
        region: str | None = None,
        domain: str | None = None,
    ):
        super().__init__(lang_in, lang_out, domain=domain)
        if genai is None:
            raise ImportError(
                "google-genai is required for Gemini translator. "
                "Install it with `uv add google-genai`."
            )

        self.model = model
        self.temperature = temperature
        self.region = (
            region or settings.GEMINI_MODEL_REGION or settings.GOOGLE_CLOUD_LOCATION
        )
        # A client-side deadline is essential: without it a single slow
        # generation blocks a pool worker indefinitely (a 454s call was
        # observed in the 2026-08-24 baseline). HttpOptions.timeout is in
        # milliseconds. Timeouts surface as retryable errors via
        # `_RETRYABLE_ERROR_SUBSTRINGS`, so `llm_retry` handles them.
        self.client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=self.region,
            http_options=genai_types.HttpOptions(
                timeout=int(float(settings.LLM_CALL_TIMEOUT_SECONDS) * 1000)
            ),
        )
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

    def prompt(self, text: str) -> str:
        return build_translation_prompt(text, self.lang_out, domain=self.domain)

    def _build_thinking_config(self) -> genai_types.ThinkingConfig | None:
        """Resolve the reasoning budget for this model.

        Gemini defaults to *dynamic* thinking, which in the 2026-08-24
        baseline produced calls emitting 13 response tokens after 123s of
        hidden reasoning. Pro-class models cannot fully disable thinking, so
        they get a separate (small) budget rather than 0.

        Returning None restores the SDK default (configure with -1), which is
        the documented escape hatch if a quality regression is observed.
        """
        is_pro = "pro" in str(self.model).lower()
        budget = int(
            settings.LLM_THINKING_BUDGET_PRO if is_pro else settings.LLM_THINKING_BUDGET
        )
        if budget < 0:
            return None
        return genai_types.ThinkingConfig(thinking_budget=budget)

    def extract_text(self, response, response_schema=None) -> str:
        """Prefer the SDK's server-validated `response.parsed` for every
        structured-output schema, avoiding redundant client-side re-parsing
        of `response.text` (see docs/plan.md Section 4.6). `translate_text`
        is returned directly for single-unit translation; the batch/term
        schemas are serialized back to the same JSON shape callers already
        expect (`{"items": [...]}` / `{"terms": [...]}`) so
        paragraph_translator.py/term_extractor.py/il_translator_llm_only.py
        need no changes to their existing json-parsing logic.

        Accessing `response.parsed` is not risk-free: it is the SDK
        constructing/validating our pydantic schema from the model's raw
        JSON, and a single field the model failed to populate (e.g. a
        `TermExtractionResponse` item missing the required `src_lang`) makes
        that construction raise instead of returning `None`. `getattr(...,
        default)` only swallows a genuine `AttributeError` from attribute
        lookup, not an exception raised by the property getter itself, so
        that failure was previously left to propagate out of this method and
        abort the whole batch/thread it was called from. Wrapping the access
        lets one malformed item fall through to the `response.text`
        re-parse below instead.
        """
        try:
            parsed = getattr(response, "parsed", None)
        except Exception:
            logger.warning(
                "Gemini response.parsed construction failed (likely a "
                "required field missing from the model's structured output); "
                "falling back to response.text.",
                exc_info=True,
            )
            parsed = None
        if parsed is not None:
            if isinstance(parsed, TranslationResponse):
                return parsed.translated_text
            if isinstance(parsed, (BatchTranslationResponse, TermExtractionResponse)):
                return parsed.model_dump_json()

        text = getattr(response, "text", "")
        if text:
            text = text.strip()
            if text.startswith("{"):
                try:
                    data = json.loads(text)
                    if isinstance(data, dict) and "translated_text" in data:
                        return str(data["translated_text"])
                except Exception:
                    pass
            return text
        return ""

    def extract_usage(self, response) -> TokenUsage:
        return TokenUsage.from_gemini_usage(getattr(response, "usage_metadata", None))

    def apply_usage(self, usage: TokenUsage) -> None:
        if usage.total_tokens:
            self.token_count.inc(usage.total_tokens)
        if usage.input_tokens:
            self.prompt_token_count.inc(usage.input_tokens)
        if usage.output_tokens:
            self.completion_token_count.inc(usage.output_tokens)
        if usage.cache_hit_input_tokens:
            self.cache_hit_prompt_token_count.inc(usage.cache_hit_input_tokens)

    @llm_retry(logger=logger)
    def _generate_content_with_retry(
        self, *, model: str, contents: str, config: genai_types.GenerateContentConfig
    ):
        prompt_chars, prompt_hash, prompt_preview = prompt_fingerprint(contents)
        temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        max_output_tokens = int(getattr(config, "max_output_tokens", 0) or 0)
        logger.debug(
            f"Gemini generate_content: name={self.name} model={model} "
            f"prompt_chars={prompt_chars} prompt_hash={prompt_hash} "
            f"temperature={temperature} max_output_tokens={max_output_tokens} "
            f"prompt_preview={prompt_preview!r}",
        )

        def _call():
            return self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

        return instrumented_llm_call(
            span_name="gemini.generate_content",
            attributes={
                ATTR_LLM_NAME: model,
                ATTR_LLM_MODEL: model,
                ATTR_LLM_PROVIDER: LLMProvider.GEMINI_VERTEXAI,
                ATTR_LLM_TEMPERATURE: temperature,
                ATTR_LLM_PROMPT_CHARS: prompt_chars,
                ATTR_LLM_PROMPT_HASH: prompt_hash,
                ATTR_LLM_PROMPT_PREVIEW: prompt_preview,
                "llm.max_output_tokens": max_output_tokens,
            },
            call=_call,
            usage_fn=lambda resp: TokenUsage.from_gemini_usage(
                getattr(resp, "usage_metadata", None)
            ),
            log_prefix=f"Gemini generate_content model={model}",
        )

    def invoke(self, contents: str, response_schema: Any | None = None) -> Any:
        schema = response_schema if response_schema is not None else TranslationResponse
        translate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=self._build_thinking_config(),
        )
        response = self._generate_content_with_retry(
            model=self.model,
            contents=contents,
            config=translate_config,
        )
        logger.debug(
            f"Gemini invoke totals: name={self.name} {_usage_metadata_summary(response)}"
        )
        return response
