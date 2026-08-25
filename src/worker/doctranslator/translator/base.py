"""Base translator template methods and shared APIs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import unicodedata
from abc import ABC
from abc import abstractmethod
from typing import Any

from opentelemetry.trace import SpanKind

from src.config.tracing import set_root_span_attribute
from src.config.tracing import tracer_llm
from src.worker.doctranslator.batching import llm_call_slot
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_INPUT_CHARS
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_INPUT_TOKENS
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_LATENCY_S
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_MODEL
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_NAME
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_OUTPUT_CHARS
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_OUTPUT_TOKENS
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_PROMPT_CHARS
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_PROVIDER
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_TEMPERATURE
from src.worker.doctranslator.translator.instrumentation import ATTR_LLM_TOTAL_TOKENS
from src.worker.doctranslator.translator.rate_limiter import get_translate_rate_limiter
from src.worker.doctranslator.translator.translation_cache import build_cache_key
from src.worker.doctranslator.translator.translation_cache import get_translation_cache
from src.worker.doctranslator.translator.usage import TokenUsage

logger = logging.getLogger(__name__)

_MAX_CHARS_LOG_PREVIEW = 120


def remove_control_characters(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class BaseTranslator(ABC):
    """Template-method base for Gemini and Claude translators."""

    name = "base"
    lang_map: dict[str, str] = {}
    provider: str = "base"

    def __init__(self, lang_in: str, lang_out: str, domain: str | None = None):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.domain = domain
        self.translate_call_count = 0

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.info(
                f"{self.name} translate call count: {self.translate_call_count}"
            )

    def translate(self, text, rate_limit_params: dict | None = None):
        self.translate_call_count += 1
        get_translate_rate_limiter(str(self.provider)).wait(rate_limit_params)
        return self.do_translate(text, rate_limit_params)

    def llm_translate(
        self,
        text,
        rate_limit_params: dict | None = None,
        response_schema=None,
        batch_items: int | None = None,
    ):
        self.translate_call_count += 1
        in_len = len(text) if isinstance(text, str) else 0
        rl_keys = sorted(rate_limit_params.keys()) if rate_limit_params else []
        logger.debug(
            f"llm_translate: translator={self.name} call={self.translate_call_count} "
            f"chars_in={in_len} rate_limit_param_keys={rl_keys}",
        )
        get_translate_rate_limiter(str(self.provider)).wait(rate_limit_params)
        return self.do_llm_translate(
            text, rate_limit_params, response_schema, batch_items=batch_items
        )


    async def llm_translate_async(
        self, text, rate_limit_params: dict | None = None, response_schema=None
    ):
        in_len = len(text) if isinstance(text, str) else 0
        logger.debug(
            f"llm_translate_async: translator={self.name} scheduling thread "
            f"chars_in={in_len}",
        )
        return await asyncio.to_thread(
            self.llm_translate, text, rate_limit_params, response_schema
        )

    @abstractmethod
    def prompt(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def invoke(
        self,
        contents: str,
        response_schema: Any | None = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def extract_text(self, response: Any, response_schema: Any | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_usage(self, response: Any) -> TokenUsage:
        raise NotImplementedError

    @abstractmethod
    def apply_usage(self, usage: TokenUsage) -> None:
        raise NotImplementedError

    def do_translate(
        self, text, rate_limit_params: dict | None = None, batch_items: int | None = None
    ):
        from src.worker.doctranslator.translator.schemas import TranslationResponse

        return self._run_translation_batch(
            text=text,
            response_schema=TranslationResponse,
            rate_limit_params=rate_limit_params,
            operation="do_translate",
            batch_items=batch_items,
        )

    def do_llm_translate(
        self,
        text,
        rate_limit_params: dict | None = None,
        response_schema=None,
        batch_items: int | None = None,
    ):
        if text is None:
            logger.debug("do_llm_translate skipped: text is None")
            return None
        from src.worker.doctranslator.translator.schemas import TranslationResponse

        schema = response_schema if response_schema is not None else TranslationResponse
        return self._run_translation_batch(
            text=text,
            response_schema=schema,
            rate_limit_params=rate_limit_params,
            operation="do_llm_translate",
            batch_items=batch_items,
        )

    def _cache_key_for(self, text: str) -> str | None:
        if not isinstance(text, str) or not text:
            return None
        return build_cache_key(
            provider=str(self.provider),
            model=str(getattr(self, "model", "")),
            lang_in=self.lang_in,
            lang_out=self.lang_out,
            text=text,
            domain=self.domain,
        )

    def _run_translation_batch(
        self,
        *,
        text,
        response_schema,
        rate_limit_params: dict | None,
        operation: str,
        batch_items: int | None = None,
    ) -> str:
        cache_key = self._cache_key_for(text if isinstance(text, str) else None)
        if cache_key is not None:
            cached = get_translation_cache().get(cache_key)
            if cached is not None:
                logger.info(
                    f"{operation} cache hit: name={self.name} model={self.model} "
                    f"{self.lang_in}->{self.lang_out} cache_key={cache_key[:12]}",
                )
                return cached

        contents = self.prompt(text)
        c_len = len(contents)
        input_chars = len(text) if isinstance(text, str) else 0
        rl_keys = sorted(rate_limit_params.keys()) if rate_limit_params else []
        logger.debug(
            f"{operation}: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"input_chars={input_chars} prompt_chars={c_len} "
            f"temperature={self.temperature} rate_limit_param_keys={rl_keys}",
        )
        t0 = time.monotonic()

        with tracer_llm.start_as_current_span(
            "llm.translate_batch",
            kind=SpanKind.CLIENT,
            attributes={
                ATTR_LLM_NAME: self.model,
                ATTR_LLM_MODEL: self.model,
                ATTR_LLM_PROVIDER: self.provider,
                ATTR_LLM_TEMPERATURE: float(self.temperature),
                ATTR_LLM_INPUT_CHARS: input_chars,
                ATTR_LLM_PROMPT_CHARS: c_len,
                "translation.source_lang": self.lang_in,
                "translation.target_lang": self.lang_out,
            },
        ) as span:
            # Single choke point for the process-wide in-flight-call budget.
            # Nested thread pools (batch pool -> per-batch fallback pool) can
            # otherwise put ~156 simultaneous Vertex calls in flight from one
            # job on a cpu=4 / concurrency=1 instance.
            with llm_call_slot():
                response = self.invoke(contents, response_schema)
            usage = self.extract_usage(response)
            self.apply_usage(usage)
            out = self.extract_text(response, response_schema)
            o_len = len(out)
            span.set_attribute(ATTR_LLM_INPUT_TOKENS, usage.input_tokens)
            span.set_attribute(ATTR_LLM_OUTPUT_TOKENS, usage.output_tokens)
            span.set_attribute(ATTR_LLM_TOTAL_TOKENS, usage.total_tokens)
            span.set_attribute(ATTR_LLM_OUTPUT_CHARS, o_len)
            elapsed = time.monotonic() - t0
            span.set_attribute(ATTR_LLM_LATENCY_S, round(elapsed, 3))

        set_root_span_attribute("llm.model", self.model)
        set_root_span_attribute("llm.name", self.model)
        if not out.strip() and c_len > 0:
            prev = contents[:_MAX_CHARS_LOG_PREVIEW].replace("\n", "\\n")
            logger.warning(
                f"{operation} empty model output: name={self.name} model={self.model} "
                f"prompt_chars={c_len} prompt_head={prev!r}",
            )
        # E1/Phase-0 instrumentation: `thinking_tokens` exposes reasoning cost
        # that `output_tokens` alone hides, and `batch_items` lets a log reader
        # derive per-item latency and pool utilisation without guesswork.
        batch_items_str = "" if batch_items is None else f" batch_items={batch_items}"
        logger.info(
            f"{operation} done: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"latency_s={elapsed:.3f} input_chars={input_chars} "
            f"prompt_chars={c_len} out_chars={o_len} "
            f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens} "
            f"thinking_tokens={usage.thinking_tokens} "
            f"billable_output_tokens={usage.billable_output_tokens}"
            f"{batch_items_str}",
        )
        if cache_key is not None and out.strip():
            get_translation_cache().set(
                cache_key,
                out,
                provider=str(self.provider),
                model=str(getattr(self, "model", "")),
                lang_in=self.lang_in,
                lang_out=self.lang_out,
            )
        return out


    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, placeholder_id: int | str):
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int | str):
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int | str):
        return self.get_rich_text_left_placeholder(placeholder_id)
