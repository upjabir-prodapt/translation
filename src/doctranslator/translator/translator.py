import asyncio
import contextlib
import hashlib
import logging
import threading
import time
import unicodedata
from abc import ABC
from abc import abstractmethod

from google import genai
from google.genai import types as genai_types
from opentelemetry.trace import SpanKind
from pydantic import BaseModel
from pydantic import Field

from src.config.constants import settings
from src.config.retry import llm_retry
from src.config.tracing import set_root_span_attribute
from src.config.tracing import tracer_llm
from src.doctranslator.translator.prompts import build_translation_prompt
from src.doctranslator.translator.providers import LLMProvider
from src.doctranslator.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)

_MAX_CHARS_LOG_PREVIEW = 120


class TranslationResponse(BaseModel):
    """Structured translation response from the model."""

    translated_text: str = Field(description="The translated text.")


class BatchTranslationItem(BaseModel):
    id: int = Field(description="The id of the input paragraph.")
    output: str = Field(description="The translated text for this paragraph.")


class BatchTranslationResponse(BaseModel):
    """Structured response for batch paragraph translation."""

    items: list[BatchTranslationItem] = Field(
        description="Translated paragraphs in the same order as the input."
    )


class ExtractedTerm(BaseModel):
    src: str = Field(description="Source language term.")
    tgt: str = Field(description="Translated term in the target language.")


class TermExtractionResponse(BaseModel):
    """Structured response for automatic term extraction."""

    terms: list[ExtractedTerm] = Field(
        description="Extracted term pairs from the source text."
    )


# OTel span attribute key constants (avoids S1192 duplicate-literal warnings)
_ATTR_LLM_MODEL = "llm.model"
_ATTR_LLM_NAME = "llm.name"
_ATTR_LLM_PROVIDER = "llm.provider"
_ATTR_LLM_TEMPERATURE = "llm.temperature"
_ATTR_LLM_INPUT_TOKENS = "llm.input_tokens"
_ATTR_LLM_OUTPUT_TOKENS = "llm.output_tokens"
_ATTR_LLM_TOTAL_TOKENS = "llm.total_tokens"
_ATTR_LLM_PROMPT_CHARS = "llm.prompt_chars"
_ATTR_LLM_PROMPT_HASH = "llm.prompt_hash"
_ATTR_LLM_PROMPT_PREVIEW = "llm.prompt_preview"
_ATTR_LLM_INPUT_CHARS = "llm.input_chars"
_ATTR_LLM_OUTPUT_CHARS = "llm.output_chars"
_ATTR_LLM_LATENCY_S = "llm.latency_s"

_MAX_CHARS_PROMPT_PREVIEW = 300


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


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class RateLimiter:
    """
    A rate limiter using the leaky bucket algorithm to ensure a smooth, constant rate of requests.
    This implementation is thread-safe and robust against system clock changes.
    """

    def __init__(self, max_qps: int):
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.lock = threading.Lock()
        # Use monotonic time to prevent issues with system time changes
        self.next_request_time = time.monotonic()

    def wait(self, _rate_limit_params: dict = None):
        """
        Blocks until the next request can be processed, ensuring the rate limit is not exceeded.
        """
        with self.lock:
            now = time.monotonic()

            wait_duration = self.next_request_time - now
            if wait_duration > 0:
                time.sleep(wait_duration)

            # Update the next allowed request time.
            # If the limiter has been idle, the next request should start from 'now'.
            now = time.monotonic()
            self.next_request_time = (
                max(self.next_request_time, now) + self.min_interval
            )

    def set_max_qps(self, max_qps: int):
        """
        Updates the maximum queries per second. This operation is thread-safe.
        """
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        with self.lock:
            self.max_qps = max_qps
            self.min_interval = 1.0 / max_qps


_translate_rate_limiter = RateLimiter(max(int(settings.TRANSLATION_MAX_QPS), 1))


def set_translate_rate_limiter(max_qps):
    _translate_rate_limiter.set_max_qps(max_qps)


class BaseTranslator(ABC):
    name = "base"
    lang_map = {}

    def __init__(self, lang_in, lang_out):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.translate_call_count = 0

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.info(
                f"{self.name} translate call count: {self.translate_call_count}"
            )

    def translate(self, text, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        _translate_rate_limiter.wait()
        return self.do_translate(text, rate_limit_params)

    def llm_translate(self, text, rate_limit_params: dict = None, response_schema=None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :param response_schema: optional Pydantic model to enforce structured output
        :return: translated text
        """
        self.translate_call_count += 1
        in_len = len(text) if isinstance(text, str) else 0
        rl_keys = sorted(rate_limit_params.keys()) if rate_limit_params else []
        logger.debug(
            f"llm_translate: translator={self.name} call={self.translate_call_count} "
            f"chars_in={in_len} rate_limit_param_keys={rl_keys}",
        )
        _translate_rate_limiter.wait()
        return self.do_llm_translate(text, rate_limit_params, response_schema)

    async def llm_translate_async(
        self, text, rate_limit_params: dict = None, response_schema=None
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
    def do_llm_translate(
        self, text, rate_limit_params: dict = None, response_schema=None
    ):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    @abstractmethod
    def do_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        logger.critical(
            f"Do not call BaseTranslator.do_translate. "
            f"Translator: {self}. "
            f"Text: {text}. ",
        )
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, placeholder_id: int | str):
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int | str):
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int | str):
        return self.get_rich_text_left_placeholder(placeholder_id)


class GeminiVertexAITranslator(BaseTranslator):
    """Translator backed by Google GenAI SDK (Vertex AI with service account)."""

    name = "gemini"

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        temperature=0.0,
    ):
        super().__init__(lang_in, lang_out)
        if genai is None:
            raise ImportError(
                "google-genai is required for Gemini translator. "
                "Install it with `uv add google-genai`."
            )

        self.model = model
        self.temperature = temperature
        self.client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT_ID,
            location=settings.GOOGLE_CLOUD_LOCATION,
        )
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

    def prompt(self, text: str) -> str:
        return build_translation_prompt(text, self.lang_in, self.lang_out)

    def _extract_text(self, response) -> str:
        parsed = getattr(response, "parsed", None)
        if parsed and isinstance(parsed, TranslationResponse):
            return parsed.translated_text

        text = getattr(response, "text", "")
        if text:
            text = text.strip()
            if text.startswith("{"):
                try:
                    import json

                    data = json.loads(text)
                    if isinstance(data, dict) and "translated_text" in data:
                        return str(data["translated_text"])
                except Exception:
                    pass
            return text
        return ""

    def _update_token_count(self, response) -> None:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
        if total_tokens:
            self.token_count.inc(total_tokens)
        if prompt_tokens:
            self.prompt_token_count.inc(prompt_tokens)
        if completion_tokens:
            self.completion_token_count.inc(completion_tokens)
        if cached_tokens:
            self.cache_hit_prompt_token_count.inc(cached_tokens)

    @llm_retry(logger=logger)
    def _generate_content_with_retry(
        self, *, model: str, contents: str, config: genai_types.GenerateContentConfig
    ):
        prompt_chars = len(contents)
        prompt_hash = hashlib.sha256(
            contents.encode("utf-8", errors="replace")
        ).hexdigest()[:12]
        prompt_preview = contents[:_MAX_CHARS_PROMPT_PREVIEW].replace("\n", "\\n")
        temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        max_output_tokens = int(getattr(config, "max_output_tokens", 0) or 0)
        logger.debug(
            f"Gemini generate_content: name={self.name} model={model} "
            f"prompt_chars={prompt_chars} prompt_hash={prompt_hash} "
            f"temperature={temperature} max_output_tokens={max_output_tokens} "
            f"prompt_preview={prompt_preview!r}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "gemini.generate_content",
            kind=SpanKind.CLIENT,
            attributes={
                _ATTR_LLM_NAME: model,
                _ATTR_LLM_MODEL: model,
                _ATTR_LLM_PROVIDER: LLMProvider.GEMINI_VERTEXAI,
                _ATTR_LLM_TEMPERATURE: temperature,
                _ATTR_LLM_PROMPT_CHARS: prompt_chars,
                _ATTR_LLM_PROMPT_HASH: prompt_hash,
                _ATTR_LLM_PROMPT_PREVIEW: prompt_preview,
                "llm.max_output_tokens": max_output_tokens,
            },
        ) as span:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                    output_tokens = int(
                        getattr(usage, "candidates_token_count", 0) or 0
                    )
                    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
                    cached_tokens = int(
                        getattr(usage, "cached_content_token_count", 0) or 0
                    )
                    span.set_attribute(_ATTR_LLM_INPUT_TOKENS, input_tokens)
                    span.set_attribute(_ATTR_LLM_OUTPUT_TOKENS, output_tokens)
                    span.set_attribute(_ATTR_LLM_TOTAL_TOKENS, total_tokens)
                    span.set_attribute("llm.cached_tokens", cached_tokens)
            except Exception as exc:
                span.record_exception(exc)
                from opentelemetry.trace import Status
                from opentelemetry.trace import StatusCode

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                elapsed = time.monotonic() - t0
                span.set_attribute(_ATTR_LLM_LATENCY_S, round(elapsed, 3))
                logger.debug(
                    f"Gemini generate_content finished: name={self.name} model={model} "
                    f"latency_s={elapsed:.3f} prompt_hash={prompt_hash}",
                )
        return response

    def do_translate(self, text, rate_limit_params: dict = None):
        translate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=TranslationResponse,
        )
        contents = self.prompt(text)
        c_len = len(contents)
        input_chars = len(text) if isinstance(text, str) else 0
        logger.debug(
            f"do_translate: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"input_chars={input_chars} prompt_chars={c_len} "
            f"temperature={self.temperature}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "llm.translate_batch",
            kind=SpanKind.CLIENT,
            attributes={
                _ATTR_LLM_NAME: self.model,
                _ATTR_LLM_MODEL: self.model,
                _ATTR_LLM_PROVIDER: LLMProvider.GEMINI_VERTEXAI,
                _ATTR_LLM_TEMPERATURE: float(self.temperature),
                _ATTR_LLM_INPUT_CHARS: input_chars,
                _ATTR_LLM_PROMPT_CHARS: c_len,
                "translation.source_lang": self.lang_in,
                "translation.target_lang": self.lang_out,
            },
        ) as span:
            response = self._generate_content_with_retry(
                model=self.model,
                contents=contents,
                config=translate_config,
            )
            self._update_token_count(response)
            out = self._extract_text(response)
            _usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(_usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(_usage, "candidates_token_count", 0) or 0)
            total_tokens = int(getattr(_usage, "total_token_count", 0) or 0)
            o_len = len(out)
            span.set_attribute(_ATTR_LLM_INPUT_TOKENS, input_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_TOKENS, output_tokens)
            span.set_attribute(_ATTR_LLM_TOTAL_TOKENS, total_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_CHARS, o_len)
        elapsed = time.monotonic() - t0
        span.set_attribute(_ATTR_LLM_LATENCY_S, round(elapsed, 3))
        # ── Bubble key facts up to the pipeline.run root span ─────────────
        set_root_span_attribute("llm.model", self.model)
        set_root_span_attribute("llm.name", self.model)
        if not out.strip() and c_len > 0:
            prev = contents[:_MAX_CHARS_LOG_PREVIEW].replace("\n", "\\n")
            logger.warning(
                f"do_translate empty model output: name={self.name} model={self.model} "
                f"prompt_chars={c_len} prompt_head={prev!r}",
            )
        logger.info(
            f"do_translate done: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"latency_s={elapsed:.3f} input_chars={input_chars} "
            f"prompt_chars={c_len} out_chars={o_len} "
            f"{_usage_metadata_summary(response)}",
        )
        logger.debug(
            f"do_translate totals: name={self.name} translator prompt_tokens="
            f"{self.prompt_token_count.value} completion_tokens="
            f"{self.completion_token_count.value}",
        )
        return out

    def do_llm_translate(
        self, text, rate_limit_params: dict = None, response_schema=None
    ):
        if text is None:
            logger.debug("do_llm_translate skipped: text is None")
            return None
        schema = response_schema if response_schema is not None else TranslationResponse
        translate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=schema,
        )
        contents = self.prompt(text)
        c_len = len(contents)
        input_chars = len(text) if isinstance(text, str) else 0
        rl_keys = sorted(rate_limit_params.keys()) if rate_limit_params else []
        logger.debug(
            f"do_llm_translate begin: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"input_chars={input_chars} prompt_chars={c_len} "
            f"temperature={self.temperature} "
            f"rate_limit_param_keys={rl_keys}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "llm.translate_batch",
            kind=SpanKind.CLIENT,
            attributes={
                _ATTR_LLM_NAME: self.model,
                _ATTR_LLM_MODEL: self.model,
                _ATTR_LLM_PROVIDER: LLMProvider.GEMINI_VERTEXAI,
                _ATTR_LLM_TEMPERATURE: float(self.temperature),
                _ATTR_LLM_INPUT_CHARS: input_chars,
                _ATTR_LLM_PROMPT_CHARS: c_len,
                "translation.source_lang": self.lang_in,
                "translation.target_lang": self.lang_out,
            },
        ) as span:
            response = self._generate_content_with_retry(
                model=self.model,
                contents=contents,
                config=translate_config,
            )
            self._update_token_count(response)
            out = self._extract_text(response)
            _usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(_usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(_usage, "candidates_token_count", 0) or 0)
            total_tokens = int(getattr(_usage, "total_token_count", 0) or 0)
            o_len = len(out)
            span.set_attribute(_ATTR_LLM_INPUT_TOKENS, input_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_TOKENS, output_tokens)
            span.set_attribute(_ATTR_LLM_TOTAL_TOKENS, total_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_CHARS, o_len)
        elapsed = time.monotonic() - t0
        span.set_attribute(_ATTR_LLM_LATENCY_S, round(elapsed, 3))
        # ── Bubble key facts up to the pipeline.run root span ─────────────
        set_root_span_attribute("llm.model", self.model)
        set_root_span_attribute("llm.name", self.model)
        if not out.strip() and c_len > 0:
            prev = contents[:_MAX_CHARS_LOG_PREVIEW].replace("\n", "\\n")
            logger.warning(
                f"do_llm_translate empty model output: name={self.name} model={self.model} "
                f"prompt_chars={c_len} prompt_head={prev!r}",
            )
        logger.info(
            f"do_llm_translate done: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"latency_s={elapsed:.3f} input_chars={input_chars} "
            f"prompt_chars={c_len} out_chars={o_len} "
            f"{_usage_metadata_summary(response)}",
        )
        logger.debug(
            f"do_llm_translate totals: name={self.name} translator prompt_tokens="
            f"{self.prompt_token_count.value} completion_tokens="
            f"{self.completion_token_count.value}",
        )
        return out


class ClaudeVertexAITranslator(BaseTranslator):
    """Translator backed by Anthropic Claude via Vertex AI Model Garden."""

    name = LLMProvider.CLAUDE

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        temperature=0.0,
    ):
        super().__init__(lang_in, lang_out)
        try:
            from anthropic import AnthropicVertex
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for Claude translator. "
                "Install it with `uv add anthropic[vertex]`."
            ) from exc

        self.model = model
        self.temperature = temperature
        self.client = AnthropicVertex(
            project_id=settings.GOOGLE_CLOUD_PROJECT_ID,
            region=settings.CLAUDE_VERTEX_REGION,
        )
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

    def prompt(self, text: str) -> str:
        return build_translation_prompt(text, self.lang_in, self.lang_out)

    def _build_tool_schema(self, response_schema) -> dict:
        """Convert a Pydantic model class to a Claude tool definition for structured output."""
        return {
            "name": "structured_output",
            "description": "Return the result in the required structured format.",
            "input_schema": response_schema.model_json_schema(),
        }

    def _update_token_count(self, usage) -> None:
        if not usage:
            return
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        total = input_tokens + output_tokens
        if total:
            self.token_count.inc(total)
        if input_tokens:
            self.prompt_token_count.inc(input_tokens)
        if output_tokens:
            self.completion_token_count.inc(output_tokens)
        if cache_read:
            self.cache_hit_prompt_token_count.inc(cache_read)

    def _extract_text(self, response, response_schema=None) -> str:
        """Extract a JSON string (structured) or plain text from a Claude response."""
        import json as _json

        if response_schema is not None:
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    return _json.dumps(block.input)
            logger.warning(
                f"Claude response missing tool_use block: name={self.name} "
                f"model={self.model} stop_reason={getattr(response, 'stop_reason', None)}",
            )
            return ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text.strip()
        return ""

    @llm_retry(logger=logger)
    def _messages_create_with_retry(self, *, contents: str, response_schema=None):
        prompt_chars = len(contents)
        prompt_hash = hashlib.sha256(
            contents.encode("utf-8", errors="replace")
        ).hexdigest()[:12]
        prompt_preview = contents[:_MAX_CHARS_PROMPT_PREVIEW].replace("\n", "\\n")
        temperature = float(self.temperature)
        schema_name = response_schema.__name__ if response_schema else None
        logger.debug(
            f"Claude messages.create: name={self.name} model={self.model} "
            f"prompt_chars={prompt_chars} prompt_hash={prompt_hash} "
            f"temperature={temperature} max_tokens={settings.LLM_MAX_OUTPUT_TOKENS} "
            f"schema={schema_name} prompt_preview={prompt_preview!r}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "claude.messages.create",
            kind=SpanKind.CLIENT,
            attributes={
                _ATTR_LLM_NAME: self.model,
                _ATTR_LLM_MODEL: self.model,
                _ATTR_LLM_PROVIDER: LLMProvider.CLAUDE,
                _ATTR_LLM_TEMPERATURE: temperature,
                _ATTR_LLM_PROMPT_CHARS: prompt_chars,
                _ATTR_LLM_PROMPT_HASH: prompt_hash,
                _ATTR_LLM_PROMPT_PREVIEW: prompt_preview,
                "llm.max_output_tokens": settings.LLM_MAX_OUTPUT_TOKENS,
            },
        ) as span:
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": settings.LLM_MAX_OUTPUT_TOKENS,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": contents}],
                }
                if response_schema is not None:
                    kwargs["tools"] = [self._build_tool_schema(response_schema)]
                    kwargs["tool_choice"] = {
                        "type": "tool",
                        "name": "structured_output",
                    }
                response = self.client.messages.create(**kwargs)
                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                    span.set_attribute(_ATTR_LLM_INPUT_TOKENS, input_tokens)
                    span.set_attribute(_ATTR_LLM_OUTPUT_TOKENS, output_tokens)
                    span.set_attribute(
                        _ATTR_LLM_TOTAL_TOKENS, input_tokens + output_tokens
                    )
            except Exception as exc:
                span.record_exception(exc)
                from opentelemetry.trace import Status
                from opentelemetry.trace import StatusCode

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                elapsed = time.monotonic() - t0
                span.set_attribute(_ATTR_LLM_LATENCY_S, round(elapsed, 3))
                logger.debug(
                    f"Claude messages.create finished: name={self.name} model={self.model} "
                    f"latency_s={elapsed:.3f} prompt_hash={prompt_hash}",
                )
        return response

    def do_translate(self, text, rate_limit_params: dict = None):
        import json as _json

        contents = self.prompt(text)
        c_len = len(contents)
        input_chars = len(text) if isinstance(text, str) else 0
        logger.debug(
            f"do_translate: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"input_chars={input_chars} prompt_chars={c_len} "
            f"temperature={self.temperature}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "llm.translate_batch",
            kind=SpanKind.CLIENT,
            attributes={
                _ATTR_LLM_NAME: self.model,
                _ATTR_LLM_MODEL: self.model,
                _ATTR_LLM_PROVIDER: LLMProvider.CLAUDE,
                _ATTR_LLM_TEMPERATURE: float(self.temperature),
                _ATTR_LLM_INPUT_CHARS: input_chars,
                _ATTR_LLM_PROMPT_CHARS: c_len,
                "translation.source_lang": self.lang_in,
                "translation.target_lang": self.lang_out,
            },
        ) as span:
            response = self._messages_create_with_retry(
                contents=contents,
                response_schema=TranslationResponse,
            )
            self._update_token_count(getattr(response, "usage", None))
            json_str = self._extract_text(response, TranslationResponse)
            _usage = getattr(response, "usage", None)
            input_tokens = int(getattr(_usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(_usage, "output_tokens", 0) or 0)
            try:
                out = _json.loads(json_str).get("translated_text", "")
            except Exception:
                out = json_str
            o_len = len(out)
            span.set_attribute(_ATTR_LLM_INPUT_TOKENS, input_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_TOKENS, output_tokens)
            span.set_attribute(_ATTR_LLM_TOTAL_TOKENS, input_tokens + output_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_CHARS, o_len)
        elapsed = time.monotonic() - t0
        span.set_attribute(_ATTR_LLM_LATENCY_S, round(elapsed, 3))
        set_root_span_attribute("llm.model", self.model)
        set_root_span_attribute("llm.name", self.model)
        if not out.strip() and c_len > 0:
            prev = contents[:_MAX_CHARS_LOG_PREVIEW].replace("\n", "\\n")
            logger.warning(
                f"do_translate empty model output: name={self.name} model={self.model} "
                f"prompt_chars={c_len} prompt_head={prev!r}",
            )
        logger.info(
            f"do_translate done: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"latency_s={elapsed:.3f} input_chars={input_chars} "
            f"prompt_chars={c_len} out_chars={o_len} "
            f"input_tokens={input_tokens} output_tokens={output_tokens}",
        )
        return out

    def do_llm_translate(
        self, text, rate_limit_params: dict = None, response_schema=None
    ):
        if text is None:
            logger.debug("do_llm_translate skipped: text is None")
            return None
        schema = response_schema if response_schema is not None else TranslationResponse
        contents = self.prompt(text)
        c_len = len(contents)
        input_chars = len(text) if isinstance(text, str) else 0
        rl_keys = sorted(rate_limit_params.keys()) if rate_limit_params else []
        logger.debug(
            f"do_llm_translate begin: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"input_chars={input_chars} prompt_chars={c_len} "
            f"temperature={self.temperature} "
            f"rate_limit_param_keys={rl_keys}",
        )
        t0 = time.monotonic()
        with tracer_llm.start_as_current_span(
            "llm.translate_batch",
            kind=SpanKind.CLIENT,
            attributes={
                _ATTR_LLM_NAME: self.model,
                _ATTR_LLM_MODEL: self.model,
                _ATTR_LLM_PROVIDER: LLMProvider.CLAUDE,
                _ATTR_LLM_TEMPERATURE: float(self.temperature),
                _ATTR_LLM_INPUT_CHARS: input_chars,
                _ATTR_LLM_PROMPT_CHARS: c_len,
                "translation.source_lang": self.lang_in,
                "translation.target_lang": self.lang_out,
            },
        ) as span:
            response = self._messages_create_with_retry(
                contents=contents,
                response_schema=schema,
            )
            self._update_token_count(getattr(response, "usage", None))
            out = self._extract_text(response, schema)
            _usage = getattr(response, "usage", None)
            input_tokens = int(getattr(_usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(_usage, "output_tokens", 0) or 0)
            o_len = len(out)
            span.set_attribute(_ATTR_LLM_INPUT_TOKENS, input_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_TOKENS, output_tokens)
            span.set_attribute(_ATTR_LLM_TOTAL_TOKENS, input_tokens + output_tokens)
            span.set_attribute(_ATTR_LLM_OUTPUT_CHARS, o_len)
        elapsed = time.monotonic() - t0
        span.set_attribute(_ATTR_LLM_LATENCY_S, round(elapsed, 3))
        set_root_span_attribute("llm.model", self.model)
        set_root_span_attribute("llm.name", self.model)
        if not out.strip() and c_len > 0:
            prev = contents[:_MAX_CHARS_LOG_PREVIEW].replace("\n", "\\n")
            logger.warning(
                f"do_llm_translate empty model output: name={self.name} model={self.model} "
                f"prompt_chars={c_len} prompt_head={prev!r}",
            )
        logger.info(
            f"do_llm_translate done: name={self.name} model={self.model} "
            f"{self.lang_in}->{self.lang_out} "
            f"latency_s={elapsed:.3f} input_chars={input_chars} "
            f"prompt_chars={c_len} out_chars={o_len} "
            f"input_tokens={input_tokens} output_tokens={output_tokens}",
        )
        logger.debug(
            f"do_llm_translate totals: name={self.name} translator prompt_tokens="
            f"{self.prompt_token_count.value} completion_tokens="
            f"{self.completion_token_count.value}",
        )
        return out
