"""Claude Vertex AI translator provider."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config.constants import settings
from src.config.retry import llm_retry
from src.doctranslator.translator.base import BaseTranslator
from src.doctranslator.translator.instrumentation import ATTR_LLM_MODEL
from src.doctranslator.translator.instrumentation import ATTR_LLM_NAME
from src.doctranslator.translator.instrumentation import ATTR_LLM_PROMPT_CHARS
from src.doctranslator.translator.instrumentation import ATTR_LLM_PROMPT_HASH
from src.doctranslator.translator.instrumentation import ATTR_LLM_PROMPT_PREVIEW
from src.doctranslator.translator.instrumentation import ATTR_LLM_PROVIDER
from src.doctranslator.translator.instrumentation import ATTR_LLM_TEMPERATURE
from src.doctranslator.translator.instrumentation import instrumented_llm_call
from src.doctranslator.translator.instrumentation import prompt_fingerprint
from src.doctranslator.translator.prompts import build_translation_prompt
from src.doctranslator.translator.provider_types import LLMProvider
from src.doctranslator.translator.schemas import TranslationResponse
from src.doctranslator.translator.usage import TokenUsage
from src.doctranslator.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)


class ClaudeVertexAITranslator(BaseTranslator):
    """Translator backed by Anthropic Claude via Vertex AI Model Garden."""

    name = LLMProvider.CLAUDE
    provider = LLMProvider.CLAUDE

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
            project_id=settings.GOOGLE_CLOUD_PROJECT,
            region=settings.CLAUDE_VERTEX_REGION,
        )
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

    def prompt(self, text: str) -> str:
        return build_translation_prompt(text, self.lang_in, self.lang_out)

    def _build_tool_schema(self, response_schema) -> dict:
        return {
            "name": "structured_output",
            "description": "Return the result in the required structured format.",
            "input_schema": response_schema.model_json_schema(),
        }

    def extract_text(self, response, response_schema=None) -> str:
        if response_schema is not None:
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    payload = block.input
                    if response_schema is TranslationResponse and isinstance(
                        payload, dict
                    ):
                        return str(payload.get("translated_text", ""))
                    return json.dumps(payload)
            logger.warning(
                f"Claude response missing tool_use block: name={self.name} "
                f"model={self.model} stop_reason={getattr(response, 'stop_reason', None)}",
            )
            return ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text.strip()
        return ""

    def extract_usage(self, response) -> TokenUsage:
        return TokenUsage.from_claude_usage(getattr(response, "usage", None))

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
    def _messages_create_with_retry(self, *, contents: str, response_schema=None):
        prompt_chars, prompt_hash, prompt_preview = prompt_fingerprint(contents)
        temperature = float(self.temperature)
        schema_name = response_schema.__name__ if response_schema else None
        logger.debug(
            f"Claude messages.create: name={self.name} model={self.model} "
            f"prompt_chars={prompt_chars} prompt_hash={prompt_hash} "
            f"temperature={temperature} max_tokens={settings.LLM_MAX_OUTPUT_TOKENS} "
            f"schema={schema_name} prompt_preview={prompt_preview!r}",
        )

        def _call():
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
            return self.client.messages.create(**kwargs)

        return instrumented_llm_call(
            span_name="claude.messages.create",
            attributes={
                ATTR_LLM_NAME: self.model,
                ATTR_LLM_MODEL: self.model,
                ATTR_LLM_PROVIDER: LLMProvider.CLAUDE,
                ATTR_LLM_TEMPERATURE: temperature,
                ATTR_LLM_PROMPT_CHARS: prompt_chars,
                ATTR_LLM_PROMPT_HASH: prompt_hash,
                ATTR_LLM_PROMPT_PREVIEW: prompt_preview,
                "llm.max_output_tokens": settings.LLM_MAX_OUTPUT_TOKENS,
            },
            call=_call,
            usage_fn=lambda resp: TokenUsage.from_claude_usage(
                getattr(resp, "usage", None)
            ),
            log_prefix=f"Claude messages.create model={self.model}",
        )

    def invoke(self, contents: str, response_schema: Any | None = None) -> Any:
        schema = response_schema if response_schema is not None else TranslationResponse
        return self._messages_create_with_retry(
            contents=contents,
            response_schema=schema,
        )
