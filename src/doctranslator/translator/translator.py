import asyncio
import contextlib
import logging
import threading
import time
import unicodedata
from abc import ABC
from abc import abstractmethod

from google import genai
from google.genai import types as genai_types

from src.config.constants import settings
from src.config.retry import llm_retry
from src.doctranslator.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)

_MAX_CHARS_LOG_PREVIEW = 120


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

    def llm_translate(self, text, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
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
        return self.do_llm_translate(text, rate_limit_params)

    async def llm_translate_async(self, text, rate_limit_params: dict = None):
        in_len = len(text) if isinstance(text, str) else 0
        logger.debug(
            f"llm_translate_async: translator={self.name} scheduling thread "
            f"chars_in={in_len}",
        )
        return await asyncio.to_thread(self.llm_translate, text, rate_limit_params)

    @abstractmethod
    def do_llm_translate(self, text, rate_limit_params: dict = None):
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
        return (
            "# Role\n"
            "You are an expert document translator: accurate, idiomatic, and faithful to the source.\n\n"
            "# Task\n"
            f"Translate the INPUT below from {self.lang_in} into {self.lang_out}.\n\n"
            "# Output format (plain text only)\n"
            "- Reply with nothing except the translated text itself.\n"
            "- Do not add explanations, notes, alternatives, or apologies—only the translation.\n"
            "- Preserve line breaks and paragraph boundaries as in the INPUT unless the target language "
            "requires a minimal, justified adjustment.\n\n"
            "# Alignment, coverage, and fidelity\n"
            "A strong translation stays structurally aligned with the source, omits nothing important, "
            "and hallucinates nothing.\n"
            "- Alignment: keep lists, numbered steps, table rows, and paragraph chunks in clear one-to-one "
            "correspondence with the INPUT; do not merge unrelated bullets, drop list items, scramble step order "
            "when order matters, or turn one coherent source sentence into several unrelated sentences (or the reverse) "
            "unless normal for "
            f"{self.lang_out} and the logic of the source is preserved.\n"
            "- Coverage (avoid omission): translate every substantive phrase, fact, obligation, condition, and "
            "negation; do not skip clauses, soften requirements, or leave meaning behind.\n"
            "- Fidelity (avoid hallucination): do not add commentary, disclaimers, hedging, examples, or details "
            "not in the source; do not invent names, dates, numbers, or causal claims.\n\n"
            "# Register, headings, and capitalization\n"
            "- Match the source register (e.g. legal, technical, marketing, UI) and keep tone consistent.\n"
            "- Preserve intentional capitalization: ALL‑CAPS headings, Title Case section titles, "
            "sentence case, and small caps patterns—mirror them in the target language when natural; "
            "if the target language uses different heading conventions, choose one clear style and "
            "apply it consistently across the segment.\n"
            "- Keep emphasis patterns implied by capitalization (e.g. acronyms vs. ordinary words) correct.\n\n"
            "# Terminology and consistency\n"
            "- Use one stable choice per technical term, product name pattern, and key entity within the segment.\n"
            "- Prefer established target‑language equivalents for domain terms; do not mix synonyms casually.\n\n"
            "# What not to translate (copy verbatim)\n"
            "- Placeholders and markup: e.g. {{1}}, {{v2}}, tokens like <b3>…</b3>, URLs, file paths, "
            "email addresses, hex/color codes, version strings, and pure numeric or alphanumeric codes.\n"
            "- Widely recognized trademarks and person names when localization would be wrong; "
            "otherwise follow normal target‑language usage.\n"
            "- If a substring is already correct and natural in the target language (e.g. a lone symbol, "
            "a code, or a no‑translate token), return it unchanged.\n\n"
            "# Numbers, dates, and units\n"
            "- Keep mathematical or identifier numbers exact unless the source clearly expects localization.\n"
            "- For dates, currencies, and units, use the conventional form for "
            f"{self.lang_out} when unambiguous; otherwise preserve the source form.\n\n"
            "# Quality bar\n"
            "- Even when you rephrase idiomatically for "
            f"{self.lang_out}, honor the alignment, coverage, and fidelity rules above.\n"
            "- Preserve negations, conditions, quantities, and legal or technical qualifiers exactly in force; "
            "do not silently soften or strengthen them.\n\n"
            "# INPUT\n"
            f"{text}"
            "# Output\n"
        )

    def _extract_text(self, response) -> str:
        text = getattr(response, "text", "")
        if text:
            return text.strip()
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
        logger.debug(
            f"Gemini generate_content: model={model} contents_chars={len(contents)} "
            f"max_output_tokens={getattr(config, 'max_output_tokens', None)} "
            f"temperature={getattr(config, 'temperature', None)}",
        )
        t0 = time.monotonic()
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        finally:
            elapsed = time.monotonic() - t0
            logger.debug(
                f"Gemini generate_content finished: model={model} latency_s={elapsed:.3f}",
            )
        return response

    def do_translate(self, text, rate_limit_params: dict = None):
        translate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        )
        contents = self.prompt(text)
        c_len = len(contents)
        logger.debug(
            f"do_translate: model={self.model} {self.lang_in}->{self.lang_out} "
            f"prompt_chars={c_len}",
        )
        t0 = time.monotonic()
        response = self._generate_content_with_retry(
            model=self.model,
            contents=contents,
            config=translate_config,
        )
        self._update_token_count(response)
        out = self._extract_text(response)
        elapsed = time.monotonic() - t0
        o_len = len(out)
        if not out.strip() and c_len > 0:
            prev = contents[:_MAX_CHARS_LOG_PREVIEW].replace("\n", "\\n")
            logger.warning(
                f"do_translate empty model output: model={self.model} "
                f"prompt_chars={c_len} prompt_head={prev!r}",
            )
        logger.info(
            f"do_translate done: model={self.model} {self.lang_in}->{self.lang_out} "
            f"latency_s={elapsed:.3f} prompt_chars={c_len} out_chars={o_len} "
            f"{_usage_metadata_summary(response)}",
        )
        logger.debug(
            f"do_translate totals: translator prompt_tokens="
            f"{self.prompt_token_count.value} completion_tokens="
            f"{self.completion_token_count.value}",
        )
        return out

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            logger.debug("do_llm_translate skipped: text is None")
            return None
        translate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        )
        contents = self.prompt(text)
        c_len = len(contents)
        rl_keys = sorted(rate_limit_params.keys()) if rate_limit_params else []
        logger.debug(
            f"do_llm_translate begin: model={self.model} {self.lang_in}->{self.lang_out} "
            f"prompt_chars={c_len} rate_limit_param_keys={rl_keys}",
        )
        t0 = time.monotonic()
        response = self._generate_content_with_retry(
            model=self.model,
            contents=contents,
            config=translate_config,
        )
        self._update_token_count(response)
        out = self._extract_text(response)
        elapsed = time.monotonic() - t0
        o_len = len(out)
        if not out.strip() and c_len > 0:
            prev = contents[:_MAX_CHARS_LOG_PREVIEW].replace("\n", "\\n")
            logger.warning(
                f"do_llm_translate empty model output: model={self.model} "
                f"prompt_chars={c_len} prompt_head={prev!r}",
            )
        logger.info(
            f"do_llm_translate done: model={self.model} {self.lang_in}->{self.lang_out} "
            f"latency_s={elapsed:.3f} prompt_chars={c_len} out_chars={o_len} "
            f"{_usage_metadata_summary(response)}",
        )
        logger.debug(
            f"do_llm_translate totals: translator prompt_tokens="
            f"{self.prompt_token_count.value} completion_tokens="
            f"{self.completion_token_count.value}",
        )
        return out
