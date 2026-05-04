import contextlib
import logging
import threading
import time
import unicodedata
from abc import ABC
from abc import abstractmethod

import httpx
import openai
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from src.doctranslator.doctranslator_exception.DocTranslatorException import ContentFilterError
from src.doctranslator.utils.atomic_integer import AtomicInteger
from src.config.constants import settings

logger = logging.getLogger(__name__)

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


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


_translate_rate_limiter = RateLimiter(5)


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
        _translate_rate_limiter.wait()
        return self.do_llm_translate(text, rate_limit_params)

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
            "You are a professional,authentic machine translation engine.\n"
            f";; Treat next line as plain text input and translate it into {self.lang_out}, "
            "output translation ONLY. If translation is unnecessary "
            "(e.g. proper nouns, codes, {{1}}, etc. ), return the original text. "
            "NO explanations. NO notes. Input:\n\n"
            f"{text}"
        )

    def _extract_text(self, response) -> str:
        text = getattr(response, "text", None)
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

    def do_translate(self, text, rate_limit_params: dict = None):
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.prompt(text),
            config={"temperature": self.temperature},
        )
        self._update_token_count(response)
        return self._extract_text(response)

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None
        response = self.client.models.generate_content(
            model=self.model,
            contents=text,
            config={"temperature": self.temperature},
        )
        self._update_token_count(response)
        return self._extract_text(response)
