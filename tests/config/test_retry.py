import logging

import httpx
from src.config.retry import is_retryable_llm_exception
from src.config.retry import llm_retry


class TestRetryPolicy:
    def test_is_retryable_llm_exception_types(self):
        assert is_retryable_llm_exception(TimeoutError("timeout")) is True
        assert is_retryable_llm_exception(ConnectionError("conn")) is True
        assert is_retryable_llm_exception(httpx.ConnectTimeout("timeout")) is True
        assert is_retryable_llm_exception(ValueError("permanent")) is False

    def test_is_retryable_llm_exception_status_codes(self):
        class MockError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        assert is_retryable_llm_exception(MockError(429)) is True
        assert is_retryable_llm_exception(MockError(500)) is True
        assert is_retryable_llm_exception(MockError(404)) is False

    def test_is_retryable_llm_exception_substrings(self):
        assert is_retryable_llm_exception(Exception("Rate limit exceeded")) is True
        assert is_retryable_llm_exception(Exception("Deadline Exceeded")) is True
        assert is_retryable_llm_exception(Exception("Something else")) is False

    def test_llm_retry_decorator(self):
        logger = logging.getLogger("test")
        attempts = 0

        @llm_retry(logger=logger)
        def failing_func():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise TimeoutError("retry me")
            return "ok"

        assert failing_func() == "ok"
        assert attempts == 2
