"""Instrumentation helper tests."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.doctranslator.translator.instrumentation import instrumented_llm_call
from src.worker.doctranslator.translator.instrumentation import prompt_fingerprint
from src.worker.doctranslator.translator.usage import TokenUsage


class TestInstrumentation:
    def test_prompt_fingerprint(self):
        chars, digest, preview = prompt_fingerprint("hello\nworld")
        assert chars == 11
        assert len(digest) == 12
        assert "\\n" in preview

    @patch("src.worker.doctranslator.translator.instrumentation.tracer_llm")
    def test_instrumented_llm_call_records_usage(self, mock_tracer):
        span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = span

        result = instrumented_llm_call(
            span_name="test.span",
            attributes={"llm.model": "gemini-2.5-flash"},
            call=lambda: {"ok": True},
            usage_fn=lambda _: TokenUsage(input_tokens=10, output_tokens=5),
        )
        assert result == {"ok": True}
        span.set_attribute.assert_any_call("llm.input_tokens", 10)
        span.set_attribute.assert_any_call("llm.output_tokens", 5)

    @patch("src.worker.doctranslator.translator.instrumentation.tracer_llm")
    def test_instrumented_llm_call_records_exception(self, mock_tracer):
        span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = span

        with pytest.raises(RuntimeError, match="boom"):
            instrumented_llm_call(
                span_name="test.span",
                attributes={},
                call=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            )
        span.record_exception.assert_called_once()
