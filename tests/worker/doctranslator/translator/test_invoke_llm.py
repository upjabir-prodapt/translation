"""The one place slot + retry + instrumentation are composed.

These three primitives already existed but were wired up by hand at each
call site, and the judge only ever wired two of them: it retried and traced
but never acquired a slot. Harmless at one call per attempt; not harmless
once the judge fans out over chunks alongside the translator.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.doctranslator.translator import invoke as invoke_module
from src.worker.doctranslator.translator.invoke import invoke_llm


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    from src.config import retry as retry_module

    monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MULTIPLIER", 0)
    monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MIN_SECONDS", 0)
    monkeypatch.setattr(retry_module.settings, "LLM_RETRY_MAX_SECONDS", 0)


def test_returns_the_call_result():
    assert invoke_llm(lambda: "ok", span_name="llm.test") == "ok"


def test_acquires_an_inflight_slot():
    """The budget is process-wide and shared with the translator, so a
    second uncapped fan-out would blow straight through it."""
    slot = MagicMock()
    with patch.object(invoke_module, "llm_call_slot", return_value=slot):
        invoke_llm(lambda: "ok", span_name="llm.test")
    slot.__enter__.assert_called_once()
    slot.__exit__.assert_called_once()


def test_transient_failures_are_retried():
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("deadline exceeded")
        return "ok"

    assert invoke_llm(_flaky, span_name="llm.test") == "ok"
    assert calls["n"] == 3


def test_non_transient_failures_propagate_immediately():
    calls = {"n": 0}

    def _broken():
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        invoke_llm(_broken, span_name="llm.test")
    assert calls["n"] == 1


def test_retry_on_widens_the_policy():
    """A malformed structured response looks like a plain ValueError to the
    generic predicate, but is transient in practice: the same prompt usually
    parses on the next sample."""

    class ParseError(RuntimeError):
        pass

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ParseError("unparseable")
        return "ok"

    assert invoke_llm(_flaky, span_name="llm.test", retry_on=(ParseError,)) == "ok"
    assert calls["n"] == 2


def test_slot_is_released_between_backoff_sleeps():
    """The slot is acquired inside the retry, not outside it.

    A call sleeping through exponential backoff must not hold one of the
    LLM_MAX_INFLIGHT_CALLS slots while doing nothing.
    """
    enters = {"n": 0}
    slot = MagicMock()
    slot.__enter__.side_effect = lambda: enters.__setitem__("n", enters["n"] + 1)
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("deadline exceeded")
        return "ok"

    with patch.object(invoke_module, "llm_call_slot", return_value=slot):
        invoke_llm(_flaky, span_name="llm.test")
    assert enters["n"] == 3
    assert slot.__exit__.call_count == 3
