"""Claude must refuse loudly while the gateway is on, not fail opaquely at Apigee."""

from __future__ import annotations

import pytest

from src.config.constants import settings
from src.worker.doctranslator.translator.providers.claude import ClaudeVertexAITranslator


def test_claude_refuses_while_gateway_enabled(monkeypatch: pytest.MonkeyPatch):
    """Claude is not on the gateway's model allow-list (docs/23 D-C).

    Apigee rejects a non-allow-listed model during credential/operation
    matching, before any policy in the proxy runs, so the failure would surface
    deep inside a translation attempt with nothing pointing at the model. The
    constructor names the real cause instead.
    """
    monkeypatch.setattr(settings, "LLM_GATEWAY_ENABLED", True, raising=False)
    monkeypatch.setattr(
        settings, "LLM_GATEWAY_BASE_URL", "https://llm.aicoedev-int.colt.net", raising=False
    )
    with pytest.raises(ValueError, match="allow-list"):
        ClaudeVertexAITranslator("en", "de", "claude-sonnet-4-6")


def test_claude_still_constructs_when_gateway_disabled(monkeypatch: pytest.MonkeyPatch):
    """The guard is about the gateway, not about Claude being unusable."""
    monkeypatch.setattr(settings, "LLM_GATEWAY_ENABLED", False, raising=False)
    with monkeypatch.context() as m:
        created: dict[str, object] = {}

        class _FakeAnthropicVertex:
            def __init__(self, **kwargs):
                created.update(kwargs)

        import anthropic

        m.setattr(anthropic, "AnthropicVertex", _FakeAnthropicVertex)
        translator = ClaudeVertexAITranslator("en", "de", "claude-sonnet-4-6")

    assert translator.model == "claude-sonnet-4-6"
    # Direct-to-Vertex: no gateway base_url or api key injected.
    assert "base_url" not in created
    assert "default_headers" not in created
