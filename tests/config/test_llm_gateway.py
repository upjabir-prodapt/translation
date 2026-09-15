"""Contract tests for the Apigee LLM gateway wiring.

The centrepiece is `test_request_url_is_the_allow_listed_resource`. The gateway
enforces a model allow-list by matching the exact request path, and Apigee does
that matching BEFORE any policy in the proxy runs -- so a wrong path is rejected
with a credential-shaped error that looks nothing like a URL problem. This
module asserts the full URL string so that failure mode is caught here instead.

It is also the tripwire for a google-genai upgrade. The correct URL depends on
three things holding together (project/location both None, a base_url ending
/v1, and ResourceScope.COLLECTION); an SDK change to any of them silently
re-breaks the path, which is exactly what happened before this test existed.
"""

from __future__ import annotations

import pytest
from src.config.constants import settings
from src.config.llm_gateway import HEADER_USER_DEPARTMENT
from src.config.llm_gateway import HEADER_USER_OID
from src.config.llm_gateway import gateway_anthropic_vertex_kwargs
from src.config.llm_gateway import gateway_base_url
from src.config.llm_gateway import gateway_enabled
from src.config.llm_gateway import gateway_http_options_kwargs
from src.config.llm_gateway import gateway_identity_headers
from src.config.llm_gateway import gateway_vertex_identity_kwargs
from src.config.llm_identity import current_llm_identity
from src.config.llm_identity import use_llm_identity

GATEWAY_HOST = "https://llm.aicoedev-int.colt.net"
MODEL_PATH = "publishers/google/models/gemini-3.5-flash:generateContent"
EXPECTED_URL = f"{GATEWAY_HOST}/v1/{MODEL_PATH}"


@pytest.fixture
def gateway_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "LLM_GATEWAY_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "LLM_GATEWAY_BASE_URL", GATEWAY_HOST, raising=False)
    monkeypatch.setattr(
        settings, "LLM_GATEWAY_API_KEY_SECRET", "TESTKEY", raising=False
    )
    return settings


@pytest.fixture
def gateway_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "LLM_GATEWAY_ENABLED", False, raising=False)
    return settings


def _build_url_and_headers():
    """Build a real request exactly as a call site would, offline."""
    from google.genai import types as genai_types
    from google.genai._api_client import BaseApiClient

    client = BaseApiClient(
        vertexai=True,
        **gateway_vertex_identity_kwargs(settings.GOOGLE_CLOUD_PROJECT, "europe-west3"),
        http_options=genai_types.HttpOptions(**gateway_http_options_kwargs()),
    )
    request = client._build_request("post", MODEL_PATH, {}, None)
    headers = {k.lower(): v for k, v in (request.headers or {}).items()}
    return request.url, headers


@pytest.mark.usefixtures("gateway_on")
def test_request_url_is_the_allow_listed_resource():
    url, _ = _build_url_and_headers()
    assert url == EXPECTED_URL


@pytest.mark.usefixtures("gateway_on")
def test_api_key_is_sent_and_no_adc_token_is_minted():
    """`x-apikey` must be the only credential.

    A custom base_url makes google-genai skip `load_auth()`, so no Google ADC
    bearer token is minted. That matters: the proxy authenticates with
    `VA-ApiKey`, and an unexpected `Authorization` header would be a second,
    unasked-for credential leaving this service.
    """
    _, headers = _build_url_and_headers()
    assert headers["x-apikey"] == "TESTKEY"
    assert "authorization" not in headers


@pytest.mark.usefixtures("gateway_on")
def test_identity_headers_are_sent_from_context():
    with use_llm_identity("OID-123", "Network Engineering"):
        _, headers = _build_url_and_headers()
    assert headers[HEADER_USER_OID] == "OID-123"
    assert headers[HEADER_USER_DEPARTMENT] == "Network Engineering"


@pytest.mark.usefixtures("gateway_on")
def test_company_header_is_never_sent():
    """Company is always Colt, so it carries no analytical value (docs/23 S4.2)."""
    with use_llm_identity("OID-123", "Network Engineering"):
        _, headers = _build_url_and_headers()
    assert "x-colt-user-company" not in headers


@pytest.mark.usefixtures("gateway_on")
def test_identity_headers_are_omitted_when_no_user_is_in_scope():
    """Omit rather than invent a sentinel.

    Apigee's own AM-Identity policy defaults absent headers to
    `system`/`unattributed`. Sending our own sentinel would give two places to
    disagree about what "no user" means.
    """
    assert current_llm_identity() is None
    assert gateway_identity_headers() == {}
    _, headers = _build_url_and_headers()
    assert HEADER_USER_OID not in headers
    assert HEADER_USER_DEPARTMENT not in headers


@pytest.mark.usefixtures("gateway_on")
def test_blank_identity_values_are_omitted():
    with use_llm_identity("", "   "):
        assert gateway_identity_headers() == {}


@pytest.mark.usefixtures("gateway_on")
def test_identity_does_not_leak_between_scopes():
    """Cloud Run reuses containers, so a job must not inherit the previous user."""
    with use_llm_identity("OID-first", "Alpha"):
        assert gateway_identity_headers()[HEADER_USER_OID] == "OID-first"
    assert gateway_identity_headers() == {}
    with use_llm_identity("OID-second", "Beta"):
        assert gateway_identity_headers()[HEADER_USER_OID] == "OID-second"


@pytest.mark.parametrize(
    "configured",
    [GATEWAY_HOST, f"{GATEWAY_HOST}/", f"{GATEWAY_HOST}/v1", f"{GATEWAY_HOST}/v1/"],
)
@pytest.mark.usefixtures("gateway_on")
def test_base_url_normalisation_is_idempotent(monkeypatch, configured):
    """Never `/v1/v1`, and never a missing `/v1`.

    The proxy BasePath is `/v1` and COLLECTION scope appends no version segment,
    so the `/v1` must come from this setting. Normalising means a payload that
    omits it is not a silent 404.
    """
    monkeypatch.setattr(settings, "LLM_GATEWAY_BASE_URL", configured, raising=False)
    assert gateway_base_url() == f"{GATEWAY_HOST}/v1"
    url, _ = _build_url_and_headers()
    assert url == EXPECTED_URL


@pytest.mark.usefixtures("gateway_on")
def test_project_and_location_are_forced_to_none():
    """Passing either one re-prefixes the path and re-enables ADC token minting."""
    assert gateway_vertex_identity_kwargs("gclt-aicoe-dev-st", "europe-west1") == {
        "project": None,
        "location": None,
    }


@pytest.mark.usefixtures("gateway_off")
def test_disabled_gateway_is_a_full_passthrough():
    assert gateway_enabled() is False
    assert gateway_http_options_kwargs() == {}
    assert gateway_anthropic_vertex_kwargs() == {}
    assert gateway_vertex_identity_kwargs("proj", "europe-west1") == {
        "project": "proj",
        "location": "europe-west1",
    }


@pytest.mark.usefixtures("gateway_on")
def test_missing_api_key_fails_loudly(monkeypatch):
    """Previously the key was silently omitted and every call 401'd.

    An unauthenticated call to the gateway fails with no local signal at all,
    which is strictly worse than refusing to build the client.
    """
    monkeypatch.setattr(settings, "LLM_GATEWAY_API_KEY_SECRET", "", raising=False)
    with pytest.raises(ValueError, match="LLM_GATEWAY_API_KEY_SECRET"):
        gateway_http_options_kwargs()
    with pytest.raises(ValueError, match="LLM_GATEWAY_API_KEY_SECRET"):
        gateway_anthropic_vertex_kwargs()


@pytest.mark.usefixtures("gateway_on")
def test_anthropic_kwargs_share_the_versioned_base_url(monkeypatch):
    """AnthropicVertex appends no version segment and always embeds the project.

    So a base URL ending `/v1` is right for it too -- both SDKs share one
    setting -- and `project_id` must be overridden to the central inference
    project, since this client sends whatever it is given straight through.
    """
    monkeypatch.setattr(
        settings, "LLM_GATEWAY_VERTEX_PROJECT", "gclt-aicoe-dev-llm", raising=False
    )
    with use_llm_identity("OID-9", "Ops"):
        kwargs = gateway_anthropic_vertex_kwargs()
    assert kwargs["base_url"] == f"{GATEWAY_HOST}/v1"
    assert kwargs["project_id"] == "gclt-aicoe-dev-llm"
    headers = kwargs["default_headers"]
    assert headers["x-apikey"] == "TESTKEY"
    assert headers[HEADER_USER_OID] == "OID-9"
