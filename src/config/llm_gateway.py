"""Shared helper for routing google-genai / Anthropic Vertex `Client` calls
through the Apigee `llm` gateway instead of calling Vertex AI directly.

No workload service account in `gclt-aicoe-dev-st` may hold
`roles/aiplatform.user` (LLD decision D-30), so the Apigee `llm` gateway is the
only sanctioned path to Vertex AI from this project -- a direct call returns 403
regardless of what the code does. This helper is consulted at every LLM client
construction site so the gateway is configured in one place rather than
re-implemented per call site.

The URL contract -- read this before changing anything here
-----------------------------------------------------------
`base_url` on its own DOES NOT WORK, and fails in a way that looks like
something else entirely. `google-genai` joins `base_url` with the request path
only when `not custom_base_url or (project and location) or api_key`. With a
custom base_url and no project/location/api-key all three are false, so
`url = base_url` and **the entire path is discarded** -- the SDK POSTs to the
bare host. Apigee sees `proxy.pathsuffix == "/"`, no RouteRule matches, and its
deliberate `no-match` rule returns 404. Nothing about that 404 hints at a
client-side URL problem.

`base_url_resource_scope=ResourceScope.COLLECTION` takes the other branch and
joins the **unversioned** path, which is the shape the proxy expects. All three
of these are required together:

    project=None, location=None          (via gateway_vertex_identity_kwargs)
    base_url ending in /v1               (via gateway_base_url)
    base_url_resource_scope=COLLECTION   (via gateway_http_options_kwargs)

Remove any one and the request is silently mis-addressed. Verified against
google-genai 1.67.0 by building the request:

    https://llm.aicoedev-int.colt.net/v1/publishers/google/models/<model>:generateContent

Two consequences worth knowing. `api_version` is irrelevant under COLLECTION --
no version segment is appended, the `/v1` comes from our own base_url, which is
why the proxy's BasePath is `/v1`. And because a custom base_url makes the SDK
skip `load_auth()`, no ADC bearer token is minted: `x-apikey` is the only
credential on the wire, which is what the proxy's `VA-ApiKey` policy expects.
"""

from __future__ import annotations

from src.config.constants import settings
from src.config.llm_identity import current_llm_identity

# Canonical D2 header names (AICOE-Terraform docs/20 S0a, docs/23 S4.2). Do not
# invent new ones -- the `int` proxy already injects these downstream and the
# gateway's analytics keys on them. `x-colt-user-company` is deliberately absent:
# it is always Colt, so it carries no analytical value.
HEADER_USER_OID = "x-colt-user-oid"
HEADER_USER_DEPARTMENT = "x-colt-user-department"


def gateway_enabled() -> bool:
    """True when LLM traffic should be routed through Apigee."""
    return bool(settings.LLM_GATEWAY_ENABLED and settings.LLM_GATEWAY_BASE_URL)


def gateway_base_url() -> str:
    """The gateway base URL, normalised to exactly one trailing `/v1`.

    The proxy's BasePath is `/v1`, and under `ResourceScope.COLLECTION` the SDK
    appends no version segment of its own, so the `/v1` has to come from here.
    Normalising rather than requiring it means a payload that omits `/v1` (as
    the deployed one does) is not a silent 404, and one that includes it does
    not become `/v1/v1`. Idempotent by design.
    """
    base = str(settings.LLM_GATEWAY_BASE_URL or "").rstrip("/")
    if not base:
        return ""
    return base if base.endswith("/v1") else f"{base}/v1"


def _require_api_key() -> str:
    """The Apigee developer-app key, or a clear failure.

    An earlier version omitted `x-apikey` entirely when this setting was blank,
    so the gateway was called *unauthenticated* and rejected every request with
    no local signal at all. Failing here names the real cause instead. Despite
    the `_SECRET` suffix this setting holds the raw consumer key value, not a
    Secret Manager resource name.
    """
    key = str(settings.LLM_GATEWAY_API_KEY_SECRET or "").strip()
    if not key:
        raise ValueError(
            "LLM_GATEWAY_ENABLED is true but LLM_GATEWAY_API_KEY_SECRET is empty. "
            "The gateway would reject every call with 401. Set the Apigee "
            "developer-app key in this service's /secrets/.env payload, or set "
            "LLM_GATEWAY_ENABLED=false."
        )
    return key


def gateway_identity_headers() -> dict[str, str]:
    """Per-user attribution headers, or `{}` when nothing is in scope.

    Blank values are omitted rather than replaced with sentinels: Apigee's own
    `AM-Identity` policy applies documented `system` / `unattributed` defaults,
    and duplicating that decision here would give two places to disagree.

    Safe to read at client-construction time *in this service* because every
    LLM client here is built fresh inside a per-job or per-attempt call. Do not
    copy that assumption to a service with a cached or singleton client -- there
    the first job's identity would be frozen onto every later call.
    """
    identity = current_llm_identity()
    if identity is None:
        return {}
    headers: dict[str, str] = {}
    if identity.oid:
        headers[HEADER_USER_OID] = identity.oid
    if identity.department:
        headers[HEADER_USER_DEPARTMENT] = identity.department
    return headers


def gateway_http_options_kwargs() -> dict[str, object]:
    """Extra `genai_types.HttpOptions(...)` kwargs for the LLM gateway.

    Returns `{}` when the gateway is disabled, so callers can always write
    `HttpOptions(**base_kwargs, **gateway_http_options_kwargs())` with no
    conditional at the call site.
    """
    if not gateway_enabled():
        return {}

    # Imported lazily: this module is pulled in by the config layer, and the
    # API image has no reason to load the genai SDK at import time.
    from google.genai import types as genai_types

    return {
        "base_url": gateway_base_url(),
        "base_url_resource_scope": genai_types.ResourceScope.COLLECTION,
        "headers": {"x-apikey": _require_api_key(), **gateway_identity_headers()},
    }


def gateway_vertex_identity_kwargs(project: str, location: str) -> dict[str, object]:
    """The `project=`/`location=` kwargs for `genai.Client(vertexai=True, ...)`.

    **Read the module docstring before adding a new `genai.Client()` site.**
    When the gateway is on, both must be None. Passing either one makes
    google-genai prepend `projects/{project}/locations/{location}/` to the path
    -- this workload's own project and the infra region, neither of which is
    where inference runs -- and also re-enables ADC token minting. The proxy's
    TargetEndpoint supplies the real `projects/gclt-aicoe-dev-llm/locations/
    europe-west3` prefix instead, so no path-rewriting policy is needed.

    When the gateway is off this returns the given values unchanged, which is
    the direct-to-Vertex behaviour the existing tests assert.
    """
    if not gateway_enabled():
        return {"project": project, "location": location}
    return {"project": None, "location": None}


def gateway_anthropic_vertex_kwargs() -> dict[str, object]:
    """Extra `AnthropicVertex(...)` kwargs for the LLM gateway.

    A separate helper because `AnthropicVertex` takes `base_url` and
    `default_headers` directly rather than an `HttpOptions` object, and because
    it addresses Vertex differently: it ALWAYS embeds
    `projects/{project_id}/locations/{region}` in the path with no opt-out, and
    appends no version segment of its own. A base URL ending `/v1` is therefore
    exactly right for it too -- the same value google-genai needs under
    COLLECTION scope -- so both SDKs share one setting.

    `project_id` is overridden to the central inference project because this
    client sends whatever project it is given straight through to the proxy.

    UNVERIFIED, and currently unreachable by design: Claude is deliberately not
    on the gateway's model allow-list (AICOE-Terraform docs/23 D-C) and has been
    removed from `model_selection.json`, so nothing should route here. Apigee
    rejects a non-allow-listed model before any policy runs. `AnthropicVertex`
    also mints its own ADC token and sends `Authorization: Bearer <token>`;
    whether the proxy tolerates that alongside `x-apikey` has never been tested.
    """
    if not gateway_enabled():
        return {}
    kwargs: dict[str, object] = {
        "base_url": gateway_base_url(),
        "default_headers": {
            "x-apikey": _require_api_key(),
            **gateway_identity_headers(),
        },
    }
    if settings.LLM_GATEWAY_VERTEX_PROJECT:
        kwargs["project_id"] = settings.LLM_GATEWAY_VERTEX_PROJECT
    return kwargs
