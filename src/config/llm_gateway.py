"""Shared helper for routing google-genai / Anthropic Vertex `Client` calls
through the Apigee `llm` gateway instead of calling Vertex AI directly.

No workload service account in `gclt-aicoe-dev-st` may hold
`roles/aiplatform.user` (LLD decision D-30), so the Apigee `llm` gateway is
the only sanctioned path to Vertex AI from this project. This helper is
consulted at every LLM client construction site so the gateway can be
enabled/disabled from one place (`settings.LLM_GATEWAY_ENABLED`) rather than
re-implemented per call site.

Defaults to disabled: the `llm` proxy does not exist yet
(`AICOE-Terraform/terraform/GAP-REGISTER.md` R-08), and whether it works at
all for this service's traffic shape has not been verified -- see
`AICOE-Terraform/docs/21-apigee-llm-gateway-manual-setup-guide.md`.
"""

from __future__ import annotations

from src.config.constants import settings


def gateway_http_options_kwargs() -> dict[str, object]:
    """Extra `genai_types.HttpOptions(...)` kwargs for the LLM gateway.

    Returns `{}` when the gateway is disabled or misconfigured, so callers
    can always do `HttpOptions(**base_kwargs, **gateway_http_options_kwargs())`
    without a conditional at the call site.
    """
    if not settings.LLM_GATEWAY_ENABLED or not settings.LLM_GATEWAY_BASE_URL:
        return {}
    kwargs: dict[str, object] = {"base_url": settings.LLM_GATEWAY_BASE_URL}
    if settings.LLM_GATEWAY_API_KEY_SECRET:
        kwargs["headers"] = {"x-apikey": settings.LLM_GATEWAY_API_KEY_SECRET}
    return kwargs


def gateway_vertex_identity_kwargs(project: str, location: str) -> dict[str, object]:
    """The `project=`/`location=` kwargs to pass to `genai.Client(vertexai=True, ...)`.

    **Read this before adding a new `genai.Client()` call site.** `base_url`
    alone is not enough: `google-genai` builds each request path as
    `projects/{project}/locations/{location}/models/{model}:generateContent`
    whenever `project`/`location` are set, custom `base_url` or not (verified
    empirically against the installed SDK -- `base_url` itself is honored
    either way, but the *path* still embeds whatever project/location you
    pass). Passing this workload's own project (`gclt-aicoe-dev-st`) would
    send that project's id in the path to the gateway, requiring Apigee to
    rewrite it to the real inference project (`gclt-aicoe-dev-llm`) before
    forwarding to Vertex.

    Simpler and what this build does instead: when the gateway is enabled,
    omit project/location entirely. The SDK then sends a bare
    `models/{model}:generateContent` path with no project prefix, and the
    Apigee target endpoint's own fixed `.../projects/gclt-aicoe-dev-llm/
    locations/{region}` prefix supplies the real (correct, central) project
    and location -- no path-rewriting policy needed in the proxy. When the
    gateway is disabled, this returns the given values unchanged (today's
    direct-to-Vertex behavior, verified by the existing test suite).
    """
    if not settings.LLM_GATEWAY_ENABLED or not settings.LLM_GATEWAY_BASE_URL:
        return {"project": project, "location": location}
    return {"project": None, "location": None}


def gateway_anthropic_vertex_kwargs() -> dict[str, object]:
    """Extra `AnthropicVertex(...)` kwargs for the LLM gateway.

    `AnthropicVertex` (the Claude-via-Vertex-Model-Garden client) does not
    accept a `genai_types.HttpOptions` object like `google-genai` does -- it
    takes `base_url` and `default_headers` directly, so this is a separate
    helper. It also behaves differently from `google-genai` on the
    project-in-path question: `AnthropicVertex` always embeds `project_id` in
    the request path regardless of `base_url` (there is no "omit project and
    let the gateway supply it" mode like `google-genai` has). So when the
    gateway is enabled, this ALSO overrides `project_id` to
    `settings.LLM_GATEWAY_VERTEX_PROJECT` (the central inference project,
    `gclt-aicoe-dev-llm`) -- without this override the path would carry this
    workload's own project id instead.

    Returns `{}` when the gateway is disabled or misconfigured.

    NOTE: `AnthropicVertex` still mints its own Google access token via ADC
    on every call (see `_ensure_access_token` in the `anthropic` package) and
    sends it as `Authorization: Bearer <token>`. Whether the gateway's caller
    identity check (docs/09 SS22.8 step 2) actually accepts that token, and
    whether Claude/Anthropic calls work through Apigee at all, is
    UNVERIFIED -- there is no `llm` proxy deployed yet to test against
    (GAP-REGISTER R-08). Treat this wiring as unverified until WS-E1/E4 land.
    """
    if not settings.LLM_GATEWAY_ENABLED or not settings.LLM_GATEWAY_BASE_URL:
        return {}
    kwargs: dict[str, object] = {"base_url": settings.LLM_GATEWAY_BASE_URL}
    if settings.LLM_GATEWAY_API_KEY_SECRET:
        kwargs["default_headers"] = {"x-apikey": settings.LLM_GATEWAY_API_KEY_SECRET}
    if settings.LLM_GATEWAY_VERTEX_PROJECT:
        kwargs["project_id"] = settings.LLM_GATEWAY_VERTEX_PROJECT
    return kwargs
