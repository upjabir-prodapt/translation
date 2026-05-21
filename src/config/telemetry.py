"""OTel SDK initialization — must be called before setup_logging().

Exports traces to GCP Cloud Trace via the standard OTLP/HTTP endpoint
(telemetry.googleapis.com/v1/traces) using an AuthorizedSession for
ADC-based token refresh — the same pattern used by google-adk internally.

LLM call tracing is covered by two complementary layers:
  1. GoogleGenAiSdkInstrumentor (auto) — patches google-genai SDK and emits
     gen_ai.* semantic-convention spans with model, token usage, finish_reason,
     and (optionally) full prompt/response when
     OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true.
  2. Manual spans in translator.py / quality_judge_service.py — add
     llm.model, llm.temperature, llm.prompt_preview, llm.prompt_hash,
     llm.input_tokens, llm.output_tokens, llm.latency_s, etc.
"""

import logging
import os

import google.auth
from google.auth.transport.requests import AuthorizedSession
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import OTELResourceDetector
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

logger = logging.getLogger(__name__)

# GCP Cloud Trace OTLP/HTTP ingestion endpoint (protobuf over HTTPS)
# This is the same endpoint used internally by google-adk.
_GCP_TRACES_ENDPOINT = "https://telemetry.googleapis.com/v1/traces"


def setup_telemetry(app, settings) -> None:
    """Initialize OTel SDK, register instrumentors, and configure export to Cloud Trace.

    Must be called before setup_logging() so LoggingInstrumentor's record factory
    is installed before GcpJsonFormatter starts reading otelTraceID/otelSpanID.

    LLM call tracing highlights
    ---------------------------
    - GoogleGenAiSdkInstrumentor auto-instruments every call to
      client.models.generate_content(), emitting spans with:
        * gen_ai.system              ("google_genai" / "google_vertexai")
        * gen_ai.request.model       (e.g. "gemini-2.5-flash")
        * gen_ai.request.temperature
        * gen_ai.usage.input_tokens
        * gen_ai.usage.output_tokens
        * gen_ai.response.finish_reasons
        * gen_ai.prompt / gen_ai.completion  (only when
          OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true)
    - Manual spans in translator.py and quality_judge_service.py additionally
      record llm.model, llm.temperature, llm.prompt_preview (300 chars),
      llm.prompt_hash, llm.input_tokens, llm.output_tokens, llm.latency_s, etc.
    """
    if not settings.TRACE_ENABLED:
        return

    # ── 1. Resolve ADC credentials + project ─────────────────────────────────
    try:
        credentials, adc_project = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        logger.warning(
            "No ADC credentials found — OTel trace export to Cloud Trace disabled. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or run on GCP with a service account."
        )
        return

    project_id = settings.GOOGLE_CLOUD_PROJECT_ID or adc_project or ""
    if not project_id:
        logger.warning(
            "GOOGLE_CLOUD_PROJECT_ID not set — trace export may fail. "
            "Pass it as an env var or via Cloud Run --set-env-vars."
        )

    # ── 2. Build OTel Resource ────────────────────────────────────────────────
    # Priority: explicit env vars > OTEL_SERVICE_NAME > K_SERVICE > settings
    service_name = (
        os.environ.get("OTEL_SERVICE_NAME")
        or os.environ.get("K_SERVICE")
        or settings.OTEL_SERVICE_NAME
        or "translation_service"
    )
    service_version = (
        os.environ.get("COMMIT_SHA")
        or settings.APP_VERSION
        or "dev"
    )

    base_resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "gcp.project_id": project_id,
    })

    # Merge OTEL_RESOURCE_ATTRIBUTES env var, then Cloud Run auto-detected attrs
    # (K_SERVICE → cloud_run.service, K_REVISION → cloud_run.revision, etc.)
    resource = base_resource.merge(OTELResourceDetector().detect())
    try:
        from opentelemetry.resourcedetector.gcp_resource_detector import (
            GoogleCloudResourceDetector,
        )
        resource = resource.merge(
            GoogleCloudResourceDetector(raise_on_error=False).detect()
        )
    except Exception:
        logger.warning(
            "GCP resource detector unavailable; Cloud Run metadata will be missing "
            "from span resource attributes."
        )

    # ── 3. Build OTLP exporter with Google-signed HTTP session ────────────────
    # AuthorizedSession automatically refreshes the ADC access token before each
    # export batch, mirroring google-adk's _get_gcp_span_exporter() pattern.
    authorized_session = AuthorizedSession(credentials=credentials)
    exporter = OTLPSpanExporter(
        endpoint=_GCP_TRACES_ENDPOINT,
        session=authorized_session,
    )

    # ── 4. Register TracerProvider globally ───────────────────────────────────
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(settings.TRACE_SAMPLE_RATE)),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info(
        "TracerProvider configured → %s (project=%s service=%s version=%s)",
        _GCP_TRACES_ENDPOINT,
        project_id,
        service_name,
        service_version,
    )

    # ── 5. Instrumentation ────────────────────────────────────────────────────

    # 5a. Inject trace/span IDs into every log record so Cloud Logging can
    #     correlate log lines to their parent trace in Cloud Trace.
    #     MUST run before setup_logging() installs GcpJsonFormatter.
    LoggingInstrumentor().instrument(set_logging_format=False)

    # 5b. Auto-instrument every FastAPI HTTP request → one span per route.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="api/v1/health")

    # 5c. Auto-instrument all outbound HTTPX calls (e.g. Vertex AI REST calls).
    HTTPXClientInstrumentor().instrument()

    # 5d. Auto-instrument all google-genai SDK calls (generate_content, etc.).
    #     Emits gen_ai.* semantic-convention spans for every LLM invocation.
    #     Full prompt/response text is captured only when the env var
    #     OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true is set.
    try:
        from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
        GoogleGenAiSdkInstrumentor().instrument()
        logger.info(
            "GoogleGenAiSdkInstrumentor activated — gen_ai.* spans will appear "
            "nested under LLM call spans in Cloud Trace. "
            "Set OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true to also "
            "capture full prompt/response text on each span."
        )
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-google-genai is not installed; "
            "automatic LLM span generation is disabled. "
            "Add 'opentelemetry-instrumentation-google-genai>=0.7b1' to pyproject.toml."
        )
