"""OTel SDK initialization — must be called before setup_logging().

Exports traces to GCP Cloud Trace via the standard OTLP/HTTP endpoint
(telemetry.googleapis.com/v1/traces) using an AuthorizedSession for
ADC-based token refresh — the same pattern used by google-adk internally.

LLM call tracing is covered by two complementary layers:
  1. GoogleGenAiSdkInstrumentor (auto) — patches google-genai SDK and emits
     gen_ai.* semantic-convention spans with model, token usage, finish_reason,
     and (optionally) full prompt/response when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
     is set in .env (e.g. EVENT_ONLY with OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental).
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

_LOG_PREFIX = "[OTEL]"

_VALID_CAPTURE_MODES = frozenset(
    {"", "NO_CONTENT", "SPAN_ONLY", "EVENT_ONLY", "SPAN_AND_EVENT"}
)


def _otel_log_failure(
    reason: str, *, hint: str | None = None, exc: BaseException | None = None
) -> None:
    """Emit a grep-friendly ERROR for OTel startup/export failures."""
    message = f"{_LOG_PREFIX} FAILED: {reason}"
    if hint:
        message = f"{message} — {hint}"
    if exc is not None:
        logger.error(message, exc_info=exc)
    else:
        logger.error(message)


def _set_otel_state(app, *, enabled: bool) -> None:
    if app is not None:
        app.state.otel_enabled = enabled


def _log_otel_content_capture_config(settings) -> None:
    """Validate and log GenAI content-capture settings from .env at startup."""
    semconv = settings.OTEL_SEMCONV_STABILITY_OPT_IN
    capture_mode = settings.OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
    effective = (
        capture_mode.upper()
        if capture_mode.upper() in _VALID_CAPTURE_MODES
        else "NO_CONTENT"
    )
    if capture_mode.upper() not in _VALID_CAPTURE_MODES:
        logger.warning(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=%r is invalid "
            "(valid: %s) — effective capture mode is NO_CONTENT",
            capture_mode,
            ", ".join(sorted(_VALID_CAPTURE_MODES - {""})),
        )
    experimental = "gen_ai_latest_experimental" in semconv
    logger.info(
        "OTEL content capture: OTEL_SEMCONV_STABILITY_OPT_IN=%r "
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=%r (effective=%s) "
        "experimental_semconv=%s",
        semconv,
        capture_mode,
        effective,
        experimental,
    )
    if effective in ("EVENT_ONLY", "SPAN_AND_EVENT") and not experimental:
        logger.warning(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=%r requires "
            "OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental",
            capture_mode,
        )


def setup_telemetry(app, settings) -> bool:
    """Initialize OTel SDK, register instrumentors, and configure export to Cloud Trace.

    Must be called before setup_logging() so LoggingInstrumentor's record factory
    is installed before GcpJsonFormatter starts reading otelTraceID/otelSpanID.

    Returns True when trace export to Cloud Trace is active, False otherwise.
    """
    if not settings.TRACE_ENABLED:
        logger.info("%s Disabled — TRACE_ENABLED=false", _LOG_PREFIX)
        _set_otel_state(app, enabled=False)
        return False

    _log_otel_content_capture_config(settings)
    otlp_endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT.strip().rstrip("/")
    logger.info(
        "%s Initializing export to %s (service=%s)",
        _LOG_PREFIX,
        otlp_endpoint,
        settings.OTEL_SERVICE_NAME,
    )

    try:
        # ── 1. Resolve ADC credentials + project ─────────────────────────────
        try:
            credentials, adc_project = google.auth.default()
        except google.auth.exceptions.GoogleAuthError as exc:
            _otel_log_failure(
                "Application Default Credentials unavailable — cannot authenticate "
                "to Cloud Trace",
                hint=(
                    "Set GOOGLE_APPLICATION_CREDENTIALS locally, or run on Cloud Run "
                    "with a service account. Ensure runtime image has no http_proxy "
                    "env vars blocking the metadata server."
                ),
                exc=exc,
            )
            _set_otel_state(app, enabled=False)
            return False

        project_id = settings.GOOGLE_CLOUD_PROJECT or adc_project or ""
        if not project_id:
            _otel_log_failure(
                "GOOGLE_CLOUD_PROJECT is empty",
                hint=(
                    "Set GOOGLE_CLOUD_PROJECT in .env or Cloud Run --set-env-vars; "
                    "ADC did not resolve a default project."
                ),
            )
            _set_otel_state(app, enabled=False)
            return False

        # ── 2. Build OTel Resource ─────────────────────────────────────────
        service_name = (
            os.environ.get("OTEL_SERVICE_NAME")
            or os.environ.get("K_SERVICE")
            or settings.OTEL_SERVICE_NAME
            or "translation_service"
        )
        service_version = os.environ.get("COMMIT_SHA") or settings.APP_VERSION or "dev"

        base_resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
                "gcp.project_id": project_id,
            }
        )

        resource = base_resource.merge(OTELResourceDetector().detect())
        try:
            from opentelemetry.resourcedetector.gcp_resource_detector import (
                GoogleCloudResourceDetector,
            )

            resource = resource.merge(
                GoogleCloudResourceDetector(raise_on_error=False).detect()
            )
        except Exception as exc:
            logger.warning(
                "%s GCP resource detector unavailable — Cloud Run metadata may be "
                "missing from spans: %s",
                _LOG_PREFIX,
                exc,
            )

        # ── 3. Build OTLP exporter with Google-signed HTTP session ─────────
        authorized_session = AuthorizedSession(credentials=credentials)
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            session=authorized_session,
        )

        # ── 4. Register TracerProvider globally ────────────────────────────
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(root=TraceIdRatioBased(settings.TRACE_SAMPLE_RATE)),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # ── 5. Instrumentation ─────────────────────────────────────────────
        LoggingInstrumentor().instrument(set_logging_format=False)
        FastAPIInstrumentor.instrument_app(app, excluded_urls="api/v1/health")
        HTTPXClientInstrumentor().instrument()

        try:
            from opentelemetry.instrumentation.google_genai import (
                GoogleGenAiSdkInstrumentor,
            )

            GoogleGenAiSdkInstrumentor().instrument()
            logger.info(
                "%s GoogleGenAiSdkInstrumentor active — capture mode=%r",
                _LOG_PREFIX,
                settings.OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
            )
        except ImportError:
            logger.warning(
                "%s GoogleGenAiSdkInstrumentor not installed — LLM auto-spans disabled "
                "(add opentelemetry-instrumentation-google-genai>=0.7b1)",
                _LOG_PREFIX,
            )

        logger.info(
            "%s Ready — exporting traces to %s (project=%s service=%s version=%s)",
            _LOG_PREFIX,
            otlp_endpoint,
            project_id,
            service_name,
            service_version,
        )
        _set_otel_state(app, enabled=True)
        return True

    except Exception as exc:
        _otel_log_failure(
            "OpenTelemetry setup failed unexpectedly",
            hint="See stack trace below; app continues without Cloud Trace export.",
            exc=exc,
        )
        _set_otel_state(app, enabled=False)
        return False


def shutdown_telemetry(app=None) -> None:
    """Flush and shut down the TracerProvider before process exit."""
    if app is not None and not getattr(app.state, "otel_enabled", False):
        logger.info("%s Shutdown skipped — telemetry was not initialized", _LOG_PREFIX)
        return

    provider = trace.get_tracer_provider()
    if not hasattr(provider, "shutdown"):
        logger.warning(
            "%s Shutdown skipped — no SDK TracerProvider registered", _LOG_PREFIX
        )
        return

    try:
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=30_000)
        provider.shutdown()
        logger.info("%s TracerProvider shut down — buffered spans flushed", _LOG_PREFIX)
    except Exception as exc:
        _otel_log_failure(
            "TracerProvider shutdown/flush failed — some spans may be lost",
            exc=exc,
        )
