"""OTel SDK initialization — must be called before setup_logging()."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased


def setup_telemetry(app, settings) -> None:
    """Initialize OTel SDK, register LoggingInstrumentor, and auto-instrument FastAPI/HTTPX.

    Must be called before setup_logging() so LoggingInstrumentor's record factory
    is installed before GcpJsonFormatter starts reading otelTraceID/otelSpanID.
    """
    if not settings.TRACE_ENABLED:
        return

    try:
        from opentelemetry.resourcedetector.gcp_resource_detector import (
            GoogleCloudResourceDetector,
        )
        gcp_resource = GoogleCloudResourceDetector().detect()
    except Exception:
        gcp_resource = Resource.get_empty()

    import os

    service_name = (
        os.environ.get("OTEL_SERVICE_NAME")
        or os.environ.get("K_SERVICE")
        or settings.OTEL_SERVICE_NAME
        or "translation_service"
    )

    base_resource = Resource.create({
        "service.name": service_name,
        "service.version": settings.APP_VERSION,
    })
    resource = base_resource.merge(gcp_resource)

    exporter = OTLPSpanExporter(
        endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        headers={"x-goog-user-project": settings.GOOGLE_CLOUD_PROJECT_ID},
    )

    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(settings.TRACE_SAMPLE_RATE)),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Must run before setup_logging() so the record factory is in place
    # before the GcpJsonFormatter starts reading otelTraceID/otelSpanID.
    LoggingInstrumentor().instrument(set_logging_format=False)

    FastAPIInstrumentor.instrument_app(app, excluded_urls="api/v1/health")
    HTTPXClientInstrumentor().instrument()
