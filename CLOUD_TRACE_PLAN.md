# Cloud Trace Implementation Plan — Translation API Service

## Background

Google now recommends **OpenTelemetry (OTel) over the legacy `google-cloud-trace` client library** for all new instrumentation. The preferred export path is OTLP → **Telemetry API** (`telemetry.googleapis.com`), not the older Cloud Trace v2 write API. This is vendor-neutral, future-proof, and supported on Cloud Run without a sidecar collector.

**Updates from review (2026-05-18):**
- Loguru removed in favour of Python's built-in `logging` module with a custom `GcpJsonFormatter`
- Log–span correlation wired via OTel `LoggingInstrumentor` + GCP structured JSON fields written to stdout

---

## Trace + Log Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Cloud Run — Translation API                         │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     OTel SDK Initialization Layer                      │  │
│  │                                                                        │  │
│  │  setup_telemetry()  [called first, before setup_logging()]             │  │
│  │  ├── TracerProvider ──► BatchSpanProcessor ──► OTLPSpanExporter        │  │
│  │  │   Resource: service.name, version, instance.id, cloud.region        │  │
│  │  │   Sampler: ParentBased(TraceIdRatioBased(1.0 → adjustable))         │  │
│  │  └── LoggingInstrumentor  ← injects otelTraceID/otelSpanID into every  │  │
│  │                              Python LogRecord via record factory        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                    │ spans                        │ enriched LogRecords       │
│         ┌──────────┼────────────────┐             │                          │
│         ▼          ▼                ▼             ▼                          │
│  ┌───────────┐ ┌────────┐ ┌──────────────┐ ┌──────────────────────────────┐ │
│  │  FastAPI  │ │  LLM   │ │  Repository  │ │   GcpJsonFormatter           │ │
│  │  Layer    │ │  Layer │ │  Layer       │ │                              │ │
│  │           │ │(Gemini)│ │  BQ + GCS    │ │  logging.getLogger(...)      │ │
│  │ Auto-inst │ │ Manual │ │  Manual      │ │  → JSON to stdout            │ │
│  │ via OTel  │ │ spans  │ │  spans       │ │  → Cloud Run agent collects  │ │
│  └─────┬─────┘ └───┬────┘ └──────┬───────┘ └──────────────┬───────────────┘ │
│        │            │             │                         │                 │
│        └────────────┼─────────────┘                        │                 │
│                     ▼                                       ▼                 │
│  ┌──────────────────────────────┐         ┌────────────────────────────────┐ │
│  │        Pipeline Layer        │         │         stdout (JSON)          │ │
│  │  Root span per job —         │         │  severity, message, time,      │ │
│  │  context propagated across   │         │  logging.googleapis.com/trace  │ │
│  │  asyncio background task     │         │  logging.googleapis.com/spanId │ │
│  └──────────────────────────────┘         │  logging.googleapis.com/       │ │
│                                           │    trace_sampled               │ │
└───────────────────┬───────────────────────┴──────────────┬─────────────────┘
                    │ OTLP/gRPC (443)                       │ stdout
                    ▼                                       ▼
        telemetry.googleapis.com              Cloud Run Log Agent
                    │                                       │
                    ▼                                       ▼
           Google Cloud Trace  ◄────── linked ──────  Cloud Logging
```

---

## Span Hierarchy Per Translation Job

```
[HTTP] POST /api/v1/translate                         ← FastAPI auto-instrumented
  └── translation_service.submit                      ← service span
        ├── bigquery.merge (insert job row)            ← repository span
        └── [link] pipeline.run (job_id=...)           ← linked span (async boundary)

[Background Task] pipeline.run
  ├── pipeline.language_detection
  │     └── llm.detect_language (if Gemini-based)
  ├── pipeline.glossary_load
  │     └── gcs.download (bucket, prefix, size_bytes)
  ├── pipeline.pdf_parse
  │     └── doclayout.onnx_inference (page_count, model_file)
  ├── pipeline.translation [attempt=1..N]
  │     ├── llm.translate_batch (batch_size, tokens, lang_pair, domain)
  │     │     └── gemini.generate_content (model, temperature, input_tokens, output_tokens, latency_ms)
  │     ├── pipeline.quality_judge
  │     │     └── llm.judge (model, alignment, omission, hallucination, pass=bool)
  │     └── pipeline.retry (if quality < threshold, attempt count)
  ├── pipeline.pdf_typeset
  ├── gcs.upload (output PDF, size_bytes)
  ├── bigquery.merge (update job → completed/human_review)
  └── [optional] dlp.scan
```

---

## Four Separate Tracer Namespaces

| Layer | Tracer Name | Instrumentation Type | Key Spans |
|---|---|---|---|
| **FastAPI** | `translation_api.http` | Auto via `FastAPIInstrumentor` + enriched middleware | HTTP method, route, status, user, org, job_id |
| **LLM** | `translation_api.llm` | Manual spans | `llm.translate_batch`, `llm.judge`, `gemini.generate_content`, `llm.retry` |
| **Pipeline** | `translation_api.pipeline` | Manual spans | `pipeline.run`, `pipeline.language_detection`, `pipeline.pdf_parse`, `pipeline.translation`, `pipeline.typeset` |
| **Repository** | `translation_api.repository` | Manual spans | `bigquery.merge`, `bigquery.get`, `bigquery.patch`, `gcs.upload`, `gcs.download`, `gcs.list` |

---

## Files to Create and Modify

```
src/
├── config/
│   ├── telemetry.py              ← NEW: OTel SDK setup, provider init, resource config,
│   │                                    LoggingInstrumentor (must init before setup_logging)
│   ├── tracing.py                ← NEW: get_tracer() helpers + @span() decorator
│   └── logging_config.py         ← REWRITE: drop Loguru entirely, add GcpJsonFormatter
│                                           writing structured JSON to stdout
├── api/
│   ├── main.py                   ← MODIFY: call setup_telemetry() then setup_logging()
│   │                                       in lifespan startup (ordering is critical)
│   ├── middleware/
│   │   └── trace_middleware.py   ← NEW: enrich FastAPI spans with user/org/job_id
│   └── services/
│       ├── translation_service.py      ← MODIFY: capture + pass trace context to background task
│       ├── pipeline_orchestrator.py    ← MODIFY: pipeline.run root span, context restore
│       ├── processor_service.py        ← MODIFY: pipeline child spans (parse, typeset)
│       └── quality_judge_service.py    ← MODIFY: llm.judge span
├── doctranslator/
│   └── translator/
│       └── translator.py         ← MODIFY: llm.translate_batch + gemini.generate_content spans
└── repository/
    ├── bigquery_repository.py    ← MODIFY: bigquery.* spans
    └── api_storage_repository.py ← MODIFY: gcs.* spans

pyproject.toml                    ← ADD OTel dependencies, REMOVE loguru dependency
```

---

## Dependencies

### Add to `pyproject.toml`

```toml
"opentelemetry-api>=1.27",
"opentelemetry-sdk>=1.27",
"opentelemetry-exporter-otlp-proto-grpc>=1.27",
"opentelemetry-instrumentation-fastapi>=0.48",
"opentelemetry-instrumentation-httpx>=0.48",
"opentelemetry-instrumentation-logging>=0.48",
"opentelemetry-resourcedetector-gcp>=1.9",
```

### Remove from `pyproject.toml`

```toml
"loguru>=0.7.3",   ← remove entirely; replaced by stdlib logging + GcpJsonFormatter
```

### Package Responsibilities

| Package | Purpose |
|---|---|
| `opentelemetry-api` | Core API — tracer, span, context interfaces |
| `opentelemetry-sdk` | SDK implementation — `TracerProvider`, `BatchSpanProcessor`, samplers |
| `opentelemetry-exporter-otlp-proto-grpc` | OTLP/gRPC exporter → `telemetry.googleapis.com` |
| `opentelemetry-instrumentation-fastapi` | Auto-instruments all HTTP routes |
| `opentelemetry-instrumentation-httpx` | Auto-instruments outbound HTTP (Gemini API calls) |
| `opentelemetry-instrumentation-logging` | Injects `otelTraceID`/`otelSpanID` into every Python `LogRecord` |
| `opentelemetry-resourcedetector-gcp` | Auto-populates Cloud Run metadata (region, instance ID) |

> `opentelemetry-resourcedetector-gcp` automatically populates `cloud.run.job.name`, `cloud.region`, and `service.instance.id` from the Cloud Run metadata server — no manual configuration needed for those fields.

---

## Implementation Details by Layer

### 1. OTel Initialization — `src/config/telemetry.py`

> **Must be called before `setup_logging()`** in `main.py` lifespan startup. `LoggingInstrumentor` registers its record factory here; if the formatter is attached first, the first log records during startup will have empty trace fields.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.resourcedetector.gcp_resource_detector import GoogleCloudResourceDetector

def setup_telemetry(app, settings) -> None:
    if not settings.TRACE_ENABLED:
        return

    gcp_resource = GoogleCloudResourceDetector().detect()
    base_resource = Resource.create({
        "service.name": "translation-api",
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
    # before the GcpJsonFormatter starts reading otelTraceID/otelSpanID
    LoggingInstrumentor().instrument(set_logging_format=False)

    FastAPIInstrumentor.instrument_app(app, excluded_urls="api/v1/health")
    HTTPXClientInstrumentor().instrument()
```

---

### 2. Logging — `src/config/logging_config.py` (full rewrite)

Loguru is removed. The new file has two responsibilities:

1. **`GcpJsonFormatter`** — formats every `LogRecord` as a JSON object that Cloud Logging understands natively, including the three trace-correlation fields populated by `LoggingInstrumentor`.
2. **`setup_logging()`** — attaches the formatter to a `StreamHandler(sys.stdout)` on the root logger.

The log-injection protection (stripping `\n` and `\r`) previously in `InterceptHandler` is preserved inside `GcpJsonFormatter.format()`.

```python
import json
import logging
import sys
import traceback
from datetime import datetime, timezone

from src.config.constants import settings


class GcpJsonFormatter(logging.Formatter):
    """Formats log records as structured JSON for Cloud Logging.

    Reads otelTraceID, otelSpanID, otelTraceSampled injected by
    LoggingInstrumentor and maps them to the Cloud Logging correlation fields:
      logging.googleapis.com/trace
      logging.googleapis.com/spanId
      logging.googleapis.com/trace_sampled
    """

    # Maps stdlib level names to GCP severity strings
    _SEVERITY = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def __init__(self, project_id: str) -> None:
        super().__init__()
        self._project_id = project_id

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize message — prevent log injection (CWE-117)
        message = record.getMessage().replace("\n", " ").replace("\r", " ")

        payload: dict = {
            "severity": self._SEVERITY.get(record.levelname, record.levelname),
            "message": message,
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "logger": record.name,
            "logging.googleapis.com/sourceLocation": {
                "file": record.filename,
                "line": str(record.lineno),
                "function": record.funcName,
            },
        }

        # Trace correlation fields — populated when a span is active.
        # otelTraceID and otelSpanID are injected by LoggingInstrumentor.
        # logging.googleapis.com/trace requires the full resource path.
        trace_id: str = getattr(record, "otelTraceID", "")
        span_id: str = getattr(record, "otelSpanID", "")
        sampled: bool = getattr(record, "otelTraceSampled", False)

        if trace_id:
            payload["logging.googleapis.com/trace"] = (
                f"projects/{self._project_id}/traces/{trace_id}"
            )
        if span_id:
            payload["logging.googleapis.com/spanId"] = span_id
        if trace_id or span_id:
            payload["logging.googleapis.com/trace_sampled"] = sampled

        # Attach exception info when present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text

        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure root logger with GcpJsonFormatter writing to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(GcpJsonFormatter(project_id=settings.GOOGLE_CLOUD_PROJECT_ID))

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    # Remove any handlers already attached (e.g. default stderr handler)
    root.handlers.clear()
    root.addHandler(handler)

    # Ensure uvicorn uses the same handler hierarchy
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        log = logging.getLogger(name)
        log.handlers.clear()
        log.propagate = True

    logging.getLogger(__name__).info("Logging configured")


# Auto-setup on import — same behaviour as previous Loguru-based config
setup_logging()
```

#### Emitted JSON (example)

```json
{
  "severity": "INFO",
  "message": "Translation job submitted successfully",
  "time": "2026-05-18T10:30:00.123456+00:00",
  "logger": "src.api.services.translation_service",
  "logging.googleapis.com/trace": "projects/my-gcp-project/traces/4bf92f3577b34da6a3ce929d0e0e4736",
  "logging.googleapis.com/spanId": "00f067aa0ba902b7",
  "logging.googleapis.com/trace_sampled": true,
  "logging.googleapis.com/sourceLocation": {
    "file": "translation_service.py",
    "line": "87",
    "function": "submit_translation"
  }
}
```

Cloud Logging reads `logging.googleapis.com/trace` and renders a **"View trace"** link in the log entry that opens the Cloud Trace waterfall for that exact span.

#### Why stdout over `CloudLoggingHandler`

| | `CloudLoggingHandler` | Structured JSON → stdout |
|---|---|---|
| Transport | Direct API calls (gRPC/HTTP) | Cloud Run agent collects automatically |
| Throughput | Slower — network I/O per batch | ~2× faster |
| Memory | Higher — thread pool + buffers | ~40% lower |
| Log loss on shutdown | Possible — unflushed buffer | None — streamed immediately |
| API quota | Consumes Cloud Logging write quota | Zero |
| Setup overhead | Requires explicit GCP client setup | Nothing extra on Cloud Run |

`CloudLoggingHandler` is appropriate for non-GCP deployments or very high-volume ingestion. For Cloud Run, **stdout is the correct and recommended choice**.

#### Impact on other files

**No other files need to change.** All 50+ source files outside `logging_config.py` already use:

```python
import logging
logger = logging.getLogger(__name__)
```

Loguru was only the sink/config layer. The `InterceptHandler` that routed stdlib → Loguru is deleted along with the class. The `logger.opt()` call inside it (the only Loguru-specific API in the codebase) disappears with it.

---

### 3. Tracer Helper — `src/config/tracing.py`

Provides named tracers and a `@span()` decorator for clean, low-boilerplate instrumentation across all layers.

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from functools import wraps
import inspect

def get_tracer(name: str):
    return trace.get_tracer(name)

# Named tracers — one per layer
tracer_http       = get_tracer("translation_api.http")
tracer_llm        = get_tracer("translation_api.llm")
tracer_pipeline   = get_tracer("translation_api.pipeline")
tracer_repository = get_tracer("translation_api.repository")

def span(tracer, name: str, kind: SpanKind = SpanKind.INTERNAL, attributes: dict = None):
    """Decorator that wraps a sync or async function in an OTel span."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as s:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    s.set_status(Status(StatusCode.ERROR, str(exc)))
                    s.record_exception(exc)
                    raise
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as s:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    s.set_status(Status(StatusCode.ERROR, str(exc)))
                    s.record_exception(exc)
                    raise
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

---

### 4. Startup Wiring — `src/api/main.py`

The ordering of `setup_telemetry` → `setup_logging` is mandatory. `LoggingInstrumentor` (called inside `setup_telemetry`) registers a log record factory. If the formatter runs before the factory is installed, the first log records will have empty `otelTraceID`/`otelSpanID` fields.

```python
# Inside lifespan() startup block, before yield
from src.config.telemetry import setup_telemetry
from src.config.logging_config import setup_logging

setup_telemetry(app, settings)   # step 1: install OTel + LoggingInstrumentor
setup_logging()                  # step 2: attach GcpJsonFormatter (reads otel fields)
```

---

### 5. FastAPI Layer

#### `src/api/middleware/trace_middleware.py` — Enrich auto-generated HTTP spans

FastAPI's auto-instrumentation creates spans but does not know about business context (user, job_id, organization). This middleware injects them as span attributes after routing.

```python
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace

class TraceEnrichmentMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        span = trace.get_current_span()
        user = request.state.__dict__.get("user")
        if user:
            span.set_attribute("user.email", user.email)
            span.set_attribute("user.organization", user.organization)
            span.set_attribute("user.business_unit", user.business_unit)
        job_id = request.path_params.get("job_id")
        if job_id:
            span.set_attribute("translation.job_id", job_id)
        return await call_next(request)
```

**Span attributes on every HTTP span:**

| Attribute | Value |
|---|---|
| `http.method` | `POST`, `GET`, etc. (auto) |
| `http.route` | `/api/v1/translate/{job_id}` (auto) |
| `http.status_code` | Response status (auto) |
| `user.email` | From JWT claim |
| `user.organization` | From JWT claim |
| `user.business_unit` | From JWT claim |
| `translation.job_id` | From path param (if present) |

---

### 6. Pipeline Layer

#### Async Context Propagation (critical design point)

FastAPI creates a trace context for the HTTP request. When `translation_service.py` schedules the background task via `asyncio.create_task()`, the OTel context is **not** automatically propagated. We must capture it explicitly and restore it inside the background task.

```python
# translation_service.py — capture context at submission time
from opentelemetry import context as otel_context

async def submit_translation(...):
    with tracer_pipeline.start_as_current_span("translation_service.submit") as span:
        # ... insert job to BigQuery ...
        span.set_attribute("translation.job_id", job_id)

        # Capture current trace context before leaving HTTP scope
        ctx = otel_context.get_current()
        asyncio.create_task(pipeline_orchestrator.run(job_id, ctx, ...))

# pipeline_orchestrator.py — restore context inside background task
from opentelemetry import context as otel_context

async def run(job_id: str, parent_ctx, ...):
    token = otel_context.attach(parent_ctx)
    try:
        with tracer_pipeline.start_as_current_span(
            "pipeline.run",
            attributes={"translation.job_id": job_id, ...}
        ):
            ...  # all child spans parented correctly
    finally:
        otel_context.detach(token)
```

**Pipeline span attributes:**

| Span | Key Attributes |
|---|---|
| `pipeline.run` | `job_id`, `source_lang`, `target_lang`, `domain`, `file_size_bytes` |
| `pipeline.language_detection` | `detected_lang`, `confidence` |
| `pipeline.glossary_load` | `domain`, `term_count`, `cache_hit` |
| `pipeline.pdf_parse` | `page_count`, `text_block_count`, `model_file` |
| `pipeline.translation` | `attempt`, `batch_count`, `total_tokens` |
| `pipeline.quality_judge` | `attempt`, `quality_score`, `passed` |
| `pipeline.retry` | `attempt`, `reason`, `previous_score` |
| `pipeline.pdf_typeset` | `page_count`, `output_size_bytes` |

---

### 7. LLM Layer

#### `src/doctranslator/translator/translator.py`

```python
from src.config.tracing import tracer_llm
from opentelemetry.trace import SpanKind

async def _translate_batch(self, batch: list[str], ...) -> list[str]:
    with tracer_llm.start_as_current_span(
        "llm.translate_batch",
        kind=SpanKind.CLIENT,
        attributes={
            "llm.model": self.model_name,
            "llm.provider": "google_vertexai",
            "translation.source_lang": source_lang,
            "translation.target_lang": target_lang,
            "translation.domain": domain,
            "llm.batch_size": len(batch),
            "llm.input_tokens": estimated_tokens,
        }
    ) as span:
        response = await self._call_gemini(prompt)
        span.set_attribute("llm.output_tokens", response.usage_metadata.candidates_token_count)
        span.set_attribute("llm.total_tokens", response.usage_metadata.total_token_count)
        return parsed_result
```

#### `src/api/services/quality_judge_service.py`

```python
with tracer_llm.start_as_current_span(
    "llm.judge",
    kind=SpanKind.CLIENT,
    attributes={
        "llm.model": self.judge_model,
        "translation.attempt": attempt_number,
    }
) as span:
    result = await self._run_judge(...)
    span.set_attribute("judge.alignment_score", result.alignment)
    span.set_attribute("judge.omission_score", result.omission)
    span.set_attribute("judge.hallucination_score", result.hallucination)
    span.set_attribute("judge.passed", result.score >= threshold)
```

**LLM span attributes (aligned with OTel Gen AI semantic conventions):**

| Attribute | Description |
|---|---|
| `llm.model` | Gemini model name (e.g. `gemini-2.5-flash`) |
| `llm.provider` | `google_vertexai` |
| `llm.temperature` | Sampling temperature |
| `llm.input_tokens` | Estimated input token count |
| `llm.output_tokens` | Actual output tokens from response metadata |
| `llm.total_tokens` | Total tokens billed |
| `llm.batch_size` | Number of paragraphs in the batch |
| `translation.source_lang` | e.g. `en` |
| `translation.target_lang` | e.g. `fr` |
| `translation.domain` | e.g. `legal`, `medical` |
| `translation.attempt` | Retry attempt number (1-based) |
| `judge.alignment_score` | 0.0–1.0 |
| `judge.omission_score` | 0.0–1.0 |
| `judge.hallucination_score` | 0.0–1.0 |
| `judge.passed` | Boolean — above quality threshold |

---

### 8. Repository Layer

#### `src/repository/bigquery_repository.py`

```python
from src.config.tracing import tracer_repository
from opentelemetry.trace import SpanKind

async def merge_job(self, job: TranslationJob) -> None:
    with tracer_repository.start_as_current_span(
        "bigquery.merge",
        kind=SpanKind.CLIENT,
        attributes={
            "db.system": "bigquery",
            "db.name": self.dataset,
            "db.sql.table": self.table,
            "db.operation": "MERGE",
            "translation.job_id": job.job_id,
        }
    ):
        await asyncio.to_thread(self._execute_merge, job)
```

#### `src/repository/api_storage_repository.py`

```python
with tracer_repository.start_as_current_span(
    "gcs.upload",
    kind=SpanKind.CLIENT,
    attributes={
        "rpc.system": "gcs",
        "gcs.bucket": self.bucket_name,
        "gcs.object": blob_path,
        "gcs.size_bytes": len(data),
        "translation.job_id": job_id,
    }
):
    blob.upload_from_string(data)
```

**Repository span attributes:**

| Span | Key Attributes |
|---|---|
| `bigquery.merge` | `db.system`, `db.name`, `db.sql.table`, `db.operation`, `job_id` |
| `bigquery.get` | Same + `row_found` (bool) |
| `bigquery.patch` | Same + `fields_updated` |
| `gcs.upload` | `gcs.bucket`, `gcs.object`, `gcs.size_bytes`, `job_id` |
| `gcs.download` | `gcs.bucket`, `gcs.object`, `gcs.size_bytes` |
| `gcs.list` | `gcs.bucket`, `gcs.prefix`, `gcs.result_count` |

---

## Log–Trace Correlation — How It Works End to End

```
1. setup_telemetry() is called
      └── LoggingInstrumentor().instrument() registers a log record factory
              with Python's logging framework

2. A span is active (e.g. inside pipeline.run)

3. Any logger.info("...") call anywhere in the app triggers the factory
      └── Factory reads the active OTel span from context
      └── Injects into the LogRecord:
            otelTraceID      = "4bf92f3577b34da6a3ce929d0e0e4736"  (hex)
            otelSpanID       = "00f067aa0ba902b7"                  (hex)
            otelTraceSampled = True

4. GcpJsonFormatter.format(record) runs
      └── Reads otelTraceID, otelSpanID, otelTraceSampled from record
      └── Builds JSON payload:
            "logging.googleapis.com/trace"
              → "projects/my-project/traces/4bf92f3577b34da6a3ce929d0e0e4736"
            "logging.googleapis.com/spanId"
              → "00f067aa0ba902b7"
            "logging.googleapis.com/trace_sampled"
              → true

5. JSON written to stdout → Cloud Run agent → Cloud Logging

6. Cloud Logging detects the correlation fields and adds a
   "View trace" link on every log entry, opening the Cloud Trace
   waterfall at the exact span that was active when the log was emitted
```

> When no span is active (e.g. during startup before the first request), `otelTraceID` and `otelSpanID` are empty strings. `GcpJsonFormatter` omits the correlation fields entirely in that case — no blank fields pollute the log entry.

---

## Environment Variables to Add

Add these to `src/config/constants.py` as `Pydantic BaseSettings` fields:

| Variable | Default | Description |
|---|---|---|
| `TRACE_ENABLED` | `true` | Master switch — set `false` in local dev to skip OTel init |
| `TRACE_SAMPLE_RATE` | `1.0` | Fraction of requests to trace (0.0–1.0) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `telemetry.googleapis.com:443` | Override to point at a local collector |

---

## IAM Permissions Required

Grant these roles to the Cloud Run service account:

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/telemetry.tracesWriter"

# Already likely set, confirm it:
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/logging.logWriter"
```

---

## GCP APIs to Enable

```bash
gcloud services enable \
  telemetry.googleapis.com \
  cloudtrace.googleapis.com
```

---

## Key Design Decisions

### Stdlib `logging` over Loguru
Only `logging_config.py` imported Loguru — all 50+ other source files already use `import logging` and `logging.getLogger(__name__)`. Loguru was solely the sink layer. Replacing it with a stdlib `StreamHandler` + `GcpJsonFormatter` requires touching exactly one file. No call sites change.

### Structured JSON to stdout over `CloudLoggingHandler`
On Cloud Run, stdout is the correct transport. The Cloud Run log agent collects stdout and forwards it to Cloud Logging with no extra setup, no API quota, no risk of log loss on container shutdown, and roughly 2× the throughput at 40% less RAM compared to `CloudLoggingHandler`. `CloudLoggingHandler` is reserved for non-GCP or very high-volume use cases.

### `logging.googleapis.com/trace` must use full resource path
The `otelTraceID` injected by `LoggingInstrumentor` is a bare hex string. Cloud Logging requires the full resource path `projects/{PROJECT_ID}/traces/{TRACE_ID}` for the correlation link to render. `GcpJsonFormatter` performs this mapping using `settings.GOOGLE_CLOUD_PROJECT_ID`.

### `LoggingInstrumentor` must run before `setup_logging()`
`LoggingInstrumentor` registers a log record factory on Python's logging framework. The factory must be in place before `GcpJsonFormatter` starts processing records. If `setup_logging()` runs first, startup log lines will have no trace fields.

### Async context propagation across the background task boundary
FastAPI's background task runs in the same event loop but OTel context does **not** automatically propagate into `asyncio.create_task()`. `otel_context.get_current()` is captured inside the HTTP request span and passed as an argument to the background task. Inside the task, `otel_context.attach()` restores it before `pipeline.run` is opened, making the entire pipeline tree a child of the originating HTTP span.

### Trace sampling
Start at `TraceIdRatioBased(1.0)` — 100% sampling. This service is job-based with low RPS and high value per trace. The rate is exposed as `TRACE_SAMPLE_RATE` so it can be reduced under load without a redeployment.

### Local development no-op
When `TRACE_ENABLED=false`, `setup_telemetry()` returns immediately. OTel's API guarantees that `trace.get_tracer(...)` and `start_as_current_span(...)` are safe no-ops when no provider is registered, so no guard clauses are needed in any instrumented code.

---

## Implementation Order

| Step | File(s) | Description |
|---|---|---|
| 1 | `pyproject.toml` | Add OTel dependencies; remove `loguru` |
| 2 | `src/config/constants.py` | Add `TRACE_ENABLED`, `TRACE_SAMPLE_RATE`, `OTEL_EXPORTER_OTLP_ENDPOINT` |
| 3 | `src/config/telemetry.py` | NEW — OTel SDK init, exporter, resource, `LoggingInstrumentor` |
| 4 | `src/config/tracing.py` | NEW — named tracers + `@span()` decorator |
| 5 | `src/config/logging_config.py` | REWRITE — drop Loguru, add `GcpJsonFormatter`, `setup_logging()` |
| 6 | `src/api/main.py` | Call `setup_telemetry()` then `setup_logging()` in lifespan startup |
| 7 | `src/api/middleware/trace_middleware.py` | NEW — HTTP span enrichment (user, org, job_id) |
| 8 | `src/api/services/translation_service.py` | Capture OTel context before background task |
| 9 | `src/api/services/pipeline_orchestrator.py` | Restore context + `pipeline.run` root span |
| 10 | `src/api/services/processor_service.py` | Child spans: `pipeline.pdf_parse`, `pipeline.typeset` |
| 11 | `src/api/services/quality_judge_service.py` | `llm.judge` span with quality score attributes |
| 12 | `src/doctranslator/translator/translator.py` | `llm.translate_batch` + `gemini.generate_content` spans |
| 13 | `src/repository/bigquery_repository.py` | `bigquery.*` spans |
| 14 | `src/repository/api_storage_repository.py` | `gcs.*` spans |
