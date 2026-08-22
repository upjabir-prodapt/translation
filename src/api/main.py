"""Main FastAPI application (public API — no asset warmup)."""

import logging
from contextlib import asynccontextmanager
from datetime import UTC
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware.exception_handler import exception_handler_middleware
from src.api.middleware.trace_middleware import TraceEnrichmentMiddleware
from src.api.routes.router import api_router
from src.config.constants import settings
from src.config.logging_config import setup_logging
from src.config.telemetry import setup_telemetry
from src.config.telemetry import shutdown_telemetry

logger = logging.getLogger(__name__)

app_start_time = datetime.now(UTC)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """API lifespan — telemetry only (assets live on the worker)."""
    otel_ready = setup_telemetry(_app, settings)
    setup_logging()
    if settings.TRACE_ENABLED:
        if otel_ready:
            logger.info(
                "[OTEL] OpenTelemetry active — traces export to Cloud Trace "
                "(endpoint=%s)",
                settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            )
        else:
            logger.error(
                "[OTEL] OpenTelemetry failed to initialize — running WITHOUT trace "
                "export to Cloud Trace. Search logs for '[OTEL] FAILED' for the "
                "root cause (ADC, project ID, or setup error)."
            )
    logger.info("Starting %s v%s (API)", settings.API_TITLE, settings.API_VERSION)
    logger.info("API startup complete (asset warmup skipped — worker-owned)")
    yield
    logger.info("API shutting down")
    shutdown_telemetry(_app)


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="PDF Translation API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(BaseHTTPMiddleware, dispatch=exception_handler_middleware)
app.add_middleware(TraceEnrichmentMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "role": "api",
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
