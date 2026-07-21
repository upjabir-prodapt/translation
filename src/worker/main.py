"""Worker FastAPI application — Cloud Tasks consumer for translation jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from datetime import UTC
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.worker.middleware.exception_handler import exception_handler_middleware
from src.config.constants import settings
from src.config.logging_config import setup_logging
from src.config.telemetry import setup_telemetry
from src.config.telemetry import shutdown_telemetry
from src.worker.routes.health import router as health_router
from src.worker.routes.tasks import router as tasks_router
from src.worker.services.startup_assets_service import StartupAssetsService
from src.worker.services.startup_assets_service import create_background_task

logger = logging.getLogger(__name__)

app_start_time = datetime.now(UTC)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Worker lifespan: telemetry + asset preflight/warmup."""
    otel_ready = setup_telemetry(_app, settings)
    setup_logging()
    if settings.TRACE_ENABLED:
        if otel_ready:
            logger.info(
                "[OTEL] Worker OpenTelemetry active (endpoint=%s)",
                settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            )
        else:
            logger.error("[OTEL] Worker OpenTelemetry failed to initialize")

    logger.info("Starting translation worker v%s", settings.API_VERSION)
    startup_assets = StartupAssetsService()
    _app.state.startup_preflight_status = None

    if settings.STARTUP_WARMUP_ENABLED:
        try:
            preflight_status = await asyncio.wait_for(
                startup_assets.run_preflight(),
                timeout=settings.STARTUP_PREFLIGHT_TIMEOUT_SECONDS,
            )
            _app.state.startup_preflight_status = preflight_status
            logger.info(
                "Worker assets preflight complete (critical_sync_ok=%s ready=%s)",
                preflight_status.critical_sync_ok,
                preflight_status.critical_files_ready,
            )
        except Exception as e:
            if settings.STARTUP_WARMUP_STRICT:
                logger.error("Worker assets preflight failed in strict mode")
                raise
            logger.warning("Worker assets preflight failed (continuing): %s", e)
    else:
        logger.info("Worker assets preflight disabled")

    if settings.STARTUP_BACKGROUND_WARMUP_ENABLED:
        _app.state.asset_background_warmup = create_background_task(
            startup_assets.run_background_warmup()
        )
        logger.info("Worker background asset warmup scheduled")
    else:
        logger.info("Worker background asset warmup disabled")

    logger.info("Worker startup complete")
    yield

    task: asyncio.Task | None = getattr(_app.state, "asset_background_warmup", None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    logger.info("Worker shutting down")
    shutdown_telemetry(_app)


app = FastAPI(
    title="Translation Worker",
    version=settings.API_VERSION,
    description="Cloud Tasks worker for PDF translation",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(BaseHTTPMiddleware, dispatch=exception_handler_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(tasks_router)


@app.get("/")
async def root():
    return {
        "service": "translation-worker",
        "version": settings.API_VERSION,
        "status": "running",
        "timestamp": datetime.now(UTC).isoformat(),
    }
