"""Main FastAPI application."""

import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import UTC
from datetime import datetime

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware.exception_handler import exception_handler_middleware
from src.api.routes.router import api_router
from src.api.services.startup_assets_service import StartupAssetsService
from src.api.services.startup_assets_service import create_background_task
from src.config.constants import settings
from src.config.logging import logger

# Track startup time
app_start_time = datetime.now(UTC)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
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
                f"Startup assets preflight complete (critical_sync_ok={preflight_status.critical_sync_ok} ready={preflight_status.critical_files_ready})",
                preflight_status.critical_sync_ok,
                preflight_status.critical_files_ready,
            )
        except Exception as e:
            if settings.STARTUP_WARMUP_STRICT:
                logger.error("Startup assets preflight failed in strict mode")
                raise
            logger.warning(f"Startup assets preflight failed (continuing): {e}")
    else:
        logger.info("Startup assets preflight disabled by STARTUP_WARMUP_ENABLED=false")

    if settings.STARTUP_BACKGROUND_WARMUP_ENABLED:
        _app.state.asset_background_warmup = create_background_task(
            startup_assets.run_background_warmup()
        )
        logger.info("Background asset warmup scheduled (non-blocking)")
    else:
        logger.info(
            "Background asset warmup disabled by STARTUP_BACKGROUND_WARMUP_ENABLED=false"
        )

    logger.info("API startup complete and ready to serve traffic")

    yield

    # Shutdown
    task: asyncio.Task | None = getattr(_app.state, "asset_background_warmup", None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    logger.info("API shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="PDF Translation API using DocTranslator",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add global exception handling middleware
app.add_middleware(BaseHTTPMiddleware, dispatch=exception_handler_middleware)

# Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)


# Root endpoint
@app.get("/")
async def root(request: Request):
    """Root endpoint."""
    preflight_status = getattr(request.app.state, "startup_preflight_status", None)
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "timestamp": datetime.now(UTC).isoformat(),
        "startup_preflight": (
            {
                "critical_sync_ok": preflight_status.critical_sync_ok,
                "critical_files_ready": preflight_status.critical_files_ready,
                "assets_root": preflight_status.assets_root,
            }
            if preflight_status
            else None
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )  # noqa: S104
