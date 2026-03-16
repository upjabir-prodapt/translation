"""Worker service main application."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.logging import logger
from worker.core.lifecycle import WorkerLifecycle
from worker.middleware import RequestLoggingMiddleware
from worker.middleware import WorkerAuthMiddleware
from worker.middleware import exception_handler_middleware
from worker.routes import worker_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await WorkerLifecycle.startup()

    yield

    # Shutdown
    await WorkerLifecycle.shutdown()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    # Create FastAPI app
    app = FastAPI(
        title="BabelDOC Worker",
        description="Worker service for processing translation jobs",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add middleware (order matters - last added is executed first)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(WorkerAuthMiddleware)

    # Add global exception handling middleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=exception_handler_middleware)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, _exc: Exception):
        """Handle all unhandled exceptions."""
        logger.exception(f"Unhandled exception on {request.url.path}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    # Include routers
    app.include_router(worker_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")  # noqa: S104

    logger.info(f"Starting BabelDOC Worker on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,  # Use our custom logging
    )
