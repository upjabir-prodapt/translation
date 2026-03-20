"""Main FastAPI application."""

from contextlib import asynccontextmanager
from datetime import UTC
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.middleware.exception_handler import exception_handler_middleware
from api.routes.router import api_router
from config.constants import settings
from config.logging import logger

# Track startup time
app_start_time = datetime.now(UTC)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("API shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="PDF Translation API using BabelDOC",
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
async def root():
    """Root endpoint."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)  # noqa: S104
