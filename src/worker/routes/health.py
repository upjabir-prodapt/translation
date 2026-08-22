"""Unauthenticated health endpoint for Cloud Run probes."""

from datetime import UTC
from datetime import datetime

from fastapi import APIRouter

from src.config.constants import settings

router = APIRouter()


@router.get("/healthz", tags=["health"])
async def health_check():
    """Liveness/readiness for Cloud Run (no OIDC)."""
    from src.worker.main import app_start_time

    uptime = (datetime.now(UTC) - app_start_time).total_seconds()
    return {
        "status": "healthy",
        "service": "translation-worker",
        "version": settings.API_VERSION,
        "uptime_seconds": uptime,
    }
