"""Health check endpoints."""

from datetime import UTC
from datetime import datetime

from fastapi import APIRouter

from api.schemas.responses import HealthResponse
from config.constants import settings

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint for Kubernetes/Cloud Run."""
    from api.main import app_start_time

    uptime = (datetime.now(UTC) - app_start_time).total_seconds()

    return HealthResponse(
        status="healthy", version=settings.api_version, uptime_seconds=uptime
    )
