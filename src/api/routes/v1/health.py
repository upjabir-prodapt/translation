"""Health check endpoints."""

from datetime import UTC
from datetime import datetime

from fastapi import APIRouter

from src.api.schemas.responses import HealthResponse
from src.config.constants import settings

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint for Kubernetes/Cloud Run."""
    from src.api.main import app_start_time

    uptime = (datetime.now(UTC) - app_start_time).total_seconds()

    return HealthResponse(
        status="healthy", version=settings.API_VERSION, uptime_seconds=uptime
    )
