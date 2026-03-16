"""Health and readiness check endpoints."""

from fastapi import APIRouter
from fastapi import HTTPException

from config.logging import logger
from worker.core.lifecycle import WorkerLifecycle

router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "babeldoc-worker"}


@router.get("/ready")
async def ready():
    """Readiness check endpoint."""
    try:
        # Check if worker is initialized
        if not WorkerLifecycle.is_initialized():
            raise HTTPException(
                status_code=503,
                detail={"status": "not ready", "error": "Worker not initialized"},
            )

        return {"status": "ready", "service": "babeldoc-worker", "models_loaded": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503, detail={"status": "not ready", "error": str(e)}
        ) from e
