"""Route aggregation for Worker v1."""

from fastapi import APIRouter

from worker.routes.v1.health import router as health_router
from worker.routes.v1.process import router as process_router

# Create main worker router
worker_router = APIRouter()

# Include all route modules
worker_router.include_router(
    health_router, tags=["health"], responses={404: {"description": "Not found"}}
)

worker_router.include_router(
    process_router, tags=["processing"], responses={404: {"description": "Not found"}}
)
