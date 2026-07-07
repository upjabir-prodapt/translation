"""Route aggregation for API v1."""

from fastapi import APIRouter

from .v1.health import router as health_router
from .v1.internal import router as internal_router
from .v1.jobs import router as jobs_router
from .v1.translate import router as translate_router

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(
    health_router, tags=["health"], responses={404: {"description": "Not found"}}
)

api_router.include_router(
    translate_router,
    tags=["translation"],
    responses={404: {"description": "Not found"}},
)

api_router.include_router(
    jobs_router, tags=["jobs"], responses={404: {"description": "Not found"}}
)

api_router.include_router(
    internal_router,
    tags=["internal"],
    responses={404: {"description": "Not found"}},
)
