"""Route aggregation for API v1."""

from fastapi import APIRouter

from .v1.auth import router as auth_router
from .v1.health import router as health_router
from .v1.jobs import router as jobs_router
from .v1.translate import router as translate_router

# Create main API router
api_router = APIRouter()

NOT_FOUND_RESPONSE = {404: {"description": "Not found"}}

# Include all route modules
api_router.include_router(
    health_router, tags=["health"], responses=NOT_FOUND_RESPONSE
)

api_router.include_router(
    auth_router, tags=["auth"], responses=NOT_FOUND_RESPONSE
)

api_router.include_router(
    translate_router,
    tags=["translation"],
    responses=NOT_FOUND_RESPONSE,
)

api_router.include_router(
    jobs_router, tags=["jobs"], responses=NOT_FOUND_RESPONSE
)
