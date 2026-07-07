"""Internal maintenance endpoints (invoked by Cloud Scheduler, not end users).

Access control is expected to be enforced at the infrastructure level
(Cloud Run IAM + Cloud Scheduler OIDC), matching the auth posture of the
worker's /process endpoint.
"""

from fastapi import APIRouter
from fastapi import Depends

from api.dependencies import get_file_lifecycle_service
from api.schemas.responses import StorageCleanupResponse
from api.services.file_lifecycle_service import FileLifecycleService

router = APIRouter()


@router.post(
    "/internal/storage/cleanup-expired-outputs",
    response_model=StorageCleanupResponse,
    tags=["internal"],
)
async def cleanup_expired_outputs(
    service: FileLifecycleService = Depends(get_file_lifecycle_service),  # noqa: B008
):
    """Delete output files past their retention window (idempotent)."""
    return await service.cleanup_expired_outputs()
