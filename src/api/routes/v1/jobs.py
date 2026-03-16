"""Job management endpoints."""

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Query

from api.dependencies import get_job_service
from api.schemas.requests import JobCancelRequest
from api.schemas.responses import DownloadResponse
from api.schemas.responses import JobListResponse
from api.schemas.responses import JobStatusResponse
from api.services.job_service import JobService

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str, service: JobService = Depends(get_job_service)):  # noqa: B008
    """Get the status of a translation job."""
    return await service.get_job_status(job_id)


@router.get("/jobs", response_model=JobListResponse, tags=["jobs"])
async def list_jobs(
    status: str = Query(
        None, pattern="^(queued|processing|completed|failed|cancelled)$"
    ),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: JobService = Depends(get_job_service),  # noqa: B008
):
    """List translation jobs with optional filtering."""
    return await service.list_jobs(status, limit, offset)


@router.delete("/jobs/{job_id}", tags=["jobs"])
async def cancel_job(
    job_id: str,
    request: JobCancelRequest = Body(default=JobCancelRequest(reason=None)),  # noqa: B008
    service: JobService = Depends(get_job_service),  # noqa: B008
):
    """Cancel a translation job."""
    await service.cancel_job(job_id, request)
    return {"message": "Job cancelled successfully"}


@router.get("/jobs/{job_id}/download", response_model=DownloadResponse, tags=["jobs"])
async def download_output(
    job_id: str,
    service: JobService = Depends(get_job_service),  # noqa: B008
):
    """Get a signed URL to download the translated PDF (mono output only)."""
    return await service.get_download_url(job_id, "mono")
