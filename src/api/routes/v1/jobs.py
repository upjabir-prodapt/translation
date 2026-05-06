"""Job management endpoints."""

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Query

from src.api.core.security import get_current_user_context
from src.api.dependencies import get_jobs_handler
from src.api.handlers.jobs_handler import JobsHandler
from src.api.schemas.requests import JobCancelRequest
from src.api.schemas.responses import DownloadResponse
from src.api.schemas.responses import JobListResponse
from src.api.schemas.responses import JobStatusResponse

router = APIRouter(dependencies=[Depends(get_current_user_context)])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str, handler: JobsHandler = Depends(get_jobs_handler)):  # noqa: B008
    """Get the status of a translation job."""
    return await handler.get_job_status(job_id)


@router.get("/jobs", response_model=JobListResponse, tags=["jobs"])
async def list_jobs(
    status: str = Query(
        None, pattern="^(queued|processing|completed|failed|cancelled)$"
    ),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    handler: JobsHandler = Depends(get_jobs_handler),  # noqa: B008
):
    """List translation jobs with optional filtering."""
    return await handler.list_jobs(status, limit, offset)


@router.delete("/jobs/{job_id}", tags=["jobs"])
async def cancel_job(
    job_id: str,
    request: JobCancelRequest = Body(default=JobCancelRequest(reason=None)),  # noqa: B008
    handler: JobsHandler = Depends(get_jobs_handler),  # noqa: B008
):
    """Cancel a translation job."""
    await handler.cancel_job(job_id, request)
    return {"message": "Job cancelled successfully"}


@router.get("/jobs/{job_id}/download", response_model=DownloadResponse, tags=["jobs"])
async def download_output(
    job_id: str,
    handler: JobsHandler = Depends(get_jobs_handler),  # noqa: B008
):
    """Get a signed URL to download the translated PDF (mono output only)."""
    return await handler.download_output(job_id)
