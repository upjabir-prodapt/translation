"""Job management endpoints."""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Query

from src.api.core.security import AuthenticatedUser
from src.api.core.security import get_current_user_context
from src.api.dependencies import get_jobs_handler
from src.api.handlers.jobs_handler import JobsHandler
from src.api.schemas.requests import JobCancelRequest
from src.api.schemas.requests import MultiJobStatusRequest
from src.api.schemas.responses import DownloadResponse
from src.api.schemas.responses import JobListResponse
from src.api.schemas.responses import JobStatusResponse
from src.api.schemas.responses import MultiJobStatusResponse

router = APIRouter(dependencies=[Depends(get_current_user_context)])


@router.post("/jobs/status", response_model=MultiJobStatusResponse, tags=["jobs"])
async def get_jobs_status(
    request: MultiJobStatusRequest,
    current_user: Annotated[
        AuthenticatedUser, Depends(get_current_user_context)
    ] = None,  # noqa: B008
    handler: Annotated[JobsHandler, Depends(get_jobs_handler)] = None,  # noqa: B008
):
    """Get ordered statuses for multiple translation jobs."""
    return await handler.get_jobs_status(request.job_ids, current_user.email)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(
    job_id: str, handler: Annotated[JobsHandler, Depends(get_jobs_handler)] = None
):  # noqa: B008
    """Get the status of a translation job."""
    return await handler.get_job_status(job_id)


@router.get("/jobs", response_model=JobListResponse, tags=["jobs"])
async def list_jobs(
    status: Annotated[
        str | None, Query(pattern="^(queued|processing|completed|failed|cancelled)$")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0,
    handler: Annotated[JobsHandler, Depends(get_jobs_handler)] = None,  # noqa: B008
):
    """List translation jobs with optional filtering."""
    return await handler.list_jobs(status, limit, offset)


@router.delete("/jobs/{job_id}", tags=["jobs"])
async def cancel_job(
    job_id: str,
    request: Annotated[JobCancelRequest, Body()] = JobCancelRequest(reason=None),  # noqa: B008
    handler: Annotated[JobsHandler, Depends(get_jobs_handler)] = None,  # noqa: B008
):
    """Cancel a translation job."""
    await handler.cancel_job(job_id, request)
    return {"message": "Job cancelled successfully"}


@router.get("/jobs/{job_id}/download", response_model=DownloadResponse, tags=["jobs"])
async def download_output(
    job_id: str,
    handler: Annotated[JobsHandler, Depends(get_jobs_handler)] = None,  # noqa: B008
):
    """Get a signed URL to download the translated PDF (mono output only)."""
    return await handler.download_output(job_id)
