"""Handlers for job endpoints."""

from src.api.schemas.requests import JobCancelRequest
from src.api.schemas.responses import DownloadResponse
from src.api.schemas.responses import JobListResponse
from src.api.schemas.responses import JobStatusResponse
from src.api.schemas.responses import MultiJobStatusResponse
from src.api.services.job_service import JobService


class JobsHandler:
    """Thin request handler for job routes."""

    def __init__(self, job_service: JobService):
        self.job_service = job_service

    async def get_job_status(self, job_id: str, user_id: str) -> JobStatusResponse:
        return await self.job_service.get_job_status(job_id, user_id)

    async def get_jobs_status(
        self, job_ids: list[str], user_id: str
    ) -> MultiJobStatusResponse:
        return await self.job_service.get_jobs_status(job_ids, user_id)

    async def list_jobs(
        self, status: str | None, limit: int, offset: int, user_id: str
    ) -> JobListResponse:
        return await self.job_service.list_jobs(status, limit, offset, user_id=user_id)

    async def cancel_job(
        self, job_id: str, request: JobCancelRequest, user_id: str
    ) -> None:
        await self.job_service.cancel_job(job_id, request, user_id)

    async def download_output(self, job_id: str, user_id: str) -> DownloadResponse:
        return await self.job_service.get_download_url(job_id, "mono", user_id)
