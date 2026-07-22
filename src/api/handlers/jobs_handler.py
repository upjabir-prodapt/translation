"""Handlers for job endpoints."""

from src.api.schemas.requests import JobCancelRequest
from src.api.schemas.responses import DownloadResponse
from src.api.schemas.responses import JobListResponse
from src.api.schemas.responses import JobStatusResponse
from src.api.services.job_service import JobService


class JobsHandler:
    """Thin request handler for job routes."""

    def __init__(self, job_service: JobService):
        self.job_service = job_service

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        return await self.job_service.get_job_status(job_id)

    async def list_jobs(
        self, status: str | None, limit: int, offset: int
    ) -> JobListResponse:
        return await self.job_service.list_jobs(status, limit, offset)

    async def cancel_job(self, job_id: str, request: JobCancelRequest) -> None:
        await self.job_service.cancel_job(job_id, request)

    async def download_output(self, job_id: str) -> DownloadResponse:
        return await self.job_service.get_download_url(job_id, "mono")
