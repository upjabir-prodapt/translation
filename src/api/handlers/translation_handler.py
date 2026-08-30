"""Handlers for translation endpoints."""

from src.api.schemas.requests import TranslateRequest
from src.api.schemas.responses import JobDetailResponse
from src.api.schemas.responses import MultiTranslateResponse
from src.api.schemas.responses import TranslateResponse
from src.api.services.job_service import JobService
from src.api.services.translation_service import TranslationService


class TranslationHandler:
    """Thin request handler for translation routes."""

    def __init__(
        self,
        translation_service: TranslationService,
        job_service: JobService,
    ):
        self.translation_service = translation_service
        self.job_service = job_service

    async def submit_translation(self, request: TranslateRequest) -> TranslateResponse:
        return await self.translation_service.submit_translation(request)

    async def submit_translations(
        self, requests: list[TranslateRequest]
    ) -> MultiTranslateResponse:
        return await self.translation_service.submit_translations(requests)

    async def get_translation_status(
        self, job_id: str, user_id: str
    ) -> JobDetailResponse:
        return await self.job_service.get_translation_status(job_id, user_id)
