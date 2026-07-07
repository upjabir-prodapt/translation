"""Translation endpoints."""

from fastapi import APIRouter
from fastapi import Depends

from api.dependencies import get_job_service
from api.dependencies import get_translation_service
from api.schemas.requests import TranslateRequest
from api.schemas.responses import JobDetailResponse
from api.schemas.responses import TranslateResponse
from api.services.job_service import JobService
from api.services.translation_service import TranslationService

router = APIRouter()


@router.post("/translate", response_model=TranslateResponse, tags=["translation"])
async def submit_translation(
    request: TranslateRequest,
    service: TranslationService = Depends(get_translation_service),  # noqa: B008
):
    """Submit a document for translation."""
    return await service.submit_translation(request)


@router.get(
    "/translate/{job_id}",
    response_model=JobDetailResponse,
    tags=["translation"],
)
async def get_translation_status(
    job_id: str,
    service: JobService = Depends(get_job_service),  # noqa: B008
):
    """Check translation status and retrieve results."""
    return await service.get_translation_status(job_id)
