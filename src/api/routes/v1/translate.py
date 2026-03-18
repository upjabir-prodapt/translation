"""Translation endpoints."""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

from api.dependencies import get_translation_service
from api.exceptions import ValidationError
from api.schemas.responses import TranslateResponse
from api.services.translation_service import TranslationService
from config.logging import logger

router = APIRouter()


@router.post("/translate", response_model=TranslateResponse, tags=["translation"])
async def submit_translation(
    file: UploadFile = File(...),  # noqa: B008
    domain: str = Form(...),
    lang_in: str = Form("auto"),
    lang_out: str = Form(...),
    user: str = Form(...),
    department: str = Form(...),
    service: TranslationService = Depends(get_translation_service),  # noqa: B008
):
    """Submit a PDF for translation."""
    # Validate file
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.error("Invalid file type: %s", file.filename)
        raise ValidationError("Only PDF files are allowed", "file")

    # Submit translation - exceptions handled by middleware
    return await service.submit_translation(
        file, domain, lang_in, lang_out, user, department
    )
