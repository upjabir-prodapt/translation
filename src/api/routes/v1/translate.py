"""Translation endpoints."""

import base64

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

from src.api.dependencies import get_translation_handler
from src.api.handlers.translation_handler import TranslationHandler
from src.api.schemas.requests import CostAttributionInput
from src.api.schemas.requests import DocumentInput
from src.api.schemas.requests import ProcessingOptions
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.requests import TranslationConfigInput
from src.api.schemas.responses import JobDetailResponse
from src.api.schemas.responses import TranslateResponse
from src.api.exceptions import ValidationError

router = APIRouter()


@router.post("/translate", response_model=TranslateResponse, tags=["translation"])
async def submit_translation(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    domain: str = Form(...),
    user_id: str = Form(...),
    business_unit: str = Form(...),
    organization: str = Form(...),
    source_language: str | None = Form(None),
    enable_dlp: bool = Form(True),
    enable_chunking: bool = Form(True),
    priority: str = Form("standard"),
    handler: TranslationHandler = Depends(get_translation_handler),  # noqa: B008
):
    """Submit a document for translation via multipart file upload."""
    content = await file.read()
    if content is None:
        raise ValidationError("No content provided", "document.content")
    request = TranslateRequest(
        document=DocumentInput(
            content=base64.b64encode(content).decode("utf-8"),
            format="docx"
            if (file.filename or "").lower().endswith(".docx")
            else "pdf",
            filename=file.filename or "document.pdf",
        ),
        translation_config=TranslationConfigInput(
            source_language=source_language,
            target_language=target_language,
            domain=domain,
        ),
        cost_attribution=CostAttributionInput(
            user_id=user_id,
            business_unit=business_unit,
            organization=organization,
        ),
        processing_options=ProcessingOptions(
            enable_dlp=enable_dlp,
            enable_chunking=enable_chunking,
            priority=priority,  # validated by ProcessingOptions Literal type
        ),
    )
    return await handler.submit_translation(request)


@router.get(
    "/translate/{job_id}",
    response_model=JobDetailResponse,
    tags=["translation"],
)
async def get_translation_status(
    job_id: str,
    handler: TranslationHandler = Depends(get_translation_handler),  # noqa: B008
):
    """Check translation status and retrieve results."""
    return await handler.get_translation_status(job_id)
