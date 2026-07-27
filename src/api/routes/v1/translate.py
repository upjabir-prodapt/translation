"""Translation endpoints."""

import base64
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import Response
from fastapi import UploadFile
from starlette import status

from src.api.core.security import AuthenticatedUser
from src.api.core.security import get_current_user_context
from src.api.dependencies import get_translation_handler
from src.api.exceptions import ValidationError
from src.api.handlers.translation_handler import TranslationHandler
from src.api.schemas.requests import CostAttributionInput
from src.api.schemas.requests import DocumentInput
from src.api.schemas.requests import ProcessingOptions
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.requests import TranslationConfigInput
from src.api.schemas.requests import TranslationTargetsInput
from src.api.schemas.responses import JobDetailResponse
from src.api.schemas.responses import MultiTranslateResponse
from src.config.constants import settings

router = APIRouter()


@router.post(
    "/translate",
    response_model=MultiTranslateResponse,
    tags=["translation"],
)
async def submit_translation(
    file: Annotated[UploadFile, File(...)],
    domain: Annotated[str, Form(...)],
    response: Response,
    target_languages: Annotated[list[str], Form(...)],
    source_language: Annotated[str | None, Form()] = None,
    enable_dlp: Annotated[bool, Form()] = True,
    enable_chunking: Annotated[bool, Form()] = True,
    priority: Annotated[str, Form()] = "standard",
    current_user: Annotated[
        AuthenticatedUser, Depends(get_current_user_context)
    ] = None,  # noqa: B008
    handler: Annotated[TranslationHandler, Depends(get_translation_handler)] = None,  # noqa: B008
):
    """Submit a document for translation via multipart upload and bearer auth.

    Accepts one or more target languages and returns HTTP 202 with batch details.
    """
    content = await file.read()
    if not content:
        raise ValidationError("Document content is empty", "document.content")
    if len(content) > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        raise ValidationError(
            f"Document size exceeds {max_mb}MB limit", "document.size"
        )
    targets = TranslationTargetsInput(
        target_languages=target_languages,
    ).normalized_targets
    document = DocumentInput(
        content=base64.b64encode(content).decode("utf-8"),
        format="docx" if (file.filename or "").lower().endswith(".docx") else "pdf",
        filename=file.filename or "document.pdf",
    )
    cost_attribution = CostAttributionInput(
        user_id=current_user.email,
        business_unit=current_user.business_unit,
        organization=current_user.organization,
    )
    processing_options = ProcessingOptions(
        enable_dlp=enable_dlp,
        enable_chunking=enable_chunking,
        priority=priority,
    )
    requests = [
        TranslateRequest(
            document=document,
            translation_config=TranslationConfigInput(
                source_language=source_language,
                target_language=target,
                domain=domain,
            ),
            cost_attribution=cost_attribution,
            processing_options=processing_options,
        )
        for target in targets
    ]
    response.status_code = status.HTTP_202_ACCEPTED
    return await handler.submit_translations(requests)


@router.get(
    "/translate/{job_id}", response_model=JobDetailResponse, tags=["translation"]
)
async def get_translation_status(
    job_id: str,
    _current_user: Annotated[
        AuthenticatedUser, Depends(get_current_user_context)
    ] = None,  # noqa: B008
    handler: Annotated[TranslationHandler, Depends(get_translation_handler)] = None,  # noqa: B008
):
    """Get the status of a translation job."""
    return await handler.get_translation_status(job_id)
