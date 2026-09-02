"""Review endpoints for translation jobs."""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends

from src.api.core.security import AuthenticatedUser
from src.api.core.security import get_current_user_context
from src.api.dependencies import get_reviews_handler
from src.api.handlers.reviews_handler import ReviewsHandler
from src.api.schemas.requests import CreateReviewRequest
from src.api.schemas.responses import ReviewListResponse
from src.api.schemas.responses import ReviewSubmitResponse

router = APIRouter(dependencies=[Depends(get_current_user_context)])


@router.post(
    "/reviews/{job_id}",
    response_model=ReviewSubmitResponse,
    status_code=201,
    tags=["reviews"],
)
async def create_review(
    job_id: str,
    request: Annotated[CreateReviewRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user_context)],
    handler: Annotated[ReviewsHandler, Depends(get_reviews_handler)] = None,  # noqa: B008
):
    """Submit a review for a completed translation job."""
    return await handler.create_review(job_id, request, user)


@router.get(
    "/reviews/{job_id}",
    response_model=ReviewListResponse,
    tags=["reviews"],
)
async def get_reviews(
    job_id: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user_context)],
    handler: Annotated[ReviewsHandler, Depends(get_reviews_handler)] = None,  # noqa: B008
):
    """Fetch all reviews for a translation job.

    Only the job's owner may read its reviews; another user's job returns
    404 rather than 403 so this endpoint cannot enumerate job IDs.
    """
    return await handler.get_reviews(job_id, user)
