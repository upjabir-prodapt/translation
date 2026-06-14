"""Handlers for review endpoints."""

from src.api.core.security import AuthenticatedUser
from src.api.schemas.requests import CreateReviewRequest
from src.api.schemas.responses import ReviewListResponse
from src.api.schemas.responses import ReviewSubmitResponse
from src.api.services.review_service import ReviewService


class ReviewsHandler:
    """Thin request handler for review routes."""

    def __init__(self, review_service: ReviewService):
        self.review_service = review_service

    async def create_review(
        self,
        job_id: str,
        request: CreateReviewRequest,
        user: AuthenticatedUser,
    ) -> ReviewSubmitResponse:
        return await self.review_service.create_review(job_id, request, user.email)

    async def get_reviews(self, job_id: str) -> ReviewListResponse:
        return await self.review_service.get_reviews(job_id)
