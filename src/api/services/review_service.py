"""Service layer for translation reviews."""

import uuid
from datetime import UTC
from datetime import datetime

from src.api.exceptions import JobNotFoundError
from src.api.exceptions import ReviewNotFoundError
from src.api.schemas.requests import CreateReviewRequest
from src.api.schemas.responses import ReviewListResponse
from src.api.schemas.responses import ReviewResponse
from src.api.schemas.responses import ReviewSubmitResponse
from src.repository.bigquery_repository import BigQueryRepository


class ReviewService:
    """Business logic for creating and fetching translation reviews."""

    def __init__(self, bigquery: BigQueryRepository):
        self.bigquery = bigquery

    async def create_review(
        self,
        job_id: str,
        request: CreateReviewRequest,
        reviewer_email: str,
    ) -> ReviewSubmitResponse:
        """Submit a review for a translation job.

        Creates a new review row. Each reviewer can submit multiple reviews per job;
        each call produces a distinct review_id.
        """
        job = await self.bigquery.get_translation_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)

        now = datetime.now(UTC)
        review_id = str(uuid.uuid4())
        review_data = {
            "review_id": review_id,
            "job_id": job_id,
            "rating": request.rating,
            "comment": request.comment,
            "reviewer_email": reviewer_email,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self.bigquery.upsert_review(review_data)
            return ReviewSubmitResponse(status="successfully sent", review_id=review_id)
        except Exception:
            return ReviewSubmitResponse(status="failed", review_id=None)

    async def get_reviews(self, job_id: str) -> ReviewListResponse:
        """Fetch all reviews for a translation job."""
        job = await self.bigquery.get_translation_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)

        rows = await self.bigquery.get_reviews_by_job_id(job_id)
        if not rows:
            raise ReviewNotFoundError(job_id)

        reviews = [
            ReviewResponse(
                review_id=row["review_id"],
                job_id=row["job_id"],
                rating=row["rating"],
                comment=row.get("comment"),
                reviewer_email=row["reviewer_email"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]
        return ReviewListResponse(job_id=job_id, reviews=reviews, total=len(reviews))
