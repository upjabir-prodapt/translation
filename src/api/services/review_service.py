"""Service layer for translation reviews."""

import logging
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

logger = logging.getLogger(__name__)


class ReviewService:
    """Business logic for creating and fetching translation reviews."""

    def __init__(self, bigquery: BigQueryRepository):
        self.bigquery = bigquery

    @staticmethod
    def _assert_owner(job: dict, job_id: str, user_id: str) -> None:
        """Reject access to reviews on a job belonging to another user.

        UAT EC-09/S-04 (D-12): both review endpoints authenticated the
        caller but never checked ownership, and the BigQuery query filters
        on `job_id` alone. Any authenticated user holding another user's
        `job_id` could therefore read every review on it -- the free-text
        comment and the reviewer's e-mail address -- and post reviews onto
        it. Mirrors `JobService._assert_owner`: raises `JobNotFoundError`
        (404), not 403, so the endpoint cannot be used to enumerate job IDs.
        """
        owner = (job.get("cost_attribution") or {}).get("user_id")
        if owner != user_id:
            logger.warning(
                "Rejected review access to job %s: ownership mismatch", job_id
            )
            raise JobNotFoundError(job_id)

    async def create_review(
        self,
        job_id: str,
        request: CreateReviewRequest,
        reviewer_email: str,
    ) -> ReviewSubmitResponse:
        """Submit a review for a translation job owned by `reviewer_email`.

        Creates a new review row. Each reviewer can submit multiple reviews per job;
        each call produces a distinct review_id.
        """
        job = await self.bigquery.get_translation_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)

        self._assert_owner(job, job_id, reviewer_email)

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
        await self.bigquery.upsert_review(review_data)
        return ReviewSubmitResponse(status="successfully sent", review_id=review_id)

    async def get_reviews(self, job_id: str, user_id: str) -> ReviewListResponse:
        """Fetch all reviews for a translation job owned by `user_id`."""
        job = await self.bigquery.get_translation_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)

        self._assert_owner(job, job_id, user_id)

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
