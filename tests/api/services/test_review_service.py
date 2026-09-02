from unittest.mock import AsyncMock

import pytest
from src.api.exceptions import JobNotFoundError
from src.api.exceptions import ReviewNotFoundError
from src.api.schemas.requests import CreateReviewRequest
from src.api.services.review_service import ReviewService

OWNER = "user@example.com"
OTHER_USER = "someone.else@example.com"


def _job(owner: str = OWNER) -> dict:
    """A job row as BigQuery returns it, owned by `owner`."""
    return {"job_id": "job-1", "cost_attribution": {"user_id": owner}}


class TestReviewService:
    @pytest.mark.asyncio
    async def test_create_review_success(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = _job()
        mock_bq.upsert_review.return_value = None

        service = ReviewService(mock_bq)
        req = CreateReviewRequest(rating=5, comment="Great translation")
        res = await service.create_review("job-1", req, OWNER)

        assert res.status == "successfully sent"
        assert res.review_id is not None
        mock_bq.upsert_review.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_review_job_not_found(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = None

        service = ReviewService(mock_bq)
        req = CreateReviewRequest(rating=5)
        with pytest.raises(JobNotFoundError):
            await service.create_review("unknown-job", req, OWNER)

    @pytest.mark.asyncio
    async def test_create_review_propagates_bigquery_error(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = _job()
        mock_bq.upsert_review.side_effect = RuntimeError("BigQuery connection error")

        service = ReviewService(mock_bq)
        req = CreateReviewRequest(rating=4)
        with pytest.raises(RuntimeError, match="BigQuery connection error"):
            await service.create_review("job-1", req, OWNER)

    @pytest.mark.asyncio
    async def test_get_reviews_success(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = _job()
        mock_bq.get_reviews_by_job_id.return_value = [
            {
                "review_id": "rev-1",
                "job_id": "job-1",
                "rating": 5,
                "comment": "Nice",
                "reviewer_email": "u@test.com",
                "created_at": "2026-08-25T12:00:00Z",
                "updated_at": "2026-08-25T12:00:00Z",
            }
        ]

        service = ReviewService(mock_bq)
        res = await service.get_reviews("job-1", OWNER)
        assert res.total == 1
        assert res.reviews[0].review_id == "rev-1"

    @pytest.mark.asyncio
    async def test_get_reviews_not_found(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = _job()
        mock_bq.get_reviews_by_job_id.return_value = []

        service = ReviewService(mock_bq)
        with pytest.raises(ReviewNotFoundError):
            await service.get_reviews("job-1", OWNER)


class TestReviewOwnership:
    """UAT EC-09/S-04 (D-12): reviews used to be readable and writable by any
    authenticated user who knew another user's job_id."""

    @pytest.mark.asyncio
    async def test_get_reviews_rejects_another_users_job(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = _job(owner=OWNER)
        mock_bq.get_reviews_by_job_id.return_value = [
            {
                "review_id": "rev-1",
                "job_id": "job-1",
                "rating": 1,
                "comment": "confidential feedback",
                "reviewer_email": OWNER,
                "created_at": "2026-08-25T12:00:00Z",
                "updated_at": "2026-08-25T12:00:00Z",
            }
        ]

        service = ReviewService(mock_bq)
        with pytest.raises(JobNotFoundError):
            await service.get_reviews("job-1", OTHER_USER)
        # The review rows must never even be fetched for a non-owner.
        mock_bq.get_reviews_by_job_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_review_rejects_another_users_job(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = _job(owner=OWNER)

        service = ReviewService(mock_bq)
        with pytest.raises(JobNotFoundError):
            await service.create_review(
                "job-1", CreateReviewRequest(rating=1), OTHER_USER
            )
        mock_bq.upsert_review.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_cost_attribution_is_treated_as_not_owned(self):
        """A job row with no owner recorded must not be readable by anyone."""
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = {"job_id": "job-1"}

        service = ReviewService(mock_bq)
        with pytest.raises(JobNotFoundError):
            await service.get_reviews("job-1", OWNER)
