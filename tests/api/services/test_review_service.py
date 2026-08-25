from unittest.mock import AsyncMock

import pytest
from src.api.exceptions import JobNotFoundError
from src.api.exceptions import ReviewNotFoundError
from src.api.schemas.requests import CreateReviewRequest
from src.api.services.review_service import ReviewService


class TestReviewService:
    @pytest.mark.asyncio
    async def test_create_review_success(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = {"job_id": "job-1"}
        mock_bq.upsert_review.return_value = None

        service = ReviewService(mock_bq)
        req = CreateReviewRequest(rating=5, comment="Great translation")
        res = await service.create_review("job-1", req, "user@example.com")

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
            await service.create_review("unknown-job", req, "user@example.com")

    @pytest.mark.asyncio
    async def test_create_review_propagates_bigquery_error(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = {"job_id": "job-1"}
        mock_bq.upsert_review.side_effect = RuntimeError("BigQuery connection error")

        service = ReviewService(mock_bq)
        req = CreateReviewRequest(rating=4)
        with pytest.raises(RuntimeError, match="BigQuery connection error"):
            await service.create_review("job-1", req, "user@example.com")

    @pytest.mark.asyncio
    async def test_get_reviews_success(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = {"job_id": "job-1"}
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
        res = await service.get_reviews("job-1")
        assert res.total == 1
        assert res.reviews[0].review_id == "rev-1"

    @pytest.mark.asyncio
    async def test_get_reviews_not_found(self):
        mock_bq = AsyncMock()
        mock_bq.get_translation_job.return_value = {"job_id": "job-1"}
        mock_bq.get_reviews_by_job_id.return_value = []

        service = ReviewService(mock_bq)
        with pytest.raises(ReviewNotFoundError):
            await service.get_reviews("job-1")
