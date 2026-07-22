"""
Integration tests for review endpoints:
  POST /api/v1/reviews/{job_id}
  GET  /api/v1/reviews/{job_id}

Uses FastAPI TestClient with mocked service layer.
"""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from src.api.core.security import create_access_token
from src.api.dependencies import get_review_service
from src.api.exceptions import JobNotFoundError
from src.api.exceptions import ReviewNotFoundError
from src.api.main import app
from src.api.schemas.responses import ReviewListResponse
from src.api.schemas.responses import ReviewResponse
from src.api.schemas.responses import ReviewSubmitResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_REVIEW = ReviewResponse(
    review_id="rev-001",
    job_id="test-job-id-001",
    rating=4,
    comment="Good translation overall",
    reviewer_email="user@colt.net",
    created_at=_NOW,
    updated_at=_NOW,
)

_REVIEW_LIST = ReviewListResponse(
    job_id="test-job-id-001",
    reviews=[_REVIEW],
    total=1,
)

_REVIEW_SUBMIT = ReviewSubmitResponse(
    status="successfully sent",
    review_id="rev-001",
)


@pytest.fixture(scope="module")
def mock_review_service():
    service = AsyncMock()
    service.create_review.return_value = _REVIEW_SUBMIT
    service.get_reviews.return_value = _REVIEW_LIST
    return service


@pytest.fixture(scope="module")
def review_client(mock_review_service):
    app.dependency_overrides[get_review_service] = lambda: mock_review_service

    token = create_access_token(
        {
            "sub": "user@colt.net",
            "business_unit": "engineering",
            "organization": "colt",
        }
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"x-app-auth": f"Bearer {token}"})
        yield client

    app.dependency_overrides.pop(get_review_service, None)


# ---------------------------------------------------------------------------
# POST /api/v1/reviews/{job_id}
# ---------------------------------------------------------------------------


class TestCreateReview:
    def test_returns_201_on_success(self, review_client):
        resp = review_client.post(
            "/api/v1/reviews/test-job-id-001",
            json={"rating": 4, "comment": "Good translation overall"},
        )
        assert resp.status_code == 201

    def test_response_schema(self, review_client):
        resp = review_client.post(
            "/api/v1/reviews/test-job-id-001",
            json={"rating": 5},
        )
        body = resp.json()
        assert "status" in body
        assert "review_id" in body
        assert body["status"] == "successfully sent"
        assert body["review_id"] == "rev-001"

    def test_rating_below_1_returns_422(self, review_client):
        resp = review_client.post(
            "/api/v1/reviews/test-job-id-001",
            json={"rating": 0},
        )
        assert resp.status_code == 422

    def test_rating_above_5_returns_422(self, review_client):
        resp = review_client.post(
            "/api/v1/reviews/test-job-id-001",
            json={"rating": 6},
        )
        assert resp.status_code == 422

    def test_missing_rating_returns_422(self, review_client):
        resp = review_client.post(
            "/api/v1/reviews/test-job-id-001",
            json={"comment": "No rating field"},
        )
        assert resp.status_code == 422

    def test_returns_404_for_unknown_job(self, review_client, mock_review_service):
        mock_review_service.create_review.side_effect = JobNotFoundError("ghost-job")
        resp = review_client.post(
            "/api/v1/reviews/ghost-job",
            json={"rating": 3},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "JOB_NOT_FOUND"
        # Reset
        mock_review_service.create_review.side_effect = None
        mock_review_service.create_review.return_value = _REVIEW_SUBMIT

    def test_returns_401_without_auth_token(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/reviews/test-job-id-001",
                json={"rating": 3},
            )
        assert resp.status_code == 401

    def test_comment_is_optional(self, review_client):
        resp = review_client.post(
            "/api/v1/reviews/test-job-id-001",
            json={"rating": 5},
        )
        assert resp.status_code == 201

    def test_comment_exceeding_max_length_returns_422(self, review_client):
        resp = review_client.post(
            "/api/v1/reviews/test-job-id-001",
            json={"rating": 3, "comment": "x" * 2001},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/reviews/{job_id}
# ---------------------------------------------------------------------------


class TestGetReviews:
    def test_returns_200_for_existing_reviews(self, review_client):
        resp = review_client.get("/api/v1/reviews/test-job-id-001")
        assert resp.status_code == 200

    def test_response_schema(self, review_client):
        resp = review_client.get("/api/v1/reviews/test-job-id-001")
        body = resp.json()
        assert "job_id" in body
        assert "reviews" in body
        assert "total" in body
        assert isinstance(body["reviews"], list)

    def test_reviews_list_contains_expected_fields(self, review_client):
        resp = review_client.get("/api/v1/reviews/test-job-id-001")
        body = resp.json()
        assert body["total"] == 1
        review = body["reviews"][0]
        assert "review_id" in review
        assert "rating" in review
        assert "reviewer_email" in review

    def test_returns_404_for_unknown_job(self, review_client, mock_review_service):
        mock_review_service.get_reviews.side_effect = JobNotFoundError("ghost-job")
        resp = review_client.get("/api/v1/reviews/ghost-job")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "JOB_NOT_FOUND"
        # Reset
        mock_review_service.get_reviews.side_effect = None
        mock_review_service.get_reviews.return_value = _REVIEW_LIST

    def test_returns_404_when_no_reviews_exist(
        self, review_client, mock_review_service
    ):
        mock_review_service.get_reviews.side_effect = ReviewNotFoundError(
            "no-reviews-job"
        )
        resp = review_client.get("/api/v1/reviews/no-reviews-job")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "REVIEW_NOT_FOUND"
        # Reset
        mock_review_service.get_reviews.side_effect = None
        mock_review_service.get_reviews.return_value = _REVIEW_LIST

    def test_returns_401_without_auth_token(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v1/reviews/test-job-id-001")
        assert resp.status_code == 401

    def test_job_id_in_response_matches_request(self, review_client):
        resp = review_client.get("/api/v1/reviews/test-job-id-001")
        body = resp.json()
        assert body["job_id"] == "test-job-id-001"
