"""
Integration tests for job management endpoints:
  GET    /api/v1/jobs/{job_id}
  GET    /api/v1/jobs
  DELETE /api/v1/jobs/{job_id}
  GET    /api/v1/jobs/{job_id}/download

Uses FastAPI TestClient with mocked service layer. See integration/conftest.py.
"""

from datetime import UTC
from datetime import datetime

from fastapi import HTTPException
from src.api.exceptions import JobAlreadyCompletedError
from src.api.exceptions import JobNotFoundError
from src.api.schemas.responses import DownloadResponse
from src.api.schemas.responses import JobStatusResponse

# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}
# ---------------------------------------------------------------------------


class TestGetJobStatus:
    def test_returns_200_for_existing_job(self, api_client):
        resp = api_client.get("/api/v1/jobs/test-job-id-001")
        assert resp.status_code == 200

    def test_response_schema(self, api_client):
        resp = api_client.get("/api/v1/jobs/test-job-id-001")
        body = resp.json()
        assert "job_id" in body
        assert "status" in body
        assert "progress" in body

    def test_returns_404_for_missing_job(self, api_client, mock_job_service):
        mock_job_service.get_job_status.side_effect = JobNotFoundError("no-job")
        resp = api_client.get("/api/v1/jobs/no-job")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "JOB_NOT_FOUND"
        # Reset
        mock_job_service.get_job_status.side_effect = None
        now = datetime.now(UTC)
        mock_job_service.get_job_status.return_value = JobStatusResponse(
            job_id="test-job-id-001",
            status="queued",
            progress=0.0,
            current_stage=None,
            user="",
            department="",
            created_at=now,
            updated_at=now,
            completed_at=None,
            error_message=None,
        )

    def test_body_error_structure_on_404(self, api_client, mock_job_service):
        mock_job_service.get_job_status.side_effect = JobNotFoundError("ghost")
        resp = api_client.get("/api/v1/jobs/ghost")
        body = resp.json()
        assert "error" in body
        assert "message" in body["error"]
        # Reset
        mock_job_service.get_job_status.side_effect = None
        now = datetime.now(UTC)
        mock_job_service.get_job_status.return_value = JobStatusResponse(
            job_id="test-job-id-001",
            status="queued",
            progress=0.0,
            current_stage=None,
            user="",
            department="",
            created_at=now,
            updated_at=now,
            completed_at=None,
            error_message=None,
        )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    def test_returns_200(self, api_client):
        resp = api_client.get("/api/v1/jobs")
        assert resp.status_code == 200

    def test_response_has_jobs_array(self, api_client):
        resp = api_client.get("/api/v1/jobs")
        body = resp.json()
        assert "jobs" in body
        assert isinstance(body["jobs"], list)

    def test_response_has_pagination_fields(self, api_client):
        resp = api_client.get("/api/v1/jobs")
        body = resp.json()
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_status_filter_passed(self, api_client, mock_job_service):
        """Valid status query param should be forwarded to service."""
        resp = api_client.get("/api/v1/jobs?status=queued")
        assert resp.status_code == 200

    def test_invalid_status_filter_returns_422(self, api_client):
        resp = api_client.get("/api/v1/jobs?status=pending")
        assert resp.status_code == 422

    def test_limit_pagination_param(self, api_client):
        resp = api_client.get("/api/v1/jobs?limit=5&offset=0")
        assert resp.status_code == 200

    def test_limit_above_max_returns_422(self, api_client):
        resp = api_client.get("/api/v1/jobs?limit=200")
        assert resp.status_code == 422

    def test_limit_zero_returns_422(self, api_client):
        resp = api_client.get("/api/v1/jobs?limit=0")
        assert resp.status_code == 422

    def test_negative_offset_returns_422(self, api_client):
        resp = api_client.get("/api/v1/jobs?offset=-5")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /api/v1/jobs/{job_id}
# ---------------------------------------------------------------------------


class TestCancelJob:
    def test_returns_200_for_queued_job(self, api_client, mock_job_service):
        mock_job_service.cancel_job.return_value = None
        resp = api_client.delete("/api/v1/jobs/test-job-id-001")
        assert resp.status_code == 200

    def test_response_has_success_message(self, api_client, mock_job_service):
        mock_job_service.cancel_job.return_value = None
        resp = api_client.delete("/api/v1/jobs/test-job-id-001")
        body = resp.json()
        assert "cancelled" in body["message"].lower()

    def test_with_reason_in_body(self, api_client, mock_job_service):
        mock_job_service.cancel_job.return_value = None
        # Starlette TestClient's .delete() doesn't accept json=; use .request() instead.
        resp = api_client.request(
            "DELETE",
            "/api/v1/jobs/test-job-id-001",
            json={"reason": "No longer needed"},
        )
        assert resp.status_code == 200

    def test_returns_404_for_missing_job(self, api_client, mock_job_service):
        mock_job_service.cancel_job.side_effect = JobNotFoundError("ghost")
        resp = api_client.delete("/api/v1/jobs/ghost")
        assert resp.status_code == 404
        # Reset
        mock_job_service.cancel_job.side_effect = None
        mock_job_service.cancel_job.return_value = None

    def test_returns_409_for_completed_job(self, api_client, mock_job_service):
        mock_job_service.cancel_job.side_effect = JobAlreadyCompletedError("done-job")
        resp = api_client.delete("/api/v1/jobs/done-job")
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"]["code"] == "JOB_ALREADY_COMPLETED"
        # Reset
        mock_job_service.cancel_job.side_effect = None
        mock_job_service.cancel_job.return_value = None


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/download
# ---------------------------------------------------------------------------


class TestDownloadOutput:
    def test_returns_200_for_completed_job(self, api_client, mock_job_service):
        # side_effect takes precedence over return_value — clear it first.
        mock_job_service.get_download_url.side_effect = None
        mock_job_service.get_download_url.return_value = DownloadResponse(
            download_url="https://storage.googleapis.com/signed/doc_es.pdf",
            expires_in=3600,
            filename="test-job_mono.pdf",
            file_size=204800,
        )
        resp = api_client.get("/api/v1/jobs/test-job-id-001/download")
        assert resp.status_code == 200
        # Reset
        mock_job_service.get_download_url.side_effect = Exception("not configured")
        mock_job_service.get_download_url.return_value = None

    def test_response_has_download_url(self, api_client, mock_job_service):
        mock_job_service.get_download_url.side_effect = None
        mock_job_service.get_download_url.return_value = DownloadResponse(
            download_url="https://signed.url/output.pdf",
            expires_in=3600,
            filename="output.pdf",
            file_size=None,
        )
        resp = api_client.get("/api/v1/jobs/test-job-id-001/download")
        body = resp.json()
        assert "download_url" in body
        assert body["download_url"].startswith("https://")
        # Reset
        mock_job_service.get_download_url.side_effect = Exception("not configured")

    def test_returns_404_for_missing_job(self, api_client, mock_job_service):
        mock_job_service.get_download_url.side_effect = JobNotFoundError("ghost")
        resp = api_client.get("/api/v1/jobs/ghost/download")
        assert resp.status_code == 404
        # Reset
        mock_job_service.get_download_url.side_effect = Exception("not configured")

    def test_returns_400_if_not_completed(self, api_client, mock_job_service):
        mock_job_service.get_download_url.side_effect = HTTPException(
            status_code=400, detail="Job is not completed yet"
        )
        resp = api_client.get("/api/v1/jobs/queued-job/download")
        assert resp.status_code == 400
        # Reset
        mock_job_service.get_download_url.side_effect = Exception("not configured")


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_200(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code == 200

    def test_root_has_service_name(self, api_client):
        resp = api_client.get("/")
        body = resp.json()
        assert "service" in body

    def test_root_has_status_running(self, api_client):
        resp = api_client.get("/")
        body = resp.json()
        assert body["status"] == "running"
