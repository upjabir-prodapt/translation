from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.api.exceptions import JobAlreadyCompletedError
from src.api.exceptions import JobNotFoundError
from src.api.schemas.requests import JobCancelRequest
from src.api.services.job_service import JobService
from tests.async_test_utils import patch_asyncio_sleep


@pytest.fixture
def mock_storage():
    return AsyncMock()


@pytest.fixture
def mock_bq():
    return AsyncMock()


@pytest.fixture
def service(mock_storage, mock_bq):
    # cloud_tasks is injected so no real Cloud Tasks client is constructed.
    return JobService(
        storage=mock_storage, bigquery=mock_bq, cloud_tasks=MagicMock()
    )


class TestJobService:
    async def test_get_job_status_not_found(self, service, mock_bq):
        mock_bq.get_translation_job.return_value = None
        with pytest.raises(JobNotFoundError):
            await service.get_job_status("nonexistent")

    async def test_get_job_status_success(self, service, mock_bq, mock_storage):
        mock_bq.get_translation_job.return_value = {
            "job_id": "job1",
            "status": "completed",
            "result": {"output_gcs_uri": "gs://bucket/mono.pdf"},
            "submitted_at": "2026-05-08T12:00:00Z",
        }
        mock_storage.generate_signed_url.return_value = "http://download"
        status = await service.get_job_status("job1")
        assert status.status == "completed"
        assert status.download_url == "http://download"

    async def test_get_jobs_status_preserves_order_and_signs_completed(
        self, service, mock_bq, mock_storage
    ):
        mock_bq.get_translation_jobs_by_ids.return_value = [
            {
                "job_id": "job-2",
                "status": "processing",
                "translation_config": {"target_language": "de"},
                "cost_attribution": {"user_id": "user@colt.net"},
            },
            {
                "job_id": "job-1",
                "status": "completed",
                "translation_config": {"target_language": "fr"},
                "cost_attribution": {"user_id": "user@colt.net"},
                "source_document": {"output_filename": "contract_fr.pdf"},
                "result": {"output_gcs_uri": "gs://bucket/contract_fr.pdf"},
            },
        ]
        mock_storage.generate_signed_url.return_value = "https://download"

        response = await service.get_jobs_status(["job-1", "job-2"], "user@colt.net")

        assert [job.job_id for job in response.jobs] == ["job-1", "job-2"]
        assert response.jobs[0].download_url == "https://download"
        assert response.jobs[0].download_filename == "contract_fr.pdf"
        assert response.jobs[1].download_url is None
        mock_storage.generate_signed_url.assert_awaited_once()

    async def test_get_jobs_status_hides_unowned_job(self, service, mock_bq):
        mock_bq.get_translation_jobs_by_ids.return_value = [
            {
                "job_id": "job-1",
                "status": "queued",
                "cost_attribution": {"user_id": "another@colt.net"},
            }
        ]
        with pytest.raises(JobNotFoundError):
            await service.get_jobs_status(["job-1"], "user@colt.net")

    async def test_get_jobs_status_rejects_missing_job(self, service, mock_bq):
        mock_bq.get_translation_jobs_by_ids.return_value = []
        with pytest.raises(JobNotFoundError):
            await service.get_jobs_status(["job-1"], "user@colt.net")

    async def test_cancel_job_already_completed(self, service, mock_bq):
        mock_bq.get_translation_job.return_value = {
            "status": "completed",
            "cost_attribution": {"user_id": OWNER},
        }
        with pytest.raises(JobAlreadyCompletedError):
            await service.cancel_job("job1", JobCancelRequest(reason="test"), OWNER)

    async def test_cancel_job_success(self, service, mock_bq):
        mock_bq.get_translation_job.return_value = {
            "job_id": "job1",
            "status": "processing",
            "cost_attribution": {"user_id": OWNER},
        }
        await service.cancel_job("job1", JobCancelRequest(reason="test"), OWNER)
        # It calls patch_translation_job
        mock_bq.patch_translation_job.assert_awaited_once()
        args, kwargs = mock_bq.patch_translation_job.call_args
        assert args[0] == "job1"
        assert args[1]["status"] == "cancelled"


OWNER = "owner@colt.net"
OTHER = "attacker@colt.net"


class TestCancelJobOwnership:
    """Cancel must be restricted to the job's owner.

    Before this guard, any authenticated user could cancel any job by ID --
    `cancel_job` took no user_id at all, unlike `list_jobs`/`get_jobs_status`.
    """

    @staticmethod
    def _job(owner=OWNER, status="processing", priority=None):
        job = {
            "job_id": "job1",
            "status": status,
            "cost_attribution": {"user_id": owner},
        }
        if priority is not None:
            job["processing_options"] = {"priority": priority}
        return job

    async def test_cancel_rejects_other_users_job(self, service, mock_bq):
        mock_bq.get_translation_job.return_value = self._job(owner=OWNER)
        with pytest.raises(JobNotFoundError):
            await service.cancel_job("job1", JobCancelRequest(reason="x"), OTHER)
        # Crucially: the job must NOT have been modified.
        mock_bq.patch_translation_job.assert_not_awaited()

    async def test_cancel_missing_cost_attribution_is_rejected(
        self, service, mock_bq
    ):
        """A legacy row with no owner must fail closed, not crash."""
        mock_bq.get_translation_job.return_value = {
            "job_id": "job1",
            "status": "processing",
        }
        with pytest.raises(JobNotFoundError):
            await service.cancel_job("job1", JobCancelRequest(reason="x"), OWNER)
        mock_bq.patch_translation_job.assert_not_awaited()

    async def test_cancel_deletes_cloud_task(self, service, mock_bq):
        mock_bq.get_translation_job.return_value = self._job(priority="high")
        service.cloud_tasks = MagicMock()
        await service.cancel_job("job1", JobCancelRequest(reason="x"), OWNER)
        service.cloud_tasks.delete_translate_task.assert_called_once_with(
            "job1", "high"
        )

    async def test_cancel_survives_task_delete_failure(self, service, mock_bq):
        """Cloud Tasks being down must never block a cancel."""
        mock_bq.get_translation_job.return_value = self._job()
        service.cloud_tasks = MagicMock()
        service.cloud_tasks.delete_translate_task.side_effect = RuntimeError("boom")
        await service.cancel_job("job1", JobCancelRequest(reason="x"), OWNER)
        mock_bq.patch_translation_job.assert_awaited_once()
        assert mock_bq.patch_translation_job.call_args[0][1]["status"] == "cancelled"

    async def test_task_deleted_after_status_patch(self, service, mock_bq):
        """Ordering matters: a Tasks outage must not leave a job un-cancelled."""
        order = []
        mock_bq.get_translation_job.return_value = self._job()
        mock_bq.patch_translation_job.side_effect = (
            lambda *_args, **_kwargs: order.append("patch")
        )
        service.cloud_tasks = MagicMock()
        service.cloud_tasks.delete_translate_task.side_effect = (
            lambda *_args, **_kwargs: order.append("delete")
        )
        await service.cancel_job("job1", JobCancelRequest(reason="x"), OWNER)
        assert order == ["patch", "delete"]

    async def test_list_jobs(self, service, mock_bq):
        mock_bq.list_translation_jobs.return_value = [
            {
                "job_id": "1",
                "status": "completed",
                "submitted_at": "2026-05-08T12:00:00Z",
            }
        ]
        res = await service.list_jobs()
        assert len(res.jobs) == 1
        assert res.total == 1

    async def test_get_download_url_success(self, service, mock_bq, mock_storage):
        mock_bq.get_translation_job.return_value = {
            "job_id": "job1",
            "status": "completed",
            "result": {"output_gcs_uri": "gs://bucket/mono.pdf"},
        }
        mock_storage.generate_signed_url.return_value = "http://download"
        mock_storage.get_file_info.return_value = {"size": 1024}

        res = await service.get_download_url("job1", "mono")
        assert res.download_url == "http://download"
        assert res.file_size == 1024

    async def test_get_translation_status_success(self, service, mock_bq):
        mock_bq.get_translation_job.return_value = {
            "job_id": "job1",
            "status": "completed",
            "source_document": {"filename": "in.pdf"},
            "translation_config": {"target_language": "fr"},
            "cost_attribution": {"user_id": "u1"},
            "result": {"mono_pdf_path": "gs://b/out.pdf"},
            "submitted_at": "2026-05-08T12:00:00Z",
        }
        res = await service.get_translation_status("job1")
        assert res.job_id == "job1"
        assert res.status == "completed"

    async def test_stream_job_progress(self, service, mock_bq):
        mock_bq.get_translation_job.side_effect = [
            {"status": "processing", "progress": 50, "job_id": "1"},
            {"status": "completed", "progress": 100, "job_id": "1"},
        ]

        # Patch sleep to make test fast
        with patch_asyncio_sleep("src.api.services.job_service.asyncio.sleep"):
            gen = service.stream_job_progress("job1")
            events = []
            async for event in gen:
                events.append(event)
                if event["status"] == "completed":
                    break
            assert len(events) >= 1
