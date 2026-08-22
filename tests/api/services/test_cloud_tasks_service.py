"""Unit tests for Cloud Tasks enqueue client."""

from unittest.mock import MagicMock

import pytest
from google.api_core import exceptions as gcp_exceptions
from src.api.services.cloud_tasks_service import CloudTasksService


@pytest.fixture
def tasks_settings(monkeypatch):
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_PROJECT",
        "proj",
    )
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_LOCATION",
        "europe-west1",
    )
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_QUEUE",
        "translation-jobs",
    )
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_WORKER_URL",
        "https://worker.example/internal/tasks/translate",
    )
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_OIDC_SERVICE_ACCOUNT",
        "tasks@proj.iam.gserviceaccount.com",
    )
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_DISPATCH_DEADLINE_SECONDS",
        1800,
    )


def test_task_id_for_job():
    assert CloudTasksService.task_id_for_job("abc-123").startswith("translate-")


def test_enqueue_translate_success(tasks_settings):  # noqa: ARG001
    client = MagicMock()
    client.queue_path.return_value = (
        "projects/proj/locations/europe-west1/queues/translation-jobs"
    )
    created = MagicMock()
    created.name = (
        "projects/proj/locations/europe-west1/queues/translation-jobs/tasks/translate-x"
    )
    client.create_task.return_value = created

    svc = CloudTasksService(client=client)
    name = svc.enqueue_translate("job-1", traceparent="00-abc-01")
    assert "tasks/" in name
    client.create_task.assert_called_once()


def test_enqueue_already_exists_is_ok(tasks_settings):  # noqa: ARG001
    client = MagicMock()
    client.queue_path.return_value = (
        "projects/proj/locations/europe-west1/queues/translation-jobs"
    )
    client.create_task.side_effect = gcp_exceptions.AlreadyExists("exists")

    svc = CloudTasksService(client=client)
    name = svc.enqueue_translate("job-1")
    assert "translate-job-1" in name


def test_enqueue_requires_queue(monkeypatch):
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_WORKER_URL",
        "https://worker.example/internal/tasks/translate",
    )
    monkeypatch.setattr(
        "src.api.services.cloud_tasks_service.settings.CLOUD_TASKS_QUEUE", ""
    )
    svc = CloudTasksService(client=MagicMock())
    with pytest.raises(RuntimeError, match="CLOUD_TASKS_QUEUE"):
        svc.enqueue_translate("job-1")
