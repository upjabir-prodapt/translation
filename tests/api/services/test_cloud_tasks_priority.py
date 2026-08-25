"""Queue selection for the two-tier (standard / high) Cloud Tasks setup.

Cloud Tasks has no per-task priority field, so tiers are separate queues.
These tests pin the routing rules and, importantly, assert that no user PII
leaks into the task body.
"""

import json
from unittest.mock import MagicMock

import pytest
from google.api_core import exceptions as gcp_exceptions
from src.api.services.cloud_tasks_service import CloudTasksService

STANDARD_QUEUE = "translation-job"
HIGH_QUEUE = "translation-job-high"


@pytest.fixture
def tasks_settings(monkeypatch):
    def _set(name, value):
        monkeypatch.setattr(
            f"src.api.services.cloud_tasks_service.settings.{name}", value
        )

    _set("CLOUD_TASKS_PROJECT", "proj")
    _set("CLOUD_TASKS_LOCATION", "europe-west1")
    _set("CLOUD_TASKS_QUEUE", STANDARD_QUEUE)
    _set("CLOUD_TASKS_QUEUE_HIGH", HIGH_QUEUE)
    _set("CLOUD_TASKS_WORKER_URL", "https://worker.example/internal/tasks/translate")
    _set("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT", "tasks@proj.iam.gserviceaccount.com")
    _set("CLOUD_TASKS_DISPATCH_DEADLINE_SECONDS", 1800)
    _set("IS_LOCAL", False)
    return _set


def _client():
    client = MagicMock()
    client.queue_path.side_effect = lambda p, loc, q: (
        f"projects/{p}/locations/{loc}/queues/{q}"
    )
    client.task_path.side_effect = lambda p, loc, q, t: (
        f"projects/{p}/locations/{loc}/queues/{q}/tasks/{t}"
    )
    created = MagicMock()
    created.name = "projects/proj/locations/europe-west1/queues/q/tasks/translate-x"
    client.create_task.return_value = created
    return client


def _parent_of_last_create(client):
    return client.create_task.call_args[1]["request"]["parent"]


def _body_of_last_create(client):
    task = client.create_task.call_args[1]["request"]["task"]
    return json.loads(task["http_request"]["body"].decode())


class TestQueueSelection:
    def test_high_priority_uses_high_queue(self, tasks_settings):  # noqa: ARG002
        client = _client()
        CloudTasksService(client=client).enqueue_translate("job-1", priority="high")
        assert HIGH_QUEUE in _parent_of_last_create(client)

    def test_standard_priority_uses_default_queue(self, tasks_settings):  # noqa: ARG002
        client = _client()
        CloudTasksService(client=client).enqueue_translate("job-1", priority="standard")
        assert _parent_of_last_create(client).endswith(STANDARD_QUEUE)

    def test_priority_none_defaults_to_standard(self, tasks_settings):  # noqa: ARG002
        """Back-compat: callers that don't pass priority keep working."""
        client = _client()
        CloudTasksService(client=client).enqueue_translate("job-1")
        assert _parent_of_last_create(client).endswith(STANDARD_QUEUE)

    def test_high_falls_back_when_unset(self, tasks_settings):
        """Config may land before the queue exists -- must not fail the job."""
        tasks_settings("CLOUD_TASKS_QUEUE_HIGH", "")
        client = _client()
        CloudTasksService(client=client).enqueue_translate("job-1", priority="high")
        assert _parent_of_last_create(client).endswith(STANDARD_QUEUE)

    def test_priority_is_case_insensitive(self, tasks_settings):  # noqa: ARG002
        client = _client()
        CloudTasksService(client=client).enqueue_translate("job-1", priority="HIGH")
        assert HIGH_QUEUE in _parent_of_last_create(client)


class TestTaskPayload:
    def test_payload_carries_priority_and_format(self, tasks_settings):  # noqa: ARG002
        client = _client()
        CloudTasksService(client=client).enqueue_translate(
            "job-1", priority="high", doc_format="txt"
        )
        body = _body_of_last_create(client)
        assert body["priority"] == "high"
        assert body["doc_format"] == "txt"
        assert body["job_id"] == "job-1"

    def test_payload_has_no_user_pii(self, tasks_settings):  # noqa: ARG002
        """Task bodies are persisted by Cloud Tasks and appear in logs.

        User identity must reach the worker via BigQuery (it re-reads the job),
        never via the task body.
        """
        client = _client()
        CloudTasksService(client=client).enqueue_translate(
            "job-1", priority="high", doc_format="txt"
        )
        body = _body_of_last_create(client)
        forbidden = {
            "user_id",
            "email",
            "organization",
            "business_unit",
            "cost_attribution",
        }
        assert forbidden.isdisjoint(body.keys())


class TestDeleteTranslateTask:
    def test_delete_uses_correct_queue(self, tasks_settings):  # noqa: ARG002
        client = _client()
        svc = CloudTasksService(client=client)
        assert svc.delete_translate_task("job-1", "high") is True
        assert HIGH_QUEUE in client.delete_task.call_args[1]["request"]["name"]

    def test_delete_without_priority_tries_both_queues(self, tasks_settings):  # noqa: ARG002
        """Priority may be unknown on older rows; the task could be on either."""
        client = _client()
        client.delete_task.side_effect = [gcp_exceptions.NotFound("nope"), None]
        svc = CloudTasksService(client=client)
        assert svc.delete_translate_task("job-1") is True
        assert client.delete_task.call_count == 2

    def test_delete_tolerates_not_found(self, tasks_settings):  # noqa: ARG002
        """Already dispatched or already gone -- not an error."""
        client = _client()
        client.delete_task.side_effect = gcp_exceptions.NotFound("gone")
        svc = CloudTasksService(client=client)
        assert svc.delete_translate_task("job-1", "standard") is False

    def test_delete_never_raises(self, tasks_settings):  # noqa: ARG002
        """A Cloud Tasks outage must not break cancellation."""
        client = _client()
        client.delete_task.side_effect = RuntimeError("cloud tasks down")
        svc = CloudTasksService(client=client)
        assert svc.delete_translate_task("job-1", "high") is False

    def test_delete_noop_in_local_mode(self, tasks_settings):
        """Local dispatch never creates a task, so there is nothing to delete."""
        tasks_settings("IS_LOCAL", True)
        tasks_settings("CLOUD_TASKS_WORKER_URL", "http://localhost:8001/x")
        client = _client()
        svc = CloudTasksService(client=client)
        assert svc.delete_translate_task("job-1") is False
        client.delete_task.assert_not_called()
