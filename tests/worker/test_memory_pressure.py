"""Memory-pressure observability for the translation pipeline.

`MemoryMonitor` previously only recorded a peak and never acted on it, so
an OOM SIGKILL on Cloud Run appeared in the logs as the container simply
vanishing. It now emits one structured WARNING naming the job and the
pipeline phase once usage crosses a configurable fraction of the
container's cgroup memory limit.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from src.worker.doctranslator.format.pdf import high_level
from src.worker.doctranslator.utils import memory

GIB = 1024 * 1024 * 1024


def _make_monitor(limit_bytes: int | None, warn_fraction: float = 0.85):
    monitor = high_level.MemoryMonitor.__new__(high_level.MemoryMonitor)
    monitor.job_id = "job-42"
    monitor.memory_limit_bytes = limit_bytes
    monitor.warn_fraction = warn_fraction
    monitor._pressure_warned = False
    return monitor


class TestMemoryPressureWarning:
    def test_warns_once_above_the_threshold(self, caplog):
        monitor = _make_monitor(8 * GIB)
        high_level._set_phase("translate")

        with caplog.at_level("WARNING"):
            monitor._maybe_warn_pressure(int(7.5 * GIB))
            monitor._maybe_warn_pressure(int(7.9 * GIB))

        assert caplog.text.count("memory pressure") == 1
        assert "job_id=job-42" in caplog.text
        assert "phase=translate" in caplog.text

    def test_silent_below_the_threshold(self, caplog):
        monitor = _make_monitor(8 * GIB)

        with caplog.at_level("WARNING"):
            monitor._maybe_warn_pressure(4 * GIB)

        assert "memory pressure" not in caplog.text

    def test_disabled_without_a_cgroup_limit(self, caplog):
        monitor = _make_monitor(None)

        with caplog.at_level("WARNING"):
            monitor._maybe_warn_pressure(100 * GIB)

        assert "memory pressure" not in caplog.text

    def test_disabled_when_fraction_is_zero(self, caplog):
        monitor = _make_monitor(8 * GIB, warn_fraction=0.0)

        with caplog.at_level("WARNING"):
            monitor._maybe_warn_pressure(8 * GIB)

        assert "memory pressure" not in caplog.text


class TestContainerMemoryLimit:
    def test_reads_cgroup_v2_value(self, monkeypatch, tmp_path: Path):
        limit_file = tmp_path / "memory.max"
        limit_file.write_text("8589934592\n")
        monkeypatch.setattr(memory, "_CGROUP_LIMIT_PATHS", (str(limit_file),))

        assert memory.get_container_memory_limit_bytes() == 8 * GIB

    def test_unlimited_cgroup_v2_returns_none(self, monkeypatch, tmp_path: Path):
        limit_file = tmp_path / "memory.max"
        limit_file.write_text("max\n")
        monkeypatch.setattr(memory, "_CGROUP_LIMIT_PATHS", (str(limit_file),))

        assert memory.get_container_memory_limit_bytes() is None

    def test_cgroup_v1_sentinel_returns_none(self, monkeypatch, tmp_path: Path):
        limit_file = tmp_path / "memory.limit_in_bytes"
        limit_file.write_text("9223372036854771712\n")
        monkeypatch.setattr(memory, "_CGROUP_LIMIT_PATHS", (str(limit_file),))

        assert memory.get_container_memory_limit_bytes() is None

    def test_missing_files_return_none(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(memory, "_CGROUP_LIMIT_PATHS", (str(tmp_path / "nope"),))

        assert memory.get_container_memory_limit_bytes() is None


class TestPhaseTracking:
    def test_collect_garbage_records_the_phase(self):
        high_level._collect_garbage("render")
        assert high_level._get_phase() == "render"

    def test_phase_is_visible_from_another_thread(self):
        """The pipeline runs in an executor thread while MemoryMonitor reads
        the phase from its own monitor thread, so the phase must not be
        thread-local."""
        import threading

        high_level._set_phase("unknown")
        threading.Thread(target=high_level._set_phase, args=("typesetting",)).start()
        deadline = time.time() + 2
        while high_level._get_phase() != "typesetting" and time.time() < deadline:
            time.sleep(0.01)

        assert high_level._get_phase() == "typesetting"


class TestDispatchDeadlineClamp:
    """Cloud Tasks rejects HTTP-target deadlines above 30 minutes, and
    several .env files still carry 3600."""

    @pytest.mark.parametrize(
        ("configured", "expected_seconds"),
        [(1800, 1800), (3600, 1800), (600, 600)],
    )
    def test_enqueue_never_exceeds_the_cloud_tasks_maximum(
        self, configured, expected_seconds, monkeypatch
    ):
        from unittest.mock import MagicMock

        from src.api.services.cloud_tasks_service import CloudTasksService

        for name, value in (
            ("CLOUD_TASKS_PROJECT", "proj"),
            ("CLOUD_TASKS_LOCATION", "europe-west1"),
            ("CLOUD_TASKS_QUEUE", "translation-jobs"),
            ("CLOUD_TASKS_WORKER_URL", "https://worker.example/x"),
            ("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT", "tasks@proj.iam.gserviceaccount.com"),
            ("CLOUD_TASKS_DISPATCH_DEADLINE_SECONDS", configured),
        ):
            monkeypatch.setattr(
                f"src.api.services.cloud_tasks_service.settings.{name}", value
            )

        client = MagicMock()
        client.queue_path.return_value = (
            "projects/proj/locations/europe-west1/queues/translation-jobs"
        )
        client.create_task.return_value = MagicMock(name="created")

        CloudTasksService(client=client).enqueue_translate("job-1")

        task = client.create_task.call_args.kwargs["request"]["task"]
        assert task["dispatch_deadline"].seconds == expected_seconds
