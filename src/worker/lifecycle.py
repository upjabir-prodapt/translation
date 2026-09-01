"""Worker shutdown state and in-flight job accounting.

Cloud Run sends SIGTERM and then SIGKILLs the container after roughly 10
seconds. The translation pipeline runs synchronously inside
``loop.run_in_executor(...)``, which uvicorn cannot cancel, so before this
module existed *every* revision rollout, instance recycle or scale-down
with a job in flight produced a SIGKILL that was indistinguishable in
Cloud Logging from an OOM kill.

This module provides the two things needed to tell those apart and to
avoid losing work that had not started yet:

* an in-flight job counter, so the shutdown log line states exactly how
  many jobs were running when SIGTERM arrived;
* a shutdown flag set the moment SIGTERM is received (not when the
  lifespan shutdown hook finally runs, which is *after* uvicorn has
  drained connections), so a task that has not begun heavy work can be
  handed back to Cloud Tasks for redelivery.

Interrupting an already-running pipeline is explicitly out of scope: it
would require threading a cancellation token through the whole
synchronous pipeline. Such a job is re-dispatched by Cloud Tasks and
picked up as a "crash retry" by TranslateTaskHandler.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_in_flight_jobs: set[str] = set()
_shutting_down = threading.Event()

# Stable identity for this worker process, generated once at import.
#
# The job lease is a compare-and-swap on this value: only the instance whose
# owner ID is still stored against the job may refresh or release it. A
# process that generated a fresh ID per acquisition could release a lease
# that had already expired and been taken over by another instance. The PID
# is folded in purely to make the value legible in logs.
_OWNER_ID = f"{os.getpid()}-{uuid.uuid4()}"


def owner_id() -> str:
    """Identity of this worker process, used as the job-lease owner token."""
    return _OWNER_ID


# Signal handlers we replaced, so uvicorn's own graceful-shutdown handling
# still runs after we record the shutdown.
_previous_handlers: dict[int, object] = {}


def in_flight_job_count() -> int:
    """Number of translation jobs currently executing in this process."""
    with _lock:
        return len(_in_flight_jobs)


def in_flight_job_ids() -> list[str]:
    with _lock:
        return sorted(_in_flight_jobs)


@contextmanager
def job_in_flight(job_id: str) -> Iterator[None]:
    """Count `job_id` as in flight for the duration of the block."""
    with _lock:
        _in_flight_jobs.add(job_id)
    try:
        yield
    finally:
        with _lock:
            _in_flight_jobs.discard(job_id)


def is_shutting_down() -> bool:
    return _shutting_down.is_set()


def request_shutdown(reason: str) -> None:
    """Mark the worker as shutting down and log the in-flight job census."""
    already = _shutting_down.is_set()
    _shutting_down.set()
    if already:
        return
    job_ids = in_flight_job_ids()
    logger.warning(
        "worker received shutdown, reason=%s in_flight_jobs=%d job_ids=%s",
        reason,
        len(job_ids),
        ",".join(job_ids) or "-",
    )


def install_signal_handlers() -> None:
    """Record SIGTERM/SIGINT immediately, then chain to uvicorn's handler.

    uvicorn only runs the FastAPI lifespan shutdown hook *after* it has
    drained open connections, which is far too late to stop accepting new
    work. Installing our own handler on top of uvicorn's (rather than
    instead of it) gives the handler layer an accurate shutdown flag
    without changing uvicorn's own graceful-shutdown behaviour.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(sig)
            signal.signal(sig, _make_handler(previous))
            _previous_handlers[sig] = previous
        except (ValueError, OSError):
            # Not on the main thread (e.g. under a test runner) — the
            # shutdown flag simply stays unset, which is the old behaviour.
            logger.debug("Could not install handler for signal %s", sig)


def _make_handler(previous):
    def _handler(signum: int, frame: FrameType | None):
        request_shutdown(signal.Signals(signum).name)
        if callable(previous):
            previous(signum, frame)

    return _handler


def reset_for_tests() -> None:
    """Clear module state. Test-only."""
    _shutting_down.clear()
    with _lock:
        _in_flight_jobs.clear()
    _previous_handlers.clear()
