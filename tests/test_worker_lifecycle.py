"""
Unit tests for worker/core/lifecycle.py — WorkerLifecycle.

async_warmup (from loaders) is mocked to avoid GCS downloads.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.core.lifecycle import WorkerLifecycle


# ---------------------------------------------------------------------------
# Helpers: reset class state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_lifecycle():
    """Reset WorkerLifecycle state before each test."""
    WorkerLifecycle._initialized = False
    WorkerLifecycle._shutdown_event = None
    yield
    WorkerLifecycle._initialized = False
    WorkerLifecycle._shutdown_event = None


# ---------------------------------------------------------------------------
# is_initialized
# ---------------------------------------------------------------------------


class TestIsInitialized:
    def test_false_before_startup(self):
        assert WorkerLifecycle.is_initialized() is False

    def test_true_after_startup(self):
        WorkerLifecycle._initialized = True
        assert WorkerLifecycle.is_initialized() is True


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


class TestStartup:
    async def test_sets_initialized_flag(self):
        with patch("worker.core.lifecycle.async_warmup", new=AsyncMock()):
            await WorkerLifecycle.startup()
        assert WorkerLifecycle.is_initialized() is True

    async def test_creates_shutdown_event(self):
        with patch("worker.core.lifecycle.async_warmup", new=AsyncMock()):
            await WorkerLifecycle.startup()
        assert WorkerLifecycle._shutdown_event is not None

    async def test_already_initialized_skips(self):
        WorkerLifecycle._initialized = True
        warmup_mock = AsyncMock()
        with patch("worker.core.lifecycle.async_warmup", new=warmup_mock):
            await WorkerLifecycle.startup()
        warmup_mock.assert_not_called()

    async def test_warmup_failure_raises_runtime_error(self):
        with patch(
            "worker.core.lifecycle.async_warmup",
            side_effect=Exception("GCS unavailable"),
        ):
            with pytest.raises(RuntimeError, match="Worker startup failed"):
                await WorkerLifecycle.startup()

    async def test_initialized_false_after_warmup_failure(self):
        with patch(
            "worker.core.lifecycle.async_warmup",
            side_effect=Exception("fail"),
        ):
            with pytest.raises(RuntimeError):
                await WorkerLifecycle.startup()
        assert WorkerLifecycle.is_initialized() is False


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_not_initialized_skips_silently(self):
        # Should not raise
        await WorkerLifecycle.shutdown()

    async def test_clears_initialized_flag(self):
        WorkerLifecycle._initialized = True
        WorkerLifecycle._shutdown_event = asyncio.Event()
        await WorkerLifecycle.shutdown()
        assert WorkerLifecycle.is_initialized() is False

    async def test_sets_shutdown_event(self):
        WorkerLifecycle._initialized = True
        event = asyncio.Event()
        WorkerLifecycle._shutdown_event = event
        await WorkerLifecycle.shutdown()
        assert event.is_set()

    async def test_no_event_shutdown_still_clears(self):
        WorkerLifecycle._initialized = True
        WorkerLifecycle._shutdown_event = None
        await WorkerLifecycle.shutdown()
        assert WorkerLifecycle.is_initialized() is False

    async def test_exception_in_shutdown_propagates(self):
        WorkerLifecycle._initialized = True
        event = asyncio.Event()
        event.set = MagicMock(side_effect=RuntimeError("event failure"))
        WorkerLifecycle._shutdown_event = event

        with pytest.raises(RuntimeError, match="event failure"):
            await WorkerLifecycle.shutdown()


# ---------------------------------------------------------------------------
# wait_for_shutdown
# ---------------------------------------------------------------------------


class TestWaitForShutdown:
    async def test_returns_immediately_when_no_event(self):
        WorkerLifecycle._shutdown_event = None
        # Should complete without blocking
        await WorkerLifecycle.wait_for_shutdown()

    async def test_waits_until_event_set(self):
        event = asyncio.Event()
        WorkerLifecycle._shutdown_event = event
        event.set()  # Pre-set so it doesn't block
        await WorkerLifecycle.wait_for_shutdown()  # Should return immediately
