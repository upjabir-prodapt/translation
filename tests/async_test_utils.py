"""Shared helpers for async unit tests."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import patch


async def noop_async_sleep(*_args, **_kwargs) -> None:
    """No-op stand-in for asyncio.sleep in tests."""
    return None


def close_coroutine(coro) -> None:
    """Close a coroutine passed to a mocked asyncio.run()."""
    if coro is not None and hasattr(coro, "close") and asyncio.iscoroutine(coro):
        coro.close()


def mock_asyncio_run(coro):
    """Fake asyncio.run for sync entrypoint tests."""
    close_coroutine(coro)
    return None


@contextmanager
def patch_asyncio_sleep(target: str = "asyncio.sleep"):
    """Patch asyncio.sleep with an awaitable no-op."""
    with patch(target, noop_async_sleep):
        yield
