"""Async helpers for proper async/sync bridging."""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Coroutine


class AsyncBridge:
    """Bridge between async and sync contexts."""

    @staticmethod
    def run_async[T](coro: Coroutine[None, None, T]) -> T:
        """Run an async coroutine from a sync context.

        This handles both cases:
        - No event loop running: creates one and runs the coroutine
        - Event loop already running: creates a new loop in a thread

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - need to schedule as a task
            # This returns a Task, not the result directly
            # Caller should await this if they're in async context
            return loop.create_task(coro)  # type: ignore[return-value]
        except RuntimeError:
            # No event loop running - safe to create one
            return asyncio.run(coro)

    @staticmethod
    def ensure_sync_result[T](coro: Coroutine[None, None, T]) -> T:
        """Ensure we get a sync result, even from async context.

        This should only be called from truly sync contexts.
        If called from async, it will return a Task.
        """
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Already in async context - this shouldn't happen
            # if used correctly, but handle gracefully
            raise RuntimeError(
                "Cannot get sync result while in async context. Use 'await' instead."
            ) from None


def sync_to_async[T](
    func: Callable[..., T],
    *args,
    **kwargs,
) -> Awaitable[T]:
    """Run a sync function in an executor thread.

    Args:
        func: The synchronous function to run
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        A Future that resolves to the result of func
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args, **kwargs)
