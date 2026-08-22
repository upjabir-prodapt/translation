"""Common utility functions for DocTranslator."""

import itertools
import multiprocessing as mp
import threading
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")

# Process pool management
_process_pool = None
_process_pool_lock = threading.Lock()
_ENABLE_PROCESS_POOL = False


def enable_process_pool():
    """Enable process pool for parallel processing.

    Development and Testing ONLY API.
    """
    global _ENABLE_PROCESS_POOL
    _ENABLE_PROCESS_POOL = True


def get_process_pool():
    """Get the process pool if enabled.

    Returns:
        multiprocessing.Pool or None: Process pool if enabled, None otherwise.
    """
    if not _ENABLE_PROCESS_POOL:
        return None
    global _process_pool
    with _process_pool_lock:
        if _process_pool is None:
            # Create pool only in main process
            if mp.current_process().name != "MainProcess":
                return None
            _process_pool = mp.Pool()
        return _process_pool


def close_process_pool():
    """Close the process pool if enabled.

    Returns:
        None or bool: None if not enabled, True if closed successfully.
    """
    if not _ENABLE_PROCESS_POOL:
        return None
    global _process_pool
    with _process_pool_lock:
        if _process_pool:
            _process_pool.close()
            _process_pool.join()
            _process_pool = None


def batched[T](
    iterable: Iterable[T], n: int, *, strict: bool = False
) -> Iterable[tuple[T, ...]]:
    """Batch an iterable into tuples of length n.

    Args:
        iterable: The iterable to batch.
        n: Batch size.
        strict: If True, raise ValueError for incomplete final batch.

    Returns:
        Iterator of tuples, each containing n items.

    Raises:
        ValueError: If n < 1 or if strict=True and final batch is incomplete.

    Example:
        batched('ABCDEFG', 3) → ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch
