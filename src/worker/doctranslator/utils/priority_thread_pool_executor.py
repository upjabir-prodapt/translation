# thanks to:
# https://github.com/oleglpts/PriorityThreadPoolExecutor/blob/master/PriorityThreadPoolExecutor/__init__.py
# https://github.com/oleglpts/PriorityThreadPoolExecutor/issues/4

import atexit
import itertools
import logging
import queue
import secrets
import sys
import threading
import weakref
from concurrent.futures import _base
from concurrent.futures.thread import BrokenThreadPool
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.thread import _python_exit
from concurrent.futures.thread import _threads_queues
from concurrent.futures.thread import _WorkItem
from heapq import heappop
from heapq import heappush

from opentelemetry import context as otel_context

logger = logging.getLogger(__name__)

########################################################################################################################
#                                                Global variables                                                      #
########################################################################################################################

NULL_ENTRY = (sys.maxsize, _WorkItem(None, None, (), {}))
_shutdown = False

########################################################################################################################
#                                           Before system exit procedure                                               #
########################################################################################################################


def python_exit():
    """

    Cleanup before system exit

    """
    global _shutdown
    _shutdown = True
    items = list(_threads_queues.items())
    for _t, q in items:
        q.put(NULL_ENTRY)
    for t, _q in items:
        t.join()


# change default cleanup


atexit.unregister(_python_exit)
atexit.register(python_exit)


class PriorityQueue(queue.Queue):
    """Variant of Queue that retrieves open entries in priority order (lowest first).

    Entries are typically tuples of the form:  (priority number, data).
    """

    REMOVED = "<removed-task>"
    DEFAULT_PRIORITY = 100

    def _init(self, maxsize):
        self.queue = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        # heappush(self.queue, item)
        try:
            if item[1] in self.entry_finder:
                self.remove(item[1])
            count = next(self.counter)
            entry = [item[0], count, item[1]]
            self.entry_finder[item[1]] = entry
            heappush(self.queue, entry)
        except TypeError:  # handle item==None
            self._put((self.DEFAULT_PRIORITY, None))

    def remove(self, task):
        """
        This simply replaces the data with the REMOVED value,
        which will get cleared out once _get reaches it.
        """
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def _get(self):
        while self.queue:
            entry = heappop(self.queue)
            if entry[2] is not self.REMOVED:
                del self.entry_finder[entry[2]]
                return entry
        return None


def _process_work_item(work_item, executor_reference, work_queue):
    """
    Process a single work item from the queue.

    Returns True if the worker should exit, False to continue the loop.
    """
    if work_item[2] is not None:
        work_item[2].run()
        # Delete references to object. See issue16284
        del work_item

        # attempt to increment idle count
        executor = executor_reference()
        if executor is not None:
            executor._idle_semaphore.release()
        del executor
        return False

    executor = executor_reference()
    # Exit if:
    #   - The interpreter is shutting down OR
    #   - The executor that owns the worker has been collected OR
    #   - The executor that owns the worker has been shutdown.
    if _shutdown or executor is None or executor._shutdown:
        # Flag the executor as shutting down as early as possible if it
        # is not gc-ed yet.
        if executor is not None:
            executor._shutdown = True
        # Notice other workers
        work_queue.put(None)
        return True
    del executor
    return False


def _run_initializer(executor_reference, initializer, initargs):
    """
    Run the thread initializer, handling failures gracefully.

    Returns True if initializer succeeded (or was absent), False on failure.
    """
    if initializer is None:
        return True
    try:
        initializer(*initargs)
        return True
    except Exception:  # noqa: BLE001 - broad catch is intentional: initializer may raise any exception type
        _base.LOGGER.critical("Exception in initializer:", exc_info=True)
        executor = executor_reference()
        if executor is not None:
            executor._initializer_failed()
        return False


def _worker(executor_reference, work_queue, initializer, initargs):
    if not _run_initializer(executor_reference, initializer, initargs):
        return
    try:
        while True:
            work_item = work_queue.get(block=True)
            try:
                should_exit = _process_work_item(
                    work_item, executor_reference, work_queue
                )
                if should_exit:
                    return
            finally:
                work_queue.task_done()
    except Exception:  # noqa: BLE001 - broad catch is intentional: re-raised immediately after logging
        _base.LOGGER.critical("Exception in worker", exc_info=True)
        raise


class PriorityThreadPoolExecutor(ThreadPoolExecutor):
    """
    Thread pool executor with priority queue (priorities must be different, lowest first)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # change work queue type to queue.PriorityQueue
        self._work_queue: PriorityQueue = PriorityQueue()
        self._all_future = []

    def submit(self, fn, *args, **kwargs):
        """

        Sending the function to the execution queue

        :param fn: function being executed
        :type fn: callable
        :param args: function's positional arguments
        :param kwargs: function's keywords arguments
        :return: future instance
        :rtype: _base.Future

        Added keyword:

        - priority (integer later sys.maxsize)

        """
        # Propagate the active OTel span context into the worker thread.
        # ThreadPoolExecutor does not copy contextvars automatically, so without
        # this every gemini.generate_content / llm.translate_batch span becomes
        # an orphaned root span in GCP instead of a child of pipeline.translation.
        _ctx = otel_context.get_current()
        _fn = fn

        def _fn_with_otel(*a, **kw):
            token = otel_context.attach(_ctx)
            try:
                return _fn(*a, **kw)
            finally:
                otel_context.detach(token)

        fn = _fn_with_otel

        with self._shutdown_lock:
            if self._broken:
                raise BrokenThreadPool(self._broken)

            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _shutdown:
                raise RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )

            priority = kwargs.get("priority", secrets.randbelow(sys.maxsize))
            if "priority" in kwargs:
                del kwargs["priority"]

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._work_queue.put((priority, w))
            self._adjust_thread_count()
            self._all_future.append(f)
            return f

    def _adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = f"{self._thread_name_prefix or self}_{num_threads:d}"
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
            )
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue

    def _drain_and_cancel_futures(self):
        """Drain all pending work items from the queue and cancel their futures."""
        while True:
            try:
                work_item = self._work_queue.get_nowait()
            except queue.Empty:
                break
            if work_item is not None:
                work_item.future.cancel()

    def _join_all_threads(self):
        """Signal all worker threads to stop and wait for them to finish."""
        logger.debug(f"Waiting for all thread done {self._thread_name_prefix or self}")
        for t in self._threads:
            self._work_queue.put(None)
            t.join()

    def shutdown(self, wait=True, *, cancel_futures=False):
        logger.debug(f"Shutting down executor {self._thread_name_prefix or self}")
        if wait:
            logger.debug(
                f"Waiting for all tasks done {self._thread_name_prefix or self}"
            )
            self._work_queue.join()
            logger.debug(f"All tasks done {self._thread_name_prefix or self}")

        with self._shutdown_lock:
            self._shutdown = True
            if cancel_futures:
                # Drain all work items from the queue, and then cancel their
                # associated futures.
                self._drain_and_cancel_futures()

            # Send a wake-up to prevent threads calling
            # _work_queue.get(block=True) from permanently blocking.
            self._work_queue.put(None)
        if wait:
            self._join_all_threads()
        logger.debug(f"shutdown finish {self._thread_name_prefix or self}")

    def __del__(self):
        for f in self._all_future:
            if f.done() and not f.cancelled():
                try:
                    f.result()
                except Exception as e:
                    logger.warning(f"Exception in future {f}: {e}", exc_info=True)
