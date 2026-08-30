"""Shared, in-process cache for language-independent document preparation.

When one source document is translated into multiple target languages
(a "batch" submitted via TranslationService.submit_translations), each
target language today runs as a fully independent PipelineOrchestrator job.
Several early pipeline steps are 100% language-independent and were
previously repeated once per target language even though they produce byte-
for-byte identical results:

  1. Downloading the source file from GCS.
  2. Converting DOCX -> PDF (LibreOffice subprocess).
  3. Auto-detecting the source language (when not explicitly provided).

This module lets sibling jobs (same `source_hash`) that execute concurrently
*in the same worker process* share that work exactly once, via a reference-
counted cache keyed on `source_hash`. The first job to ask for a given
`source_hash` does the real work and stores the result; concurrent siblings
await the same in-flight future instead of repeating it. The cached result
(a local file) is cleaned up once every job that requested it has either
copied what it needs or failed, so this does not leak across unrelated jobs.

Caveats (documented rather than hidden):
  - This only helps when sibling jobs run in the same process. In
    production, Cloud Tasks jobs may be dispatched to different Cloud Run
    instances, in which case there is no shared memory and each instance
    still does its own prep -- this is a best-effort optimization for
    same-instance concurrency (guaranteed in local/dev
    API_USE_BACKGROUND_PIPELINE=true mode, and a real but not guaranteed win
    in production depending on Cloud Run's request routing).
  - It intentionally does NOT attempt to share the deeper PDF-parsing /
    layout / OCR internals inside doctranslator's high_level.py -- that
    would require a much larger, riskier refactor of a ~1600-line module
    with no local test coverage for this change. This cache captures the
    safe, clearly-isolated language-independent prep steps only.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PreparedDocument:
    """Result of the shared, language-independent prep phase."""

    local_path: Path
    detected_source_language: str | None
    # Full char-weighted language distribution (implementation_plan.md
    # Phase C.5.1), populated alongside `detected_source_language` when
    # auto-detection ran. Empty when the source language was explicit.
    detected_language_distribution: Counter[str] = field(default_factory=Counter)


class _SharedPrepEntry:
    __slots__ = ("future", "refcount", "lock")

    def __init__(self) -> None:
        self.future: asyncio.Future[PreparedDocument] = (
            asyncio.get_running_loop().create_future()
        )
        self.refcount = 0
        self.lock = asyncio.Lock()


class SharedDocumentPrepCache:
    """Process-wide, reference-counted cache of in-flight/completed document prep."""

    def __init__(self) -> None:
        self._entries: dict[str, _SharedPrepEntry] = {}
        self._registry_lock = asyncio.Lock()

    async def get_or_prepare(
        self,
        *,
        source_hash: str,
        job_id: str,
        job_local_dir: Path,
        prepare_fn,
    ) -> PreparedDocument:
        """Return a PreparedDocument for `source_hash`, computing it at most once.

        `prepare_fn` is an async callable with no arguments that performs the
        real download/convert/detect work and returns a `PreparedDocument`
        whose `local_path` lives under a shared scratch location (not a
        specific job's workspace, since multiple jobs will read it).

        The caller is responsible for copying/using `local_path` into its own
        job workspace before calling `release()` -- once every requester has
        released, the shared file is deleted.
        """
        async with self._registry_lock:
            entry = self._entries.get(source_hash)
            is_first = entry is None
            if entry is None:
                entry = _SharedPrepEntry()
                self._entries[source_hash] = entry
            entry.refcount += 1

        if is_first:
            logger.info(
                "[shared_prep] job=%s is first to request source_hash=%s; preparing",
                job_id,
                source_hash,
            )
            try:
                result = await prepare_fn()
                entry.future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - propagate to all waiters
                entry.future.set_exception(exc)
                raise
        else:
            logger.info(
                "[shared_prep] job=%s reusing in-flight/cached prep for source_hash=%s",
                job_id,
                source_hash,
            )

        return await entry.future

    async def release(self, source_hash: str, job_id: str) -> None:
        """Decrement refcount for `source_hash`; delete the shared file when it hits zero."""
        async with self._registry_lock:
            entry = self._entries.get(source_hash)
            if entry is None:
                return
            entry.refcount -= 1
            should_cleanup = entry.refcount <= 0
            if should_cleanup:
                del self._entries[source_hash]

        if not should_cleanup:
            return

        if entry.future.done() and entry.future.exception() is None:
            prepared = entry.future.result()
            try:
                if prepared.local_path.exists():
                    if prepared.local_path.is_dir():
                        shutil.rmtree(prepared.local_path, ignore_errors=True)
                    else:
                        prepared.local_path.unlink(missing_ok=True)
                    # Clean up the parent scratch dir too if now empty.
                    parent = prepared.local_path.parent
                    try:
                        next(parent.iterdir())
                    except StopIteration:
                        parent.rmdir()
                    except (FileNotFoundError, OSError):
                        pass
            except Exception:
                logger.debug(
                    "[shared_prep] cleanup failed for source_hash=%s job=%s",
                    source_hash,
                    job_id,
                    exc_info=True,
                )
        logger.info(
            "[shared_prep] job=%s released source_hash=%s (cache entry removed)",
            job_id,
            source_hash,
        )


_shared_prep_cache = SharedDocumentPrepCache()


def get_shared_document_prep_cache() -> SharedDocumentPrepCache:
    return _shared_prep_cache
