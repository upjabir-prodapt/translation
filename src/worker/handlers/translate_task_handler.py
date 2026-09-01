"""Worker handler that runs the translation pipeline for a Cloud Task."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from src.config.constants import settings
from src.config.tracing import tracer_pipeline
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository
from src.shared import job_status
from src.shared.schemas.tasks import TranslateTaskPayload
from src.worker.lifecycle import is_shutting_down
from src.worker.services import job_lease
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator
from src.worker.services.temp_workspace_service import TempWorkspaceService

logger = logging.getLogger(__name__)

# Returned when SIGTERM arrived before the pipeline started. The route maps
# this to HTTP 503 so Cloud Tasks re-dispatches the job.
SHUTTING_DOWN_STATUS = "shutting_down"

# Returned when another live worker instance holds this job's lease. The
# route maps it to HTTP 503, which makes Cloud Tasks back off and retry:
# if the owner finishes, the retry sees a terminal status and returns 200;
# if the owner dies, the lease expires and the retry legitimately takes over.
LEASED_ELSEWHERE_STATUS = "leased_elsewhere"


class TranslateTaskHandler:
    """Load job state, honor cancel/idempotency, run PipelineOrchestrator."""

    def __init__(
        self,
        bigquery: BigQueryRepository | None = None,
        storage: APIStorageRepository | None = None,
        orchestrator: PipelineOrchestrator | None = None,
    ):
        self.bigquery = bigquery or BigQueryRepository()
        self.storage = storage or APIStorageRepository()
        self.orchestrator = orchestrator or PipelineOrchestrator(
            bigquery=self.bigquery,
            storage=self.storage,
        )
        self._workspaces = TempWorkspaceService()

    def _attach_trace(self, payload: TranslateTaskPayload):
        """Continue API submit span via W3C traceparent when present."""
        if not payload.traceparent:
            return otel_context.get_current()
        carrier: dict[str, str] = {"traceparent": payload.traceparent}
        if payload.tracestate:
            carrier["tracestate"] = payload.tracestate
        return TraceContextTextMapPropagator().extract(carrier)

    async def handle(self, payload: TranslateTaskPayload) -> dict[str, Any]:
        """Process one translate task.

        Returns a small status dict. Raises for transient failures (Tasks retry).
        Terminal no-ops return without raising so Cloud Tasks stops retrying.
        """
        job_id = payload.job_id
        parent_ctx = self._attach_trace(payload)

        with tracer_pipeline.start_as_current_span(
            "worker.translate_task",
            context=parent_ctx,
            kind=SpanKind.CONSUMER,
            attributes={"translation.job_id": job_id},
        ) as span:
            job = await self.bigquery.get_translation_job(job_id)
            if not job:
                span.set_status(trace.Status(trace.StatusCode.ERROR, "job not found"))
                # Permanent — do not retry forever
                logger.error("Job %s not found in BigQuery", job_id)
                return {"job_id": job_id, "status": "not_found", "action": "noop"}

            status = str(job.get("status") or "")
            if status in job_status.TERMINAL_STATUSES:
                logger.info("Job %s already terminal (%s); skipping", job_id, status)
                return {"job_id": job_id, "status": status, "action": "noop"}

            if status == job_status.CANCELLED:
                return {"job_id": job_id, "status": status, "action": "noop"}

            # Take the cross-instance lease *before* the crash-retry branch.
            # That branch rmtree's the job's scratch directory, which is
            # exactly the wrong thing to do to a job another instance is
            # still translating -- and with --concurrency=1 a Cloud Tasks
            # retry always lands on a *different* instance, so "status is
            # PROCESSING" on its own could never distinguish a crashed owner
            # from a live one.
            if not job_lease.acquire(job_id):
                span.set_attribute("translation.deferred_reason", "leased_elsewhere")
                return {
                    "job_id": job_id,
                    "status": LEASED_ELSEWHERE_STATUS,
                    "action": "deferred",
                }

            # queued or processing (crash retry): clean local scratch then run
            if status == job_status.PROCESSING:
                logger.warning(
                    "Job %s is processing — treating as crash retry, re-running",
                    job_id,
                )
                try:
                    self._workspaces.cleanup(job_id)
                except Exception as e:
                    logger.warning("Temp cleanup failed for %s: %s", job_id, e)

            try:
                # Re-check cancel immediately before heavy work
                latest = await self.bigquery.get_translation_job(job_id)
                if latest and str(latest.get("status")) == job_status.CANCELLED:
                    logger.info("Job %s cancelled before pipeline start", job_id)
                    return {
                        "job_id": job_id,
                        "status": job_status.CANCELLED,
                        "action": "noop",
                    }

                # Last gate before the pipeline: if SIGTERM has already arrived
                # this instance has ~10s left before Cloud Run SIGKILLs it, which
                # is nowhere near enough for a translation. Hand the job back so
                # Cloud Tasks re-dispatches it rather than losing it. A pipeline
                # that is *already* running is not interrupted -- that would need
                # a cancellation token threaded through the whole synchronous
                # pipeline; such a job is retried as a "crash retry" above.
                if is_shutting_down():
                    logger.warning(
                        "Worker is shutting down; deferring job %s for redelivery",
                        job_id,
                    )
                    span.set_attribute("translation.deferred_reason", "worker_shutdown")
                    return {
                        "job_id": job_id,
                        "status": SHUTTING_DOWN_STATUS,
                        "action": "deferred",
                    }

                job_data = latest or job
                async with self._lease_heartbeat(job_id):
                    await self.orchestrator.run(
                        job_id=job_id,
                        job_data=job_data,
                        parent_ctx=otel_context.get_current(),
                    )

                final = await self.bigquery.get_translation_job(job_id)
                final_status = str((final or {}).get("status") or "unknown")
                return {
                    "job_id": job_id,
                    "status": final_status,
                    "action": "ran",
                }
            finally:
                # Released on every exit path, including failure: a job whose
                # pipeline raised must be retryable immediately rather than
                # after JOB_LEASE_TTL_SECONDS.
                job_lease.release(job_id)

    @contextlib.asynccontextmanager
    async def _lease_heartbeat(self, job_id: str):
        """Keep the lease alive for the duration of the pipeline.

        A background task rather than a progress-callback hook:
        `_PipelineProgressTracker.update` is not called densely enough during
        typesetting and PDF rendering, which are precisely the long stages,
        so hanging the heartbeat off progress would let the lease lapse
        mid-job and invite the duplicate execution it exists to prevent.
        """
        # Floored rather than trusted: a configured 0 would turn the
        # heartbeat into a hot loop competing with the pipeline for the CPU.
        interval = max(0.01, float(settings.JOB_LEASE_REFRESH_SECONDS))

        async def _beat() -> None:
            while True:
                await asyncio.sleep(interval)
                # Runs in a thread: the Redis client is synchronous and the
                # pipeline is already saturating the event loop's executor.
                still_ours = await asyncio.to_thread(job_lease.refresh, job_id)
                if not still_ours:
                    logger.warning(
                        "Lost the lease for job %s while it was still running", job_id
                    )

        task = asyncio.create_task(_beat(), name=f"job-lease-{job_id}")
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
