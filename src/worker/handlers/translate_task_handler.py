"""Worker handler that runs the translation pipeline for a Cloud Task."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from src.config.tracing import tracer_pipeline
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository
from src.shared import job_status
from src.shared.schemas.tasks import TranslateTaskPayload
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator
from src.worker.services.temp_workspace_service import TempWorkspaceService

logger = logging.getLogger(__name__)


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

            # Re-check cancel immediately before heavy work
            latest = await self.bigquery.get_translation_job(job_id)
            if latest and str(latest.get("status")) == job_status.CANCELLED:
                logger.info("Job %s cancelled before pipeline start", job_id)
                return {
                    "job_id": job_id,
                    "status": job_status.CANCELLED,
                    "action": "noop",
                }

            job_data = latest or job
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
