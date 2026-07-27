"""Translation service for handling document translation requests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import uuid
from datetime import UTC
from datetime import datetime
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from src.api.exceptions import ValidationError
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.responses import MultiTranslateJobResponse
from src.api.schemas.responses import MultiTranslateResponse
from src.api.schemas.responses import TranslateResponse
from src.api.services.cloud_tasks_service import CloudTasksService
from src.api.utils.pdf_validator import PDFValidator
from src.config.constants import settings
from src.config.tracing import tracer_pipeline
from src.config.translation_routing import normalize_domain
from src.config.translation_routing import normalize_language
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository
from src.shared import job_status

logger = logging.getLogger(__name__)


def _current_traceparent() -> tuple[str | None, str | None]:
    """Extract W3C trace context from the current span for Cloud Tasks payload."""
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent"), carrier.get("tracestate")


class TranslationService:
    """Service for handling translation requests."""

    def __init__(
        self,
        storage: APIStorageRepository | None = None,
        bigquery: BigQueryRepository | None = None,
        orchestrator=None,
        cloud_tasks: CloudTasksService | None = None,
    ):
        self.storage = storage or APIStorageRepository()
        self.bigquery = bigquery or BigQueryRepository()
        self._orchestrator = orchestrator
        self._cloud_tasks = cloud_tasks
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def orchestrator(self):
        """Lazy local-pipeline orchestrator (dev only: API_USE_BACKGROUND_PIPELINE)."""
        if not settings.API_USE_BACKGROUND_PIPELINE:
            raise RuntimeError(
                "In-process pipeline is disabled; use Cloud Tasks worker mode"
            )
        if self._orchestrator is None:
            from src.worker.services.pipeline_orchestrator import PipelineOrchestrator

            self._orchestrator = PipelineOrchestrator(
                bigquery=self.bigquery,
                storage=self.storage,
            )
        return self._orchestrator

    @property
    def cloud_tasks(self) -> CloudTasksService:
        if self._cloud_tasks is None:
            self._cloud_tasks = CloudTasksService()
        return self._cloud_tasks

    async def submit_translation(self, request: TranslateRequest) -> TranslateResponse:
        """Submit a document for translation."""
        job_id = str(uuid.uuid4())

        with tracer_pipeline.start_as_current_span(
            "translation_service.submit",
            kind=SpanKind.INTERNAL,
            attributes={"translation.job_id": job_id},
        ):
            return await self._do_submit(request, job_id)

    async def submit_translations(
        self, requests: list[TranslateRequest]
    ) -> MultiTranslateResponse:
        """Submit independent translation jobs sharing one source document."""
        if not requests:
            raise ValidationError("At least one target language is required")

        batch_id = str(uuid.uuid4())
        first_request = requests[0]
        try:
            content = base64.b64decode(first_request.document.content)
        except Exception as e:
            raise ValidationError(
                "Failed to decode document content", "document.content"
            ) from e

        if first_request.document.format == "pdf":
            _, metadata = PDFValidator.validate_pdf_bytes(
                content, first_request.document.filename
            )
        else:
            metadata = {
                "filename": first_request.document.filename,
                "size_bytes": len(content),
                "checksum": hashlib.sha256(content).hexdigest(),
                "page_count": None,
            }

        source_hash = hashlib.sha256(content).hexdigest()
        prepared: list[
            tuple[str, int, TranslateRequest, dict[str, Any], Any, bool]
        ] = []
        for batch_index, request in enumerate(requests):
            config = self._normalize_config(request)
            job_id = str(uuid.uuid4())
            config["job_id"] = job_id
            cached_job = await self.bigquery.get_completed_job_by_hash(
                source_hash=source_hash,
                lang_out=config["lang_out"],
                domain=config["domain"],
            )
            is_cached = bool(cached_job and cached_job.get("result"))
            prepared.append(
                (job_id, batch_index, request, config, cached_job, is_cached)
            )

        has_cache_miss = any(not is_cached for *_, is_cached in prepared)
        input_gs_uri = ""
        if has_cache_miss:
            input_gs_uri = await self.storage.upload_input_pdf(
                file_content=content,
                filename=metadata["filename"],
                job_id=f"batches/{batch_id}",
            )

        now = datetime.now(UTC)
        job_records: list[tuple[dict[str, Any], bool]] = []
        for job_id, batch_index, request, config, cached_job, is_cached in prepared:
            source_uri = input_gs_uri
            if is_cached:
                source_uri = (cached_job.get("source_document") or {}).get(
                    "gcs_uri", ""
                )
            job_data = {
                "job_id": job_id,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "status": job_status.COMPLETED if is_cached else job_status.QUEUED,
                "source_document": {
                    "gcs_uri": source_uri,
                    "format": request.document.format,
                    "page_count": metadata.get("page_count"),
                    "source_language": config["lang_in"],
                    "original_filename": metadata["filename"],
                    "output_filename": metadata["filename"],
                    "file_size_bytes": metadata["size_bytes"],
                    "checksum": metadata["checksum"],
                },
                "translation_config": {
                    "source_language": config["lang_in"],
                    "target_language": config["lang_out"],
                    "domain": config["domain"],
                },
                "cost_attribution": request.cost_attribution.model_dump(),
                "processing_options": request.processing_options.model_dump(),
                "error_message": None,
                "result": cached_job.get("result") if is_cached else None,
                "config": config,
                "source_hash": source_hash,
                "submitted_at": now,
                "completed_at": now if is_cached else None,
            }
            await self.bigquery.upsert_translation_job(job_data)
            job_records.append((job_data, is_cached))

        parent_ctx = otel_context.get_current()
        responses: list[MultiTranslateJobResponse] = []
        for job_data, is_cached in job_records:
            status = job_data["status"]
            if not is_cached:
                try:
                    await self._schedule_background_pipeline(
                        job_data["job_id"], job_data, parent_ctx
                    )
                except Exception:
                    status = job_status.FAILED
            responses.append(
                MultiTranslateJobResponse(
                    job_id=job_data["job_id"],
                    target_language=job_data["translation_config"]["target_language"],
                    status=status,
                    status_url=f"/api/v1/translate/{job_data['job_id']}",
                )
            )

        logger.info(
            "Submitted translation batch %s with %d jobs", batch_id, len(responses)
        )
        return MultiTranslateResponse(batch_id=batch_id, jobs=responses)

    async def _do_submit(
        self, request: TranslateRequest, job_id: str
    ) -> TranslateResponse:
        try:
            try:
                content = base64.b64decode(request.document.content)
            except Exception as e:
                raise ValidationError(
                    "Failed to decode document content", "document.content"
                ) from e

            metadata: dict[str, Any]
            if request.document.format == "pdf":
                _, metadata = PDFValidator.validate_pdf_bytes(
                    content, request.document.filename
                )
            else:
                metadata = {
                    "filename": request.document.filename,
                    "size_bytes": len(content),
                    "checksum": hashlib.sha256(content).hexdigest(),
                    "page_count": None,
                }

            source_hash = hashlib.sha256(content).hexdigest()

            config = self._normalize_config(request)
            config["job_id"] = job_id

            cached_job = await self.bigquery.get_completed_job_by_hash(
                source_hash=source_hash,
                lang_out=config["lang_out"],
                domain=config["domain"],
            )
            if cached_job and cached_job.get("result"):
                return await self._serve_from_cache(
                    request, job_id, metadata, source_hash, config, cached_job
                )

            input_gs_uri = await self.storage.upload_input_pdf(
                file_content=content,
                filename=metadata["filename"],
                job_id=job_id,
            )

            output_filename = metadata["filename"]

            now = datetime.now(UTC)
            job_data = {
                "job_id": job_id,
                "status": job_status.QUEUED,
                "progress": 0.0,
                "source_document": {
                    "gcs_uri": input_gs_uri,
                    "format": request.document.format,
                    "page_count": metadata.get("page_count"),
                    "source_language": config["lang_in"],
                    "original_filename": metadata["filename"],
                    "output_filename": output_filename,
                    "file_size_bytes": metadata["size_bytes"],
                    "checksum": metadata["checksum"],
                },
                "translation_config": {
                    "source_language": config["lang_in"],
                    "target_language": config["lang_out"],
                    "domain": config["domain"],
                },
                "cost_attribution": request.cost_attribution.model_dump(),
                "processing_options": {
                    "enable_dlp": request.processing_options.enable_dlp,
                    "enable_chunking": request.processing_options.enable_chunking,
                    "priority": request.processing_options.priority,
                },
                "error_message": None,
                "result": None,
                "timestamps": {
                    "submitted_at": now,
                    "completed_at": None,
                },
                "config": config,
                "source_hash": source_hash,
                "submitted_at": now,
                "completed_at": None,
            }

            await self.bigquery.upsert_translation_job(job_data)

            parent_ctx = otel_context.get_current()
            await self._schedule_background_pipeline(job_id, job_data, parent_ctx)

            logger.info("Submitted translation job %s", job_id)

            return TranslateResponse(
                job_id=job_id,
                status=job_status.QUEUED,
                status_url=f"/api/v1/translate/{job_id}",
            )

        except Exception as e:
            logger.error("Failed to submit translation: %s", e)
            try:
                await self.storage.delete_job_files(job_id)
            except Exception:
                pass
            raise

    async def _serve_from_cache(
        self,
        request: TranslateRequest,
        job_id: str,
        metadata: dict[str, Any],
        source_hash: str,
        config: dict[str, Any],
        cached_job: dict[str, Any],
    ) -> TranslateResponse:
        """Write a pre-completed job record reusing the cached result."""
        now = datetime.now(UTC)
        job_data = {
            "job_id": job_id,
            "status": job_status.COMPLETED,
            "source_document": {
                "gcs_uri": (cached_job.get("source_document") or {}).get("gcs_uri", ""),
                "format": request.document.format,
                "page_count": metadata.get("page_count"),
                "source_language": config["lang_in"],
                "original_filename": metadata["filename"],
                "output_filename": metadata["filename"],
                "file_size_bytes": metadata["size_bytes"],
                "checksum": source_hash,
            },
            "translation_config": {
                "source_language": config["lang_in"],
                "target_language": config["lang_out"],
                "domain": config["domain"],
            },
            "cost_attribution": request.cost_attribution.model_dump(),
            "processing_options": {
                "enable_dlp": request.processing_options.enable_dlp,
                "enable_chunking": request.processing_options.enable_chunking,
                "priority": request.processing_options.priority,
            },
            "error_message": None,
            "result": cached_job.get("result"),
            "config": config,
            "source_hash": source_hash,
            "submitted_at": now,
            "completed_at": now,
        }
        await self.bigquery.upsert_translation_job(job_data)
        logger.info(
            "Cache hit for job %s: reusing result from %s",
            job_id,
            cached_job.get("job_id"),
        )
        return TranslateResponse(
            job_id=job_id,
            status=job_status.COMPLETED,
            status_url=f"/api/v1/translate/{job_id}",
        )

    def _normalize_config(self, request: TranslateRequest) -> dict[str, Any]:
        """Normalize translation configuration."""
        try:
            domain = normalize_domain(request.translation_config.domain)
        except ValueError as e:
            raise ValidationError(str(e), field="translation_config.domain") from e

        try:
            lang_out = normalize_language(request.translation_config.target_language)
        except ValueError as e:
            raise ValidationError(
                str(e), field="translation_config.target_language"
            ) from e

        lang_in = request.translation_config.source_language
        if lang_in:
            try:
                lang_in = normalize_language(lang_in)
            except ValueError as e:
                raise ValidationError(
                    str(e), field="translation_config.source_language"
                ) from e
        return {
            "lang_in": lang_in or "auto",
            "lang_out": lang_out,
            "domain": domain,
        }

    async def _mark_enqueue_failed(self, job_id: str, error: Exception) -> None:
        """Compensate: job was queued in BQ but Cloud Tasks enqueue failed."""
        now = datetime.now(UTC)
        message = f"Failed to enqueue Cloud Task: {error}"
        try:
            await self.bigquery.patch_translation_job(
                job_id,
                {
                    "status": job_status.FAILED,
                    "error_message": message,
                    "completed_at": now,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to mark job %s as failed after enqueue error: %s", job_id, e
            )

    async def _schedule_background_pipeline(
        self,
        job_id: str,
        job_data: dict[str, Any],
        parent_ctx=None,
    ) -> None:
        """Schedule translation via in-process task (local) or Cloud Tasks (prod)."""
        if settings.API_USE_BACKGROUND_PIPELINE:
            token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
            try:
                task = asyncio.create_task(
                    self.orchestrator.run(
                        job_id=job_id, job_data=job_data, parent_ctx=parent_ctx
                    )
                )
            finally:
                if token is not None:
                    otel_context.detach(token)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            logger.info("Scheduled in-process translation pipeline for job %s", job_id)
            return

        traceparent, tracestate = _current_traceparent()
        try:
            await asyncio.to_thread(
                self.cloud_tasks.enqueue_translate,
                job_id,
                traceparent=traceparent,
                tracestate=tracestate,
            )
        except Exception as e:
            logger.error("Cloud Tasks enqueue failed for job %s: %s", job_id, e)
            await self._mark_enqueue_failed(job_id, e)
            raise RuntimeError(f"Failed to enqueue translation job: {e}") from e

        logger.info("Enqueued Cloud Tasks translation for job %s", job_id)
