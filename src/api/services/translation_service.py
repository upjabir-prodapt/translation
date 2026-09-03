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
from src.api.schemas.requests import DocumentInput
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.responses import MultiTranslateJobResponse
from src.api.schemas.responses import MultiTranslateResponse
from src.api.schemas.responses import TranslateResponse
from src.api.services.cloud_tasks_service import PRIORITY_HIGH
from src.api.services.cloud_tasks_service import PRIORITY_STANDARD
from src.api.services.cloud_tasks_service import CloudTasksService
from src.api.utils.document_validator import DocumentValidator
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


def _validate_and_describe(content: bytes, document: DocumentInput) -> dict[str, Any]:
    """Validate decoded document bytes and return descriptive metadata.

    Single dispatch point for PDF/DOCX/TXT validation used by both
    `submit_translations` and `_do_submit` (implementation_plan.md A.3.2)
    -- previously each had its own copy-pasted `else:` branch that never
    validated DOCX/TXT content at all (no password/structure/size check),
    which is how a password-protected, ZIP-corrupted, or oversized
    DOCX/TXT could reach GCS upload and BigQuery job creation unchecked.
    """
    if document.format == "pdf":
        _, metadata = PDFValidator.validate_pdf_bytes(content, document.filename)
        return metadata
    if document.format == "docx":
        return DocumentValidator.validate_docx_bytes(content, document.filename)
    if document.format == "txt":
        return DocumentValidator.validate_txt_bytes(content, document.filename)
    raise ValidationError(
        f"Unsupported document format: {document.format}", "document.format"
    )


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

        metadata = _validate_and_describe(content, first_request.document)

        # source_hash is retained for the in-process shared_document_prep cache
        # (dedupes download/parse work across sibling jobs in one multi-target-
        # language batch). It is no longer used for a cross-job result cache --
        # see docs/local-testing-guide.md / project history: a whole-document
        # BigQuery result cache was removed because it silently served stale
        # translations whenever only part of a document changed, and it was
        # redundant with the finer-grained Redis (Memorystore via PSC)
        # per-text-batch LLM cache in
        # src/worker/doctranslator/translator/translation_cache.py.
        source_hash = hashlib.sha256(content).hexdigest()

        # implementation_plan.md D.5 (EC-15): resolve any duplicate BEFORE
        # allocating a new job_id for that request, so a double-submitted
        # batch reuses the existing job(s) instead of creating siblings
        # that both upload/process the identical document.
        prepared: list[tuple[str, int, TranslateRequest, dict[str, Any]]] = []
        duplicate_responses: dict[int, MultiTranslateJobResponse] = {}
        for batch_index, request in enumerate(requests):
            config = self._normalize_config(request)
            duplicate = await self._find_duplicate_job(
                request, source_hash, config["lang_out"], config["domain"]
            )
            if duplicate is not None:
                logger.info(
                    "Duplicate submission detected for user %s; returning "
                    "existing job %s instead of creating a new one",
                    request.cost_attribution.user_id,
                    duplicate["job_id"],
                )
                duplicate_responses[batch_index] = MultiTranslateJobResponse(
                    job_id=duplicate["job_id"],
                    target_language=config["lang_out"],
                    status=str(duplicate.get("status", job_status.QUEUED)),
                    status_url=f"/api/v1/translate/{duplicate['job_id']}",
                    is_duplicate=True,
                )
                continue
            job_id = str(uuid.uuid4())
            config["job_id"] = job_id
            prepared.append((job_id, batch_index, request, config))

        if not prepared:
            # Every requested target language was a duplicate -- skip the
            # GCS upload and all BigQuery job creation entirely.
            responses = [duplicate_responses[i] for i in range(len(requests))]
            logger.info(
                "Submitted translation batch %s: all %d job(s) were duplicates",
                batch_id,
                len(responses),
            )
            return MultiTranslateResponse(batch_id=batch_id, jobs=responses)

        input_gs_uri = await self.storage.upload_input_pdf(
            file_content=content,
            filename=metadata["filename"],
            job_id=f"batches/{batch_id}",
        )

        now = datetime.now(UTC)
        # Same document for every target language in a batch, so the routing
        # tier is identical across them -- compute it once.
        effective_priority = self._resolve_effective_priority(requests[0])
        batch_doc_format = str(
            getattr(requests[0].document, "format", "") or ""
        ).lower()
        job_records: list[dict[str, Any]] = []
        for job_id, batch_index, request, config in prepared:
            job_data = {
                "job_id": job_id,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "status": job_status.QUEUED,
                "source_document": {
                    "gcs_uri": input_gs_uri,
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
                    "enable_dlp": request.processing_options.enable_dlp,
                },
                "cost_attribution": request.cost_attribution.model_dump(),
                # Record the priority actually used, not what the client asked
                # for, so BigQuery reflects the real routing decision.
                "processing_options": {
                    **request.processing_options.model_dump(),
                    "priority": effective_priority,
                },
                "error_message": None,
                "result": None,
                "config": config,
                "source_hash": source_hash,
                "submitted_at": now,
                "completed_at": None,
            }
            await self.bigquery.upsert_translation_job(job_data)
            job_records.append(job_data)

        parent_ctx = otel_context.get_current()
        for job_data in job_records:
            status = job_data["status"]
            try:
                await self._schedule_background_pipeline(
                    job_data["job_id"],
                    job_data,
                    parent_ctx,
                    priority=effective_priority,
                    doc_format=batch_doc_format,
                )
            except Exception:
                status = job_status.FAILED
            duplicate_responses[job_data["batch_index"]] = MultiTranslateJobResponse(
                job_id=job_data["job_id"],
                target_language=job_data["translation_config"]["target_language"],
                status=status,
                status_url=f"/api/v1/translate/{job_data['job_id']}",
            )

        # Reassemble in original request order -- duplicates detected above
        # and newly created jobs were populated into the same dict keyed by
        # batch_index, so this is the single merge point for both.
        responses = [duplicate_responses[i] for i in range(len(requests))]

        logger.info(
            "Submitted translation batch %s with %d job(s) (%d new, %d duplicate)",
            batch_id,
            len(responses),
            len(job_records),
            len(responses) - len(job_records),
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

            metadata = _validate_and_describe(content, request.document)

            # source_hash is retained for the in-process shared_document_prep
            # cache only -- see the comment in submit_translations() above for
            # why the old whole-document BigQuery result cache was removed.
            source_hash = hashlib.sha256(content).hexdigest()
            effective_priority = self._resolve_effective_priority(request)

            config = self._normalize_config(request)
            config["job_id"] = job_id

            # implementation_plan.md D.5 (EC-15): a double-click on submit
            # previously created two independent jobs -- return the existing
            # one instead of re-uploading to GCS / creating a second
            # BigQuery row for an identical (user, document, target,
            # domain) submission within the idempotency window.
            duplicate = await self._find_duplicate_job(
                request, source_hash, config["lang_out"], config["domain"]
            )
            if duplicate is not None:
                logger.info(
                    "Duplicate submission detected for user %s; returning "
                    "existing job %s instead of creating a new one",
                    request.cost_attribution.user_id,
                    duplicate["job_id"],
                )
                return TranslateResponse(
                    job_id=duplicate["job_id"],
                    status=str(duplicate.get("status", job_status.QUEUED)),
                    status_url=f"/api/v1/translate/{duplicate['job_id']}",
                    is_duplicate=True,
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
                    "enable_dlp": request.processing_options.enable_dlp,
                },
                "cost_attribution": request.cost_attribution.model_dump(),
                "processing_options": {
                    "enable_dlp": request.processing_options.enable_dlp,
                    "enable_chunking": request.processing_options.enable_chunking,
                    # Effective (server-decided) priority, not the client's.
                    "priority": effective_priority,
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
            await self._schedule_background_pipeline(
                job_id,
                job_data,
                parent_ctx,
                priority=effective_priority,
                doc_format=str(getattr(request.document, "format", "") or "").lower(),
            )

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

        # Source language is mandatory: auto-detection was removed along with
        # the "auto" sentinel this used to fall back to. The worker compares
        # this canonical code against the language it detects in the document,
        # so an absent value would leave nothing to compare against.
        lang_in = request.translation_config.source_language
        if not lang_in or not str(lang_in).strip():
            raise ValidationError(
                "Source language is required",
                field="translation_config.source_language",
            )
        try:
            lang_in = normalize_language(lang_in)
        except ValueError as e:
            raise ValidationError(
                str(e), field="translation_config.source_language"
            ) from e
        return {
            "lang_in": lang_in,
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

    @staticmethod
    def _resolve_effective_priority(request: TranslateRequest) -> str:
        """Decide the queue tier for a job. Server-side only.

        The client's `processing_options.priority` is deliberately IGNORED --
        users must not be able to promote their own work. Promotion is driven
        purely by document format via `HIGH_PRIORITY_FORMATS`, so short
        text jobs are not stuck behind multi-minute PDF jobs.

        Supports both "txt" and ".txt" format representations and checks
        both `document.format` and `document.filename`.
        """
        if not settings.HIGH_PRIORITY_ROUTING_ENABLED:
            return PRIORITY_STANDARD
        doc_format = (
            str(getattr(request.document, "format", "") or "").lower().lstrip(".")
        )
        filename = getattr(request.document, "filename", "") or ""
        filename_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        high_formats = {
            str(f).lower().lstrip(".") for f in settings.HIGH_PRIORITY_FORMATS
        }
        is_high = (doc_format in high_formats) or (filename_ext in high_formats)
        return PRIORITY_HIGH if is_high else PRIORITY_STANDARD

    async def _find_duplicate_job(
        self,
        request: TranslateRequest,
        source_hash: str,
        target_language: str,
        domain: str,
    ) -> dict[str, Any] | None:
        """implementation_plan.md D.5 (EC-15): idempotency-key lookup.

        Identity is (user_id, source document hash, normalized target
        language, normalized domain) within
        `settings.DUPLICATE_SUBMISSION_WINDOW_SECONDS`. A BigQuery lookup
        failure must never block a legitimate new submission -- it is
        logged and treated as "no duplicate found" rather than propagated,
        since the cost of occasionally missing a dedupe opportunity is far
        lower than the cost of failing a translation request outright.
        """
        window = int(settings.DUPLICATE_SUBMISSION_WINDOW_SECONDS)
        if window <= 0:
            return None
        try:
            return await self.bigquery.find_recent_duplicate_job(
                user_id=request.cost_attribution.user_id,
                source_hash=source_hash,
                target_language=target_language,
                domain=domain,
                window_seconds=window,
            )
        except Exception:
            logger.warning(
                "Duplicate-submission lookup failed; proceeding as a new submission",
                exc_info=True,
            )
            return None

    async def _schedule_background_pipeline(
        self,
        job_id: str,
        job_data: dict[str, Any],
        parent_ctx=None,
        priority: str | None = None,
        doc_format: str | None = None,
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
                priority=priority,
                doc_format=doc_format,
            )
        except Exception as e:
            logger.error("Cloud Tasks enqueue failed for job %s: %s", job_id, e)
            await self._mark_enqueue_failed(job_id, e)
            raise RuntimeError(f"Failed to enqueue translation job: {e}") from e

        logger.info("Enqueued Cloud Tasks translation for job %s", job_id)
