"""Translation service for handling document translation requests."""

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

from src.api.exceptions import ValidationError
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.responses import TranslateResponse
from src.api.services.pipeline_orchestrator import PipelineOrchestrator
from src.api.utils.pdf_validator import PDFValidator
from src.config.constants import settings
from src.config.tracing import tracer_pipeline
from src.config.translation_routing import normalize_domain
from src.config.translation_routing import normalize_language
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.bigquery_repository import BigQueryRepository

logger = logging.getLogger(__name__)


class TranslationService:
    """Service for handling translation requests."""

    def __init__(
        self,
        storage: APIStorageRepository | None = None,
        bigquery: BigQueryRepository | None = None,
        orchestrator: PipelineOrchestrator | None = None,
    ):
        self.storage = storage or APIStorageRepository()
        self.bigquery = bigquery or BigQueryRepository()
        self.orchestrator = orchestrator or PipelineOrchestrator(
            bigquery=self.bigquery,
            storage=self.storage,
        )
        self._background_tasks = set()

    async def submit_translation(self, request: TranslateRequest) -> TranslateResponse:
        """Submit a document for translation."""
        job_id = str(uuid.uuid4())

        with tracer_pipeline.start_as_current_span(
            "translation_service.submit",
            kind=SpanKind.INTERNAL,
            attributes={"translation.job_id": job_id},
        ):
            return await self._do_submit(request, job_id)

    async def _do_submit(
        self, request: TranslateRequest, job_id: str
    ) -> TranslateResponse:
        try:
            # Decode base64 content
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

            # Normalize config
            config = self._normalize_config(request)
            config["job_id"] = job_id

            # Return cached result immediately if an identical completed job exists
            cached_job = await self.bigquery.get_completed_job_by_hash(
                source_hash=source_hash,
                lang_out=config["lang_out"],
                domain=config["domain"],
            )
            if cached_job and cached_job.get("result"):
                return await self._serve_from_cache(
                    request, job_id, metadata, source_hash, config, cached_job
                )

            # Upload to GCS
            input_gs_uri = await self.storage.upload_input_pdf(
                file_content=content,
                filename=metadata["filename"],
                job_id=job_id,
            )

            # Keep output filename same as input filename.
            output_filename = metadata["filename"]

            now = datetime.now(UTC)
            job_data = {
                "job_id": job_id,
                "status": "queued",
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

            # Capture trace context before leaving HTTP scope — asyncio.create_task
            # does not propagate OTel context automatically.
            parent_ctx = otel_context.get_current()
            self._schedule_background_pipeline(job_id, job_data, parent_ctx)

            logger.info(f"Submitted translation job {job_id}")

            return TranslateResponse(
                job_id=job_id,
                status="queued",
                status_url=f"/api/v1/translate/{job_id}",
            )

        except Exception as e:
            logger.error(f"Failed to submit translation: {e}")
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
        """Write a pre-completed job record reusing the cached result — no pipeline needed."""
        now = datetime.now(UTC)
        job_data = {
            "job_id": job_id,
            "status": "completed",
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
            f"Cache hit for job {job_id}: reusing result from {cached_job.get('job_id')}"
        )
        return TranslateResponse(
            job_id=job_id,
            status="completed",
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

    def _schedule_background_pipeline(
        self,
        job_id: str,
        job_data: dict[str, Any],
        parent_ctx=None,
    ) -> None:
        """Schedule API-local background translation pipeline."""
        if settings.API_USE_BACKGROUND_PIPELINE:
            task = asyncio.create_task(
                self.orchestrator.run(
                    job_id=job_id, job_data=job_data, parent_ctx=parent_ctx
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            logger.info(f"Scheduled in-process translation pipeline for job {job_id}")
            return
        raise RuntimeError("Cloud Tasks mode is disabled in this implementation")
