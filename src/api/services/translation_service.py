"""Translation service for handling document translation requests."""

import asyncio
import base64
import hashlib
import logging
import uuid
from datetime import UTC
from datetime import datetime
from typing import Any

from src.api.exceptions import ValidationError
from src.api.schemas.requests import TranslateRequest
from src.api.schemas.responses import TranslateResponse
from src.api.services.pipeline_orchestrator import PipelineOrchestrator
from src.api.utils.pdf_validator import PDFValidator
from src.config.constants import settings
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

    async def submit_translation(self, request: TranslateRequest) -> TranslateResponse:
        """Submit a document for translation."""
        job_id = str(uuid.uuid4())

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

            # Normalize config
            config = self._normalize_config(request)
            config["job_id"] = job_id

            # Upload to GCS
            input_gs_uri = await self.storage.upload_input_pdf(
                file_content=content,
                filename=metadata["filename"],
                job_id=job_id,
            )

            # Build output filename: original_name_<target_lang_code>.<ext>
            orig_name = metadata["filename"].rsplit(".", 1)
            name_stem = orig_name[0] if len(orig_name) == 2 else metadata["filename"]
            name_ext = orig_name[1] if len(orig_name) == 2 else "pdf"
            output_filename = f"{name_stem}_{config['lang_out']}.{name_ext}"

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
                "source_hash": hashlib.sha256(content).hexdigest(),
                "submitted_at": now,
                "completed_at": None,
            }

            await self.bigquery.upsert_translation_job(job_data)
            await self._schedule_background_pipeline(job_id, job_data)

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

    async def _schedule_background_pipeline(
        self,
        job_id: str,
        job_data: dict[str, Any],
    ) -> None:
        """Schedule API-local background translation pipeline."""
        if settings.API_USE_BACKGROUND_PIPELINE:
            asyncio.create_task(self.orchestrator.run(job_id=job_id, job_data=job_data))
            logger.info(f"Scheduled in-process translation pipeline for job {job_id}")
            return
        raise RuntimeError("Cloud Tasks mode is disabled in this implementation")
