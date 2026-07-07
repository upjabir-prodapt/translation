"""Translation service for handling document translation requests."""

import asyncio
import base64
import json
import logging
import uuid
from datetime import UTC
from datetime import datetime
from typing import Any

from google.cloud import tasks_v2

from api.exceptions import ValidationError
from api.repository.api_storage_repository import APIStorageRepository
from api.schemas.requests import TranslateRequest
from api.schemas.responses import TranslateResponse
from api.utils.pdf_validator import PDFValidator
from config.constants import settings
from config.translation_routing import normalize_domain
from config.translation_routing import normalize_language
from repository.firestore_repository import FirestoreRepository

logger = logging.getLogger(__name__)


class TranslationService:
    """Service for handling translation requests."""

    def __init__(
        self,
        firestore: FirestoreRepository | None = None,
        storage: APIStorageRepository | None = None,
        tasks_client: tasks_v2.CloudTasksClient | None = None,
    ):
        self.firestore = firestore or FirestoreRepository()
        self.storage = storage or APIStorageRepository()
        self.tasks_client = tasks_client or tasks_v2.CloudTasksClient()

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

            # Validate PDF bytes
            _, metadata = PDFValidator.validate_pdf_bytes(
                content, request.document.filename
            )

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

            # Create job document in Firestore
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
                    "source_language": request.translation_config.source_language,
                    "target_language": request.translation_config.target_language,
                    "domain": config["domain"],
                },
                "processing_options": {
                    "enable_dlp": request.processing_options.enable_dlp,
                    "enable_chunking": request.processing_options.enable_chunking,
                    "priority": request.processing_options.priority,
                },
                "processing": {
                    "model_used": None,
                    "model_version": None,
                    "chunks": None,
                    "chunking_applied": False,
                    "retry_count": 0,
                    "ab_variant": None,  # set by worker after model selection
                    "dlp_applied": request.processing_options.enable_dlp,
                },
                "result": None,
                "timestamps": {
                    "submitted_at": now,
                    "completed_at": None,
                },
                "config": config,
            }

            await self.firestore.create_job(job_id, job_data)

            # Create Cloud Task
            await self._create_translation_task(job_id, config)

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
                await self.firestore.delete_job(job_id)
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

        return {
            "lang_in": "auto",
            "lang_out": lang_out,
            "domain": domain,
        }

    async def _create_translation_task(
        self, job_id: str, config: dict[str, Any]
    ) -> None:
        """Create a Cloud Task for translation."""
        parent = self.tasks_client.queue_path(
            settings.GOOGLE_CLOUD_PROJECT_ID,
            settings.CLOUD_TASKS_LOCATION,
            settings.CLOUD_TASKS_QUEUE,
        )

        payload = {"job_id": job_id, "config": config}

        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{settings.WORKER_URL}/process",
                "body": json.dumps(payload).encode(),
                "headers": {"Content-Type": "application/json"},
            },
            "dispatch_deadline": {"seconds": settings.CLOUD_TASKS_DEADLINE_SECONDS},
        }

        response = await asyncio.to_thread(
            self.tasks_client.create_task, request={"parent": parent, "task": task}
        )

        logger.info(f"Created Cloud Task {response.name} for job {job_id}")
