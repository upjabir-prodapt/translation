"""Translation service for handling PDF translation requests."""

import asyncio
import json
import logging
import uuid
from datetime import UTC
from datetime import datetime
from typing import Any

from fastapi import UploadFile
from google.cloud import tasks_v2

from api.repository.api_storage_repository import APIStorageRepository
from api.schemas.requests import TranslateRequest
from api.schemas.responses import TranslateResponse
from api.utils.pdf_validator import PDFValidator
from config.constants import settings
from repository.firestore_repository import FirestoreRepository

logger = logging.getLogger(__name__)


# Domain configurations
DOMAIN_CONFIGS = {
    "legal": {
        "auto_extract_glossary": True,
        "glossary_domains": ["legal"],
        "custom_system_prompt": "Translate legal documents with precision and maintain legal terminology.",
    },
    "medical": {
        "auto_extract_glossary": True,
        "glossary_domains": ["medical"],
        "custom_system_prompt": "Translate medical documents carefully, preserving medical terminology.",
    },
    "technical": {
        "auto_extract_glossary": True,
        "glossary_domains": ["technical"],
        "custom_system_prompt": "Translate technical documentation, maintaining technical accuracy.",
    },
    "general": {"auto_extract_glossary": False},
}


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

    async def submit_translation(
        self,
        file: UploadFile,
        domain: str,
        lang_in: str,
        lang_out: str,
        user: str,
        department: str,
    ) -> TranslateResponse:
        """Submit a PDF for translation."""
        # Generate job ID
        job_id = str(uuid.uuid4())

        try:
            # Validate PDF
            content, metadata = await PDFValidator.validate_pdf_file(file)

            # Create request object for normalization
            request = TranslateRequest(
                domain=domain,
                lang_in=lang_in,
                lang_out=lang_out,
                user=user,
                department=department,
            )

            # Normalize configuration
            config = self._normalize_config(request)

            # Upload to GCS
            input_gs_uri = await self.storage.upload_input_pdf(
                file_content=content, filename=metadata["filename"], job_id=job_id
            )

            # Create job document in Firestore
            job_data = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "input_gs_uri": input_gs_uri,
                "original_filename": metadata["filename"],
                "file_size_bytes": metadata["size_bytes"],
                "domain": domain,
                "lang_in": lang_in,
                "lang_out": lang_out,
                "user": user,
                "department": department,
                "config": config,
                "checksum": metadata["checksum"],
            }

            await self.firestore.create_job(job_id, job_data)

            # Create Cloud Task
            await self._create_translation_task(job_id, config)

            logger.info(f"Submitted translation job {job_id}")

            return TranslateResponse(
                job_id=job_id, status="queued", created_at=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(f"Failed to submit translation: {e}")
            # Cleanup on failure
            try:
                await self.storage.delete_job_files(job_id)
                await self.firestore.delete_job(job_id)
            except Exception:
                pass
            raise

    def _normalize_config(self, request: TranslateRequest) -> dict[str, Any]:
        """Normalize translation configuration."""
        # Start with domain configuration
        config = {}

        # Add language settings
        config.update(
            {
                "lang_in": request.lang_in,
                "lang_out": request.lang_out,
                "domain": request.domain,
                "user": request.user,
                "department": request.department,
            }
        )

        return config

    async def _create_translation_task(
        self, job_id: str, config: dict[str, Any]
    ) -> None:
        """Create a Cloud Task for translation."""
        # Construct the fully qualified queue name
        parent = self.tasks_client.queue_path(
            settings.GOOGLE_CLOUD_PROJECT_ID,
            settings.CLOUD_TASKS_LOCATION,
            settings.CLOUD_TASKS_QUEUE,
        )

        # Prepare the task payload
        payload = {"job_id": job_id, "config": config}

        # Create the task
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{settings.WORKER_URL}/process",
                "body": json.dumps(payload).encode(),
                "headers": {"Content-Type": "application/json"},
            },
            "dispatch_deadline": {"seconds": settings.CLOUD_TASKS_DEADLINE_SECONDS},
        }

        # Send the task
        response = await asyncio.to_thread(
            self.tasks_client.create_task, request={"parent": parent, "task": task}
        )

        logger.info(f"Created Cloud Task {response.name} for job {job_id}")
