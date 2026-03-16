"""FastAPI service dependencies."""

from typing import Annotated

from fastapi import Depends

from api.repository.api_storage_repository import get_api_storage_repository
from api.services.job_service import JobService
from api.services.translation_service import TranslationService
from repository import get_firestore_repository
from repository import get_tasks_client


# Service dependencies
def get_translation_service() -> TranslationService:
    """Get translation service instance."""
    return TranslationService(
        firestore=get_firestore_repository(),
        storage=get_api_storage_repository(),
        tasks_client=get_tasks_client(),
    )


def get_job_service() -> JobService:
    """Get job service instance."""
    return JobService(
        firestore=get_firestore_repository(), storage=get_api_storage_repository()
    )


# Type aliases for cleaner code
TranslationServiceDep = Annotated[TranslationService, Depends(get_translation_service)]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]
