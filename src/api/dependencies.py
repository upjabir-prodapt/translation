"""FastAPI service dependencies."""

from typing import Annotated

from fastapi import Depends

from api.repository.api_storage_repository import get_api_storage_repository
from api.services.file_lifecycle_service import FileLifecycleService
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


def get_file_lifecycle_service() -> FileLifecycleService:
    """Get file lifecycle service instance."""
    return FileLifecycleService(
        firestore=get_firestore_repository(), storage=get_api_storage_repository()
    )


def get_job_service() -> JobService:
    """Get job service instance."""
    return JobService(
        firestore=get_firestore_repository(),
        storage=get_api_storage_repository(),
        lifecycle=get_file_lifecycle_service(),
    )


# Type aliases for cleaner code
TranslationServiceDep = Annotated[TranslationService, Depends(get_translation_service)]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]
