"""FastAPI service dependencies."""

from typing import Annotated

from fastapi import Depends

from src.api.handlers.jobs_handler import JobsHandler
from src.api.handlers.reviews_handler import ReviewsHandler
from src.api.handlers.translation_handler import TranslationHandler
from src.api.services.job_service import JobService
from src.api.services.review_service import ReviewService
from src.api.services.translation_service import TranslationService
from src.repository import get_bigquery_repository
from src.repository.api_storage_repository import get_api_storage_repository


# Service dependencies
def get_translation_service() -> TranslationService:
    """Get translation service instance."""
    return TranslationService(
        storage=get_api_storage_repository(),
        bigquery=get_bigquery_repository(),
    )


def get_job_service() -> JobService:
    """Get job service instance."""
    return JobService(
        storage=get_api_storage_repository(),
        bigquery=get_bigquery_repository(),
    )


def get_translation_handler(
    translation_service: TranslationService = Depends(get_translation_service),  # noqa: B008
    job_service: JobService = Depends(get_job_service),  # noqa: B008
) -> TranslationHandler:
    """Get translation request handler instance."""
    return TranslationHandler(
        translation_service=translation_service,
        job_service=job_service,
    )


def get_jobs_handler(
    job_service: JobService = Depends(get_job_service),  # noqa: B008
) -> JobsHandler:
    """Get jobs request handler instance."""
    return JobsHandler(job_service=job_service)


def get_review_service() -> ReviewService:
    """Get review service instance."""
    return ReviewService(bigquery=get_bigquery_repository())


def get_reviews_handler(
    review_service: ReviewService = Depends(get_review_service),  # noqa: B008
) -> ReviewsHandler:
    """Get reviews request handler instance."""
    return ReviewsHandler(review_service=review_service)


# Type aliases for cleaner code
TranslationServiceDep = Annotated[TranslationService, Depends(get_translation_service)]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]
TranslationHandlerDep = Annotated[TranslationHandler, Depends(get_translation_handler)]
JobsHandlerDep = Annotated[JobsHandler, Depends(get_jobs_handler)]
