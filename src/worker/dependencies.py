"""Worker FastAPI dependency providers."""

from src.repository import get_bigquery_repository
from src.repository.api_storage_repository import APIStorageRepository
from src.repository.api_storage_repository import get_api_storage_repository
from src.repository.bigquery_repository import BigQueryRepository
from src.worker.handlers.translate_task_handler import TranslateTaskHandler
from src.worker.services.pipeline_orchestrator import PipelineOrchestrator


def get_storage() -> APIStorageRepository:
    return get_api_storage_repository()


def get_bigquery() -> BigQueryRepository:
    return get_bigquery_repository()


def get_pipeline_orchestrator() -> PipelineOrchestrator:
    return PipelineOrchestrator(
        bigquery=get_bigquery(),
        storage=get_storage(),
    )


def get_translate_task_handler() -> TranslateTaskHandler:
    return TranslateTaskHandler(
        bigquery=get_bigquery(),
        storage=get_storage(),
        orchestrator=get_pipeline_orchestrator(),
    )
