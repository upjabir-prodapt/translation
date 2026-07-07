"""Global exception handling middleware for worker."""

from fastapi import Request
from fastapi import status
from fastapi.responses import JSONResponse

from config.logging_config import logger
from worker.utils.exceptions import FileProcessingError
from worker.utils.exceptions import JobAlreadyCompletedError
from worker.utils.exceptions import JobNotFoundError
from worker.utils.exceptions import StorageError
from worker.utils.exceptions import TranslationError
from worker.utils.exceptions import ValidationError
from worker.utils.exceptions import WorkerError


async def exception_handler_middleware(request: Request, call_next):
    """Global exception handling middleware for worker tasks."""
    try:
        return await call_next(request)
    except Exception as exc:
        return handle_exception(exc, request)


def handle_exception(exc: Exception, request: Request) -> JSONResponse:
    """Handle exceptions and return appropriate JSON responses."""

    # Log the exception with request context
    request_id = request.headers.get("X-CloudTasks-TaskName", "unknown")

    # Handle custom BabelDoc exceptions
    if isinstance(exc, ValidationError):
        logger.warning(f"Validation error in worker task {request_id}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": exc.message,
                    "code": "VALIDATION_ERROR",
                    "details": exc.details,
                }
            },
        )

    if isinstance(exc, FileProcessingError):
        logger.error(
            f"File processing error in worker task {request_id}: {exc.message}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": exc.message,
                    "code": "FILE_ERROR",
                    "details": exc.details,
                }
            },
        )

    if isinstance(exc, JobNotFoundError):
        logger.warning(f"Job not found in worker task {request_id}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "message": exc.message,
                    "code": "JOB_NOT_FOUND",
                    "details": exc.details,
                }
            },
        )

    if isinstance(exc, JobAlreadyCompletedError):
        logger.warning(
            f"Job already completed in worker task {request_id}: {exc.message}"
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {
                    "message": exc.message,
                    "code": "JOB_ALREADY_COMPLETED",
                    "details": exc.details,
                }
            },
        )

    if isinstance(exc, TranslationError):
        logger.error(f"Translation error in worker task {request_id}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": exc.message,
                    "code": "TRANSLATION_ERROR",
                    "details": exc.details,
                }
            },
        )

    if isinstance(exc, StorageError):
        logger.error(f"Storage error in worker task {request_id}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": exc.message,
                    "code": "STORAGE_ERROR",
                    "details": exc.details,
                }
            },
        )

    if isinstance(exc, WorkerError):
        logger.error(f"Worker error in task {request_id}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": exc.message,
                    "code": "WORKER_ERROR",
                    "details": exc.details,
                }
            },
        )

    # Handle built-in exceptions that might occur during processing
    if isinstance(exc, ValueError):
        logger.warning(f"Value error in worker task {request_id}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": "Invalid request data",
                    "code": "VALUE_ERROR",
                    "details": str(exc),
                }
            },
        )

    if isinstance(exc, TimeoutError):
        logger.error(f"Timeout error in worker task {request_id}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": {
                    "message": "Task processing timed out",
                    "code": "TIMEOUT_ERROR",
                    "details": str(exc),
                }
            },
        )

    if isinstance(exc, RuntimeError):
        logger.error(f"Runtime error in worker task {request_id}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": "Task processing failed",
                    "code": "PROCESSING_ERROR",
                    "details": str(exc),
                }
            },
        )

    # Handle unknown exceptions
    logger.exception(f"Unhandled exception in worker task {request_id}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal worker error",
                "code": "INTERNAL_ERROR",
                "task_id": request_id,
            }
        },
    )
