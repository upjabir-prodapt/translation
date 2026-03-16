"""Global exception handling middleware."""

from fastapi import Request
from fastapi import status
from fastapi.responses import JSONResponse

from api.exceptions import BabelDocError
from api.exceptions import FileProcessingError
from api.exceptions import JobAlreadyCompletedError
from api.exceptions import JobNotFoundError
from api.exceptions import StorageError
from api.exceptions import ValidationError
from config.logging import logger


async def exception_handler_middleware(request: Request, call_next):
    """Global exception handling middleware."""
    try:
        return await call_next(request)
    except Exception as exc:
        return handle_exception(exc)


def handle_exception(exc: Exception) -> JSONResponse:
    """Handle exceptions and return appropriate JSON responses."""

    # Handle custom BabelDoc exceptions
    if isinstance(exc, ValidationError):
        logger.warning(f"Validation error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"message": exc.message, "code": "VALIDATION_ERROR"}},
        )

    if isinstance(exc, FileProcessingError):
        logger.error(f"File processing error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "FILE_ERROR"}},
        )

    if isinstance(exc, JobNotFoundError):
        logger.warning(f"Job not found: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"message": exc.message, "code": "JOB_NOT_FOUND"}},
        )

    if isinstance(exc, JobAlreadyCompletedError):
        logger.warning(f"Job already completed: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {"message": exc.message, "code": "JOB_ALREADY_COMPLETED"}
            },
        )

    if isinstance(exc, StorageError):
        logger.error(f"Storage error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "STORAGE_ERROR"}},
        )

    if isinstance(exc, BabelDocError):
        logger.error(f"BabelDoc error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "BABELDOC_ERROR"}},
        )

    # Handle unknown exceptions
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {"message": "Internal server error", "code": "INTERNAL_ERROR"}
        },
    )
