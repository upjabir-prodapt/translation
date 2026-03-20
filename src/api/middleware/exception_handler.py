"""Global exception handling middleware."""

from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.exceptions import BabelDocError
from api.exceptions import ConfigurationError
from api.exceptions import FileProcessingError
from api.exceptions import JobAlreadyCompletedError
from api.exceptions import JobNotFoundError
from api.exceptions import StorageError
from api.exceptions import TranslationError
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

    # Pass through FastAPI/Starlette HTTP exceptions unchanged
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
            if isinstance(exc.detail, dict)
            else {"error": {"message": str(exc.detail), "code": "HTTP_ERROR"}},
        )

    # Pydantic request validation errors
    if isinstance(exc, RequestValidationError):
        errors = [
            {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        logger.warning(f"Request validation error: {errors}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "message": "Request validation failed",
                    "code": "VALIDATION_ERROR",
                    "details": errors,
                }
            },
        )

    # Configuration errors (missing files, bad settings) — must check before BabelDocError
    if isinstance(exc, ConfigurationError):
        logger.error(f"Configuration error: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "CONFIGURATION_ERROR"}},
        )

    # Input validation errors
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

    if isinstance(exc, TranslationError):
        logger.error(f"Translation error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "TRANSLATION_ERROR"}},
        )

    if isinstance(exc, BabelDocError):
        logger.error(f"BabelDoc error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": exc.message, "code": "BABELDOC_ERROR"}},
        )

    # ValueError from language/domain normalization surfaced outside service layer
    if isinstance(exc, ValueError):
        logger.warning(f"Value error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"message": str(exc), "code": "VALIDATION_ERROR"}},
        )

    # RuntimeError from missing config (e.g. language_mapper.json not found)
    if isinstance(exc, RuntimeError):
        logger.error(f"Runtime error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": str(exc), "code": "CONFIGURATION_ERROR"}},
        )

    # Unknown exceptions
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {"message": "Internal server error", "code": "INTERNAL_ERROR"}
        },
    )
