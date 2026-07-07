"""Worker-specific dependencies for FastAPI."""

from typing import Any

from fastapi import HTTPException
from fastapi import Request

from config.logging_config import logger


async def validate_task_payload(request: Request) -> dict[str, Any]:
    """
    Validate and extract task payload from request.

    Args:
        request: FastAPI request object

    Returns:
        Validated task data

    Raises:
        HTTPException: If payload is invalid
    """
    try:
        task_data = await request.json()

        if not task_data:
            raise HTTPException(status_code=400, detail="Empty task payload")

        # Basic validation
        if "job_id" not in task_data:
            raise HTTPException(
                status_code=400, detail="Missing job_id in task payload"
            )

        if "config" not in task_data:
            raise HTTPException(
                status_code=400, detail="Missing config in task payload"
            )

        return task_data

    except ValueError as e:
        logger.error(f"Invalid JSON in task payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from e
