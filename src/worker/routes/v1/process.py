"""Translation processing endpoint."""

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request

from config.logging_config import logger
from worker.handlers.translation_services import handle_translation_task
from worker.utils.payload import validate_task_payload

router = APIRouter()


@router.post("/process")
async def process(request: Request):
    """Handle translation task from Cloud Tasks."""
    try:
        # Validate and extract task payload
        task_data = await validate_task_payload(request)

        # Process the task
        result = await handle_translation_task(task_data)

        if result.get("success"):
            return ""  # Success response for Cloud Tasks
        else:
            logger.error(f"Task failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get("error"))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in process endpoint")
        raise HTTPException(status_code=500, detail="Internal server error") from e
