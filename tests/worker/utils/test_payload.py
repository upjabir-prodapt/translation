"""
Unit tests for worker/utils/payload.py — validate_task_payload().

Uses a mock FastAPI Request to avoid real HTTP connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from worker.utils.payload import validate_task_payload


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _mock_request(json_data=None, raise_value_error=False):
    """Build a mock Request whose .json() returns json_data."""
    req = MagicMock()
    if raise_value_error:
        req.json = AsyncMock(side_effect=ValueError("bad json"))
    else:
        req.json = AsyncMock(return_value=json_data)
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateTaskPayload:
    async def test_valid_payload_returned(self):
        payload = {"job_id": "j1", "config": {"lang_out": "es"}}
        result = await validate_task_payload(_mock_request(payload))
        assert result == payload

    async def test_empty_payload_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await validate_task_payload(_mock_request({}))
        assert exc_info.value.status_code == 400
        assert "Empty" in exc_info.value.detail

    async def test_none_payload_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await validate_task_payload(_mock_request(None))
        assert exc_info.value.status_code == 400

    async def test_missing_job_id_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await validate_task_payload(_mock_request({"config": {}}))
        assert exc_info.value.status_code == 400
        assert "job_id" in exc_info.value.detail

    async def test_missing_config_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await validate_task_payload(_mock_request({"job_id": "j1"}))
        assert exc_info.value.status_code == 400
        assert "config" in exc_info.value.detail

    async def test_invalid_json_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await validate_task_payload(_mock_request(raise_value_error=True))
        assert exc_info.value.status_code == 400
        assert "Invalid JSON" in exc_info.value.detail

    async def test_extra_fields_preserved(self):
        payload = {"job_id": "j1", "config": {}, "extra_field": "value"}
        result = await validate_task_payload(_mock_request(payload))
        assert result["extra_field"] == "value"
