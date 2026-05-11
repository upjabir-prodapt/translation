"""
Unit tests for api/middleware/exception_handler.py — handle_exception() mapping.

Tests verify that each exception type produces the correct HTTP status code
and structured JSON error body.
"""

import pytest
from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError

from src.api.exceptions import (
    BabelDocError,
    ConfigurationError,
    FileProcessingError,
    JobAlreadyCompletedError,
    JobNotFoundError,
    StorageError,
    TranslationError,
    ValidationError,
)
from src.api.middleware.exception_handler import handle_exception


def _status(response) -> int:
    return response.status_code


def _code(response) -> str:
    return response.body.decode() if hasattr(response, "body") else ""


def _body(response) -> dict:
    import json
    return json.loads(response.body)


# ---------------------------------------------------------------------------
# HTTPException pass-through
# ---------------------------------------------------------------------------


class TestHTTPExceptionPassThrough:
    def test_404_passthrough(self):
        exc = HTTPException(status_code=404, detail="Not found")
        resp = handle_exception(exc)
        assert resp.status_code == 404

    def test_dict_detail_preserved(self):
        detail = {"error": {"message": "custom", "code": "CUSTOM"}}
        exc = HTTPException(status_code=422, detail=detail)
        resp = handle_exception(exc)
        body = _body(resp)
        assert body["error"]["message"] == "custom"

    def test_string_detail_wrapped(self):
        exc = HTTPException(status_code=403, detail="Forbidden")
        resp = handle_exception(exc)
        body = _body(resp)
        assert "Forbidden" in body["error"]["message"]


# ---------------------------------------------------------------------------
# RequestValidationError → 422
# ---------------------------------------------------------------------------


class TestRequestValidationError:
    def test_returns_422(self):
        # Build a RequestValidationError from a fake pydantic error list
        errors = [
            {"loc": ("body", "document", "content"), "msg": "field required", "type": "missing"}
        ]
        exc = RequestValidationError(errors)
        resp = handle_exception(exc)
        assert resp.status_code == 422

    def test_body_has_validation_error_code(self):
        errors = [{"loc": ("body", "field"), "msg": "bad value", "type": "value_error"}]
        exc = RequestValidationError(errors)
        resp = handle_exception(exc)
        body = _body(resp)
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_body_contains_field_details(self):
        errors = [{"loc": ("body", "lang_in"), "msg": "required", "type": "missing"}]
        exc = RequestValidationError(errors)
        resp = handle_exception(exc)
        body = _body(resp)
        details = body["error"]["details"]
        assert isinstance(details, list)
        assert any("lang_in" in d.get("field", "") for d in details)


# ---------------------------------------------------------------------------
# ConfigurationError → 500
# ---------------------------------------------------------------------------


class TestConfigurationError:
    def test_returns_500(self):
        resp = handle_exception(ConfigurationError("config missing"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_body_code(self):
        resp = handle_exception(ConfigurationError("missing key"))
        body = _body(resp)
        assert body["error"]["code"] == "CONFIGURATION_ERROR"


# ---------------------------------------------------------------------------
# ValidationError → 400
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_returns_400(self):
        resp = handle_exception(ValidationError("invalid input"))
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_body_code(self):
        resp = handle_exception(ValidationError("bad field"))
        body = _body(resp)
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_message_propagated(self):
        resp = handle_exception(ValidationError("field is required"))
        body = _body(resp)
        assert "field is required" in body["error"]["message"]


# ---------------------------------------------------------------------------
# FileProcessingError → 500
# ---------------------------------------------------------------------------


class TestFileProcessingError:
    def test_returns_500(self):
        resp = handle_exception(FileProcessingError("parse failure"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_body_code(self):
        resp = handle_exception(FileProcessingError("parse failure"))
        body = _body(resp)
        assert body["error"]["code"] == "FILE_ERROR"


# ---------------------------------------------------------------------------
# JobNotFoundError → 404
# ---------------------------------------------------------------------------


class TestJobNotFoundError:
    def test_returns_404(self):
        resp = handle_exception(JobNotFoundError("job-1"))
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    def test_body_code(self):
        resp = handle_exception(JobNotFoundError("job-1"))
        body = _body(resp)
        assert body["error"]["code"] == "JOB_NOT_FOUND"


# ---------------------------------------------------------------------------
# JobAlreadyCompletedError → 409
# ---------------------------------------------------------------------------


class TestJobAlreadyCompletedError:
    def test_returns_409(self):
        resp = handle_exception(JobAlreadyCompletedError("job-2"))
        assert resp.status_code == status.HTTP_409_CONFLICT

    def test_body_code(self):
        resp = handle_exception(JobAlreadyCompletedError("job-2"))
        body = _body(resp)
        assert body["error"]["code"] == "JOB_ALREADY_COMPLETED"


# ---------------------------------------------------------------------------
# StorageError → 500
# ---------------------------------------------------------------------------


class TestStorageError:
    def test_returns_500(self):
        resp = handle_exception(StorageError("upload failed"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_body_code(self):
        resp = handle_exception(StorageError("download failed"))
        body = _body(resp)
        assert body["error"]["code"] == "STORAGE_ERROR"


# ---------------------------------------------------------------------------
# TranslationError → 500
# ---------------------------------------------------------------------------


class TestTranslationError:
    def test_returns_500(self):
        resp = handle_exception(TranslationError("translation failed"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_body_code(self):
        resp = handle_exception(TranslationError("translation failed"))
        body = _body(resp)
        assert body["error"]["code"] == "TRANSLATION_ERROR"


# ---------------------------------------------------------------------------
# Generic BabelDocError → 500
# ---------------------------------------------------------------------------


class TestBabelDocErrorGeneric:
    def test_returns_500(self):
        resp = handle_exception(BabelDocError("generic error"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_body_code(self):
        resp = handle_exception(BabelDocError("generic error"))
        body = _body(resp)
        assert body["error"]["code"] == "BABELDOC_ERROR"


# ---------------------------------------------------------------------------
# ValueError → 400
# ---------------------------------------------------------------------------


class TestValueError:
    def test_returns_400(self):
        resp = handle_exception(ValueError("bad value"))
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_body_code(self):
        resp = handle_exception(ValueError("unsupported language"))
        body = _body(resp)
        assert body["error"]["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# RuntimeError → 500
# ---------------------------------------------------------------------------


class TestRuntimeError:
    def test_returns_500(self):
        resp = handle_exception(RuntimeError("runtime failure"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_body_code(self):
        resp = handle_exception(RuntimeError("config error"))
        body = _body(resp)
        assert body["error"]["code"] == "CONFIGURATION_ERROR"


# ---------------------------------------------------------------------------
# Unknown exceptions → 500 INTERNAL_ERROR
# ---------------------------------------------------------------------------


class TestUnknownException:
    def test_returns_500(self):
        resp = handle_exception(Exception("unexpected"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_body_code(self):
        resp = handle_exception(Exception("crash"))
        body = _body(resp)
        assert body["error"]["code"] == "INTERNAL_ERROR"

    def test_message_generic(self):
        resp = handle_exception(Exception("crash"))
        body = _body(resp)
        assert "Internal server error" in body["error"]["message"]
