"""
Unit tests for api/exceptions.py — custom exception classes and HTTP helpers.
"""

import pytest
from fastapi import HTTPException, status

from api.exceptions import (
    BabelDocError,
    ConfigurationError,
    FileProcessingError,
    JobAlreadyCompletedError,
    JobNotFoundError,
    StorageError,
    TranslationError,
    ValidationError,
    create_http_exception,
    internal_error,
    not_found_error,
    validation_error,
)


# ---------------------------------------------------------------------------
# BabelDocError (base)
# ---------------------------------------------------------------------------


class TestBabelDocError:
    def test_message_stored(self):
        exc = BabelDocError("something went wrong")
        assert exc.message == "something went wrong"

    def test_details_default_empty(self):
        exc = BabelDocError("msg")
        assert exc.details == {}

    def test_details_stored(self):
        exc = BabelDocError("msg", details={"key": "val"})
        assert exc.details == {"key": "val"}

    def test_str_representation(self):
        exc = BabelDocError("test error")
        assert str(exc) == "test error"

    def test_is_exception(self):
        exc = BabelDocError("msg")
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_message_and_field(self):
        exc = ValidationError("bad input", field="doc.content")
        assert exc.message == "bad input"
        assert exc.details == {"field": "doc.content"}

    def test_no_field(self):
        exc = ValidationError("invalid")
        assert exc.details == {}

    def test_inherits_babel_doc_error(self):
        exc = ValidationError("x")
        assert isinstance(exc, BabelDocError)


# ---------------------------------------------------------------------------
# FileProcessingError
# ---------------------------------------------------------------------------


class TestFileProcessingError:
    def test_with_file_id(self):
        exc = FileProcessingError("parse failure", file_id="abc-123")
        assert exc.details == {"file_id": "abc-123"}

    def test_without_file_id(self):
        exc = FileProcessingError("parse failure")
        assert exc.details == {}


# ---------------------------------------------------------------------------
# JobNotFoundError
# ---------------------------------------------------------------------------


class TestJobNotFoundError:
    def test_message_contains_job_id(self):
        exc = JobNotFoundError("job-xyz")
        assert "job-xyz" in exc.message

    def test_details_contain_job_id(self):
        exc = JobNotFoundError("job-xyz")
        assert exc.details == {"job_id": "job-xyz"}


# ---------------------------------------------------------------------------
# JobAlreadyCompletedError
# ---------------------------------------------------------------------------


class TestJobAlreadyCompletedError:
    def test_message_contains_job_id(self):
        exc = JobAlreadyCompletedError("job-abc")
        assert "job-abc" in exc.message

    def test_details_contain_job_id(self):
        exc = JobAlreadyCompletedError("job-abc")
        assert exc.details == {"job_id": "job-abc"}


# ---------------------------------------------------------------------------
# TranslationError
# ---------------------------------------------------------------------------


class TestTranslationError:
    def test_with_job_id_and_stage(self):
        exc = TranslationError("failed", job_id="j1", stage="parsing")
        assert exc.details == {"job_id": "j1", "stage": "parsing"}

    def test_omits_none_details(self):
        exc = TranslationError("failed", job_id="j1")
        assert "stage" not in exc.details

    def test_no_optional_args(self):
        exc = TranslationError("failed")
        assert exc.details == {}


# ---------------------------------------------------------------------------
# StorageError
# ---------------------------------------------------------------------------


class TestStorageError:
    def test_with_operation_and_path(self):
        exc = StorageError("upload failed", operation="upload", path="gs://b/p")
        assert exc.details == {"operation": "upload", "path": "gs://b/p"}

    def test_omits_none_details(self):
        exc = StorageError("failed", operation="download")
        assert "path" not in exc.details

    def test_no_optional_args(self):
        exc = StorageError("failed")
        assert exc.details == {}


# ---------------------------------------------------------------------------
# ConfigurationError
# ---------------------------------------------------------------------------


class TestConfigurationError:
    def test_with_config_key(self):
        exc = ConfigurationError("missing key", config_key="API_KEY")
        assert exc.details == {"config_key": "API_KEY"}

    def test_without_config_key(self):
        exc = ConfigurationError("missing config")
        assert exc.details == {}


# ---------------------------------------------------------------------------
# HTTP Exception helpers
# ---------------------------------------------------------------------------


class TestCreateHttpException:
    def test_returns_http_exception(self):
        exc = create_http_exception(400, "bad request")
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 400

    def test_detail_structure(self):
        exc = create_http_exception(404, "not found", error_code="NF", details={"id": "1"})
        assert exc.detail["error"]["message"] == "not found"
        assert exc.detail["error"]["code"] == "NF"
        assert exc.detail["error"]["details"] == {"id": "1"}

    def test_default_error_code(self):
        exc = create_http_exception(500, "error")
        assert exc.detail["error"]["code"] == "UNKNOWN_ERROR"


class TestNotFoundError:
    def test_returns_404(self):
        exc = not_found_error("Job", "job-1")
        assert exc.status_code == status.HTTP_404_NOT_FOUND

    def test_detail_message(self):
        exc = not_found_error("Document")
        assert "not found" in exc.detail["error"]["message"].lower()

    def test_detail_identifier(self):
        exc = not_found_error("Job", "job-42")
        assert exc.detail["error"]["details"]["identifier"] == "job-42"


class TestValidationErrorHelper:
    def test_returns_400(self):
        exc = validation_error("field required")
        assert exc.status_code == status.HTTP_400_BAD_REQUEST

    def test_detail_with_field(self):
        exc = validation_error("required", field="name")
        assert exc.detail["error"]["details"]["field"] == "name"

    def test_detail_without_field(self):
        exc = validation_error("required")
        assert exc.detail["error"]["details"] == {}


class TestInternalError:
    def test_returns_500(self):
        exc = internal_error("oops")
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_detail_message(self):
        exc = internal_error("unexpected failure")
        assert exc.detail["error"]["message"] == "unexpected failure"
