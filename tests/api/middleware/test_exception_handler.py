from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from src.api.exceptions import ConfigurationError
from src.api.exceptions import JobNotFoundError
from src.api.exceptions import ValidationError
from src.api.middleware.exception_handler import handle_exception


class TestExceptionHandler:
    def test_handle_http_exception(self):
        exc = HTTPException(status_code=403, detail="Forbidden")
        resp = handle_exception(exc)
        assert resp.status_code == 403
        assert resp.body == b'{"error":{"message":"Forbidden","code":"HTTP_ERROR"}}'

    def test_handle_request_validation_error(self):
        exc = RequestValidationError(
            errors=[{"loc": ("body", "f"), "msg": "err", "type": "t"}]
        )
        resp = handle_exception(exc)
        assert resp.status_code == 422

    def test_handle_configuration_error(self):
        exc = ConfigurationError("bad config")
        resp = handle_exception(exc)
        assert resp.status_code == 500
        assert b"CONFIGURATION_ERROR" in resp.body

    def test_handle_validation_error(self):
        exc = ValidationError("invalid")
        resp = handle_exception(exc)
        assert resp.status_code == 422

    def test_handle_job_not_found(self):
        exc = JobNotFoundError("missing")
        resp = handle_exception(exc)
        assert resp.status_code == 404

    def test_handle_unknown_exception(self):
        exc = Exception("kaboom")
        resp = handle_exception(exc)
        assert resp.status_code == 500
        assert b"INTERNAL_ERROR" in resp.body
