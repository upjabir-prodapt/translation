from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from src.api.exceptions import BabelDocError
from src.api.exceptions import ConfigurationError
from src.api.exceptions import FileProcessingError
from src.api.exceptions import JobAlreadyCompletedError
from src.api.exceptions import JobNotFoundError
from src.api.exceptions import StorageError
from src.api.exceptions import TranslationError
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

    def test_handle_file_processing_error(self):
        exc = FileProcessingError("processing failed", file_id="f1")
        resp = handle_exception(exc)
        assert resp.status_code == 500
        assert b"FILE_ERROR" in resp.body

    def test_handle_storage_error(self):
        exc = StorageError("storage failed", operation="upload")
        resp = handle_exception(exc)
        assert resp.status_code == 500
        assert b"STORAGE_ERROR" in resp.body

    def test_handle_translation_error(self):
        exc = TranslationError("translation failed", job_id="j1")
        resp = handle_exception(exc)
        assert resp.status_code == 500
        assert b"TRANSLATION_ERROR" in resp.body

    def test_handle_babel_doc_error(self):
        exc = BabelDocError("babel error")
        resp = handle_exception(exc)
        assert resp.status_code == 500
        assert b"BABELDOC_ERROR" in resp.body

    def test_handle_value_error(self):
        exc = ValueError("bad value")
        resp = handle_exception(exc)
        assert resp.status_code == 422
        assert b"VALIDATION_ERROR" in resp.body

    def test_handle_runtime_error(self):
        exc = RuntimeError("runtime failure")
        resp = handle_exception(exc)
        assert resp.status_code == 500
        assert b"CONFIGURATION_ERROR" in resp.body

    def test_handle_job_already_completed(self):
        exc = JobAlreadyCompletedError("job1")
        resp = handle_exception(exc)
        assert resp.status_code == 409
        assert b"JOB_ALREADY_COMPLETED" in resp.body

    def test_handle_http_exception_with_dict_detail(self):
        exc = HTTPException(status_code=400, detail={"error": {"message": "bad", "code": "BAD"}})
        resp = handle_exception(exc)
        assert resp.status_code == 400

    def test_handle_str_repr_fallback(self):
        class BadStr(Exception):
            def __str__(self):
                raise RuntimeError("str failed")
        exc = BadStr()
        resp = handle_exception(exc)
        assert resp.status_code == 500
