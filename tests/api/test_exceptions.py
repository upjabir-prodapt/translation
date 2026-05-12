from src.api.exceptions import BabelDocError
from src.api.exceptions import JobAlreadyCompletedError
from src.api.exceptions import JobNotFoundError
from src.api.exceptions import StorageError
from src.api.exceptions import ValidationError
from src.api.exceptions import internal_error
from src.api.exceptions import not_found_error
from src.api.exceptions import validation_error


class TestExceptions:
    def test_babel_doc_error(self):
        exc = BabelDocError("msg", details={"a": 1})
        assert exc.message == "msg"
        assert exc.details == {"a": 1}

    def test_validation_error(self):
        exc = ValidationError("invalid", field="f1")
        assert exc.details == {"field": "f1"}

    def test_job_not_found(self):
        exc = JobNotFoundError("job1")
        assert exc.details == {"job_id": "job1"}

    def test_job_already_completed(self):
        exc = JobAlreadyCompletedError("job1")
        assert exc.details == {"job_id": "job1"}

    def test_storage_error(self):
        exc = StorageError("fail", operation="upload")
        assert exc.details["operation"] == "upload"

    def test_http_exception_helpers(self):
        exc = validation_error("bad", field="f")
        assert exc.status_code == 400
        assert exc.detail["error"]["code"] == "VALIDATION_ERROR"

        exc = not_found_error("User", "1")
        assert exc.status_code == 404

        exc = internal_error("boom")
        assert exc.status_code == 500
