"""
Unit tests for worker/utils/exceptions.py.

Covers all worker-specific exception classes and their attributes.
"""

from worker.utils.exceptions import FileProcessingError
from worker.utils.exceptions import JobAlreadyCompletedError
from worker.utils.exceptions import JobNotFoundError
from worker.utils.exceptions import StorageError
from worker.utils.exceptions import TranslationError
from worker.utils.exceptions import ValidationError
from worker.utils.exceptions import WorkerError

# ---------------------------------------------------------------------------
# WorkerError (base)
# ---------------------------------------------------------------------------


class TestWorkerError:
    def test_message_stored(self):
        err = WorkerError("base error")
        assert err.message == "base error"

    def test_default_details_empty_dict(self):
        err = WorkerError("msg")
        assert err.details == {}

    def test_custom_details(self):
        err = WorkerError("msg", details={"key": "val"})
        assert err.details["key"] == "val"

    def test_is_exception(self):
        err = WorkerError("msg")
        assert isinstance(err, Exception)

    def test_str_is_message(self):
        err = WorkerError("something went wrong")
        assert str(err) == "something went wrong"


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_message_without_field(self):
        err = ValidationError("bad input")
        assert err.message == "bad input"
        assert err.details == {}

    def test_field_in_details(self):
        err = ValidationError("invalid value", field="target_language")
        assert err.details["field"] == "target_language"

    def test_is_worker_error(self):
        assert isinstance(ValidationError("x"), WorkerError)


# ---------------------------------------------------------------------------
# FileProcessingError
# ---------------------------------------------------------------------------


class TestFileProcessingError:
    def test_message_without_file_id(self):
        err = FileProcessingError("processing failed")
        assert err.message == "processing failed"
        assert err.details == {}

    def test_file_id_in_details(self):
        err = FileProcessingError("failed", file_id="job-abc")
        assert err.details["file_id"] == "job-abc"

    def test_is_worker_error(self):
        assert isinstance(FileProcessingError("x"), WorkerError)


# ---------------------------------------------------------------------------
# JobNotFoundError
# ---------------------------------------------------------------------------


class TestJobNotFoundError:
    def test_message_contains_job_id(self):
        err = JobNotFoundError("job-xyz")
        assert "job-xyz" in err.message

    def test_job_id_in_details(self):
        err = JobNotFoundError("job-xyz")
        assert err.details["job_id"] == "job-xyz"

    def test_is_worker_error(self):
        assert isinstance(JobNotFoundError("x"), WorkerError)


# ---------------------------------------------------------------------------
# JobAlreadyCompletedError
# ---------------------------------------------------------------------------


class TestJobAlreadyCompletedError:
    def test_message_contains_job_id(self):
        err = JobAlreadyCompletedError("done-job")
        assert "done-job" in err.message

    def test_job_id_in_details(self):
        err = JobAlreadyCompletedError("done-job")
        assert err.details["job_id"] == "done-job"

    def test_is_worker_error(self):
        assert isinstance(JobAlreadyCompletedError("x"), WorkerError)


# ---------------------------------------------------------------------------
# TranslationError
# ---------------------------------------------------------------------------


class TestTranslationError:
    def test_minimal(self):
        err = TranslationError("translation failed")
        assert err.message == "translation failed"
        assert err.details == {}

    def test_with_job_id(self):
        err = TranslationError("failed", job_id="j1")
        assert err.details["job_id"] == "j1"

    def test_with_stage(self):
        err = TranslationError("failed", stage="chunking")
        assert err.details["stage"] == "chunking"

    def test_none_values_excluded(self):
        err = TranslationError("failed", job_id="j1", stage=None)
        assert "stage" not in err.details

    def test_is_worker_error(self):
        assert isinstance(TranslationError("x"), WorkerError)


# ---------------------------------------------------------------------------
# StorageError
# ---------------------------------------------------------------------------


class TestStorageError:
    def test_minimal(self):
        err = StorageError("storage failed")
        assert err.message == "storage failed"
        assert err.details == {}

    def test_with_operation(self):
        err = StorageError("failed", operation="upload")
        assert err.details["operation"] == "upload"

    def test_with_path(self):
        err = StorageError("failed", path="gs://bucket/file.pdf")
        assert err.details["path"] == "gs://bucket/file.pdf"

    def test_none_values_excluded(self):
        err = StorageError("failed", operation="download", path=None)
        assert "path" not in err.details

    def test_is_worker_error(self):
        assert isinstance(StorageError("x"), WorkerError)
