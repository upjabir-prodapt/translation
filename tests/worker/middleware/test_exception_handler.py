"""
Unit tests for worker/middleware/exception_handler.py — handle_exception().

The Request object is constructed from mock scope so no network I/O occurs.
"""

from unittest.mock import MagicMock

from worker.middleware.exception_handler import handle_exception
from worker.utils.exceptions import FileProcessingError
from worker.utils.exceptions import JobAlreadyCompletedError
from worker.utils.exceptions import JobNotFoundError
from worker.utils.exceptions import StorageError
from worker.utils.exceptions import TranslationError
from worker.utils.exceptions import ValidationError
from worker.utils.exceptions import WorkerError

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_request(task_name: str = "task-001") -> MagicMock:
    """Build a minimal mock FastAPI Request with a Cloud Tasks header."""
    req = MagicMock()
    req.headers = {"X-CloudTasks-TaskName": task_name}
    return req


# ---------------------------------------------------------------------------
# Custom worker exceptions
# ---------------------------------------------------------------------------


class TestHandleWorkerExceptions:
    def test_validation_error_returns_400(self):
        resp = handle_exception(ValidationError("bad input"), _make_request())
        assert resp.status_code == 400

    def test_validation_error_code(self):
        resp = handle_exception(ValidationError("bad input"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_file_processing_error_returns_500(self):
        resp = handle_exception(FileProcessingError("file broke"), _make_request())
        assert resp.status_code == 500

    def test_file_processing_error_code(self):
        resp = handle_exception(FileProcessingError("file broke"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "FILE_ERROR"

    def test_job_not_found_returns_404(self):
        resp = handle_exception(JobNotFoundError("j1"), _make_request())
        assert resp.status_code == 404

    def test_job_not_found_code(self):
        resp = handle_exception(JobNotFoundError("j1"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "JOB_NOT_FOUND"

    def test_job_already_completed_returns_409(self):
        resp = handle_exception(JobAlreadyCompletedError("done-job"), _make_request())
        assert resp.status_code == 409

    def test_job_already_completed_code(self):
        resp = handle_exception(JobAlreadyCompletedError("done-job"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "JOB_ALREADY_COMPLETED"

    def test_translation_error_returns_500(self):
        resp = handle_exception(TranslationError("translate failed"), _make_request())
        assert resp.status_code == 500

    def test_translation_error_code(self):
        resp = handle_exception(TranslationError("translate failed"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "TRANSLATION_ERROR"

    def test_storage_error_returns_500(self):
        resp = handle_exception(StorageError("storage down"), _make_request())
        assert resp.status_code == 500

    def test_storage_error_code(self):
        resp = handle_exception(StorageError("storage down"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "STORAGE_ERROR"

    def test_worker_error_base_returns_500(self):
        resp = handle_exception(WorkerError("generic worker error"), _make_request())
        assert resp.status_code == 500

    def test_worker_error_base_code(self):
        resp = handle_exception(WorkerError("generic worker error"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "WORKER_ERROR"


# ---------------------------------------------------------------------------
# Built-in exceptions
# ---------------------------------------------------------------------------


class TestHandleBuiltinExceptions:
    def test_value_error_returns_400(self):
        resp = handle_exception(ValueError("bad value"), _make_request())
        assert resp.status_code == 400

    def test_value_error_code(self):
        resp = handle_exception(ValueError("bad value"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "VALUE_ERROR"

    def test_timeout_error_returns_504(self):
        resp = handle_exception(TimeoutError("timed out"), _make_request())
        assert resp.status_code == 504

    def test_timeout_error_code(self):
        resp = handle_exception(TimeoutError("timed out"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "TIMEOUT_ERROR"

    def test_runtime_error_returns_500(self):
        resp = handle_exception(RuntimeError("runtime failure"), _make_request())
        assert resp.status_code == 500

    def test_runtime_error_code(self):
        resp = handle_exception(RuntimeError("runtime failure"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "PROCESSING_ERROR"

    def test_unknown_exception_returns_500(self):
        resp = handle_exception(Exception("unexpected"), _make_request())
        assert resp.status_code == 500

    def test_unknown_exception_code(self):
        resp = handle_exception(Exception("unexpected"), _make_request())
        import json

        body = json.loads(resp.body)
        assert body["error"]["code"] == "INTERNAL_ERROR"

    def test_response_contains_task_id_for_unknown(self):
        resp = handle_exception(Exception("oops"), _make_request(task_name="my-task"))
        import json

        body = json.loads(resp.body)
        assert body["error"]["task_id"] == "my-task"


# ---------------------------------------------------------------------------
# exception_handler_middleware dispatch
# ---------------------------------------------------------------------------


class TestExceptionHandlerMiddleware:
    async def test_passes_through_successful_response(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from worker.middleware.exception_handler import exception_handler_middleware

        app = FastAPI()
        app.middleware("http")(exception_handler_middleware)

        @app.get("/ok")
        def ok():
            return {"status": "ok"}

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/ok")
        assert resp.status_code == 200

    async def test_catches_unhandled_exception(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from worker.middleware.exception_handler import exception_handler_middleware

        app = FastAPI()
        app.middleware("http")(exception_handler_middleware)

        @app.get("/fail")
        def fail():
            raise RuntimeError("boom")

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/fail")
        assert resp.status_code == 500
