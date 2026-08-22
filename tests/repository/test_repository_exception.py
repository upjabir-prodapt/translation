"""Tests for repository exception classes."""

from src.repository.repository_exception import BigQueryError
from src.repository.repository_exception import RepositoryError
from src.repository.repository_exception import StorageError


class TestStorageError:
    def test_basic(self):
        exc = StorageError("upload failed")
        assert exc.message == "upload failed"
        assert exc.operation is None
        assert exc.path is None
        assert exc.details == {}

    def test_with_operation_and_path(self):
        exc = StorageError("failed", operation="upload", path="gs://bucket/file")
        assert exc.operation == "upload"
        assert exc.path == "gs://bucket/file"
        assert exc.details == {"operation": "upload", "path": "gs://bucket/file"}

    def test_with_operation_only(self):
        exc = StorageError("failed", operation="download")
        assert exc.details == {"operation": "download"}
        assert "path" not in exc.details

    def test_with_path_only(self):
        exc = StorageError("failed", path="gs://bucket/file.pdf")
        assert exc.path == "gs://bucket/file.pdf"
        assert exc.operation is None
        assert exc.details == {"path": "gs://bucket/file.pdf"}
        assert "operation" not in exc.details

    def test_str(self):
        exc = StorageError("oops")
        assert str(exc) == "oops"

    def test_is_exception(self):
        exc = StorageError("err")
        assert isinstance(exc, Exception)
        assert exc.args == ("err",)


class TestRepositoryError:
    def test_basic(self):
        exc = RepositoryError("repo error")
        assert exc.message == "repo error"
        assert exc.details == {}

    def test_with_details(self):
        exc = RepositoryError("repo error", details={"key": "val"})
        assert exc.details == {"key": "val"}

    def test_none_details_defaults_to_empty(self):
        exc = RepositoryError("repo error", details=None)
        assert exc.details == {}

    def test_str(self):
        exc = RepositoryError("message")
        assert str(exc) == "message"


class TestBigQueryError:
    def test_basic(self):
        exc = BigQueryError("bq error")
        assert exc.message == "bq error"
        assert exc.details == {}

    def test_with_all_fields(self):
        exc = BigQueryError("error", dataset="ds1", table="t1", query="SELECT 1")
        assert exc.details == {"dataset": "ds1", "table": "t1", "query": "SELECT 1"}

    def test_partial_fields(self):
        exc = BigQueryError("error", dataset="ds1")
        assert exc.details == {"dataset": "ds1"}
        assert "table" not in exc.details
        assert "query" not in exc.details

    def test_with_table_only(self):
        exc = BigQueryError("error", table="my_table")
        assert exc.details == {"table": "my_table"}
        assert "dataset" not in exc.details
        assert "query" not in exc.details

    def test_with_query_only(self):
        exc = BigQueryError("error", query="SELECT * FROM jobs")
        assert exc.details == {"query": "SELECT * FROM jobs"}
        assert "dataset" not in exc.details
        assert "table" not in exc.details

    def test_is_repository_error(self):
        exc = BigQueryError("error")
        assert isinstance(exc, RepositoryError)

    def test_message_and_str(self):
        exc = BigQueryError("bq failure msg")
        assert exc.message == "bq failure msg"
        assert str(exc) == "bq failure msg"
