"""
Unit tests for repository/bigquery_repository.py — BigQueryRepository.

The BigQuery client's insert_rows_json is synchronous and is mocked directly.
write_job_completion and write_cost_attribution are async wrappers.
"""

from unittest.mock import MagicMock

import pytest
from fixtures.sample_data import BIGQUERY_JOB_DATA
from google.api_core.exceptions import GoogleAPIError

from repository.bigquery_repository import BigQueryRepository
from repository.repository_exception import StorageError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(bq_client=None) -> BigQueryRepository:
    client = bq_client or MagicMock()
    client.project = "test-project"
    client.insert_rows_json.return_value = []  # empty list = no errors
    repo = BigQueryRepository(client=client, dataset="test_dataset")
    return repo


# ---------------------------------------------------------------------------
# write_job_completion
# ---------------------------------------------------------------------------


class TestWriteJobCompletion:
    async def test_calls_insert_rows_json(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        await repo.write_job_completion(BIGQUERY_JOB_DATA)
        client.insert_rows_json.assert_called_once()

    async def test_row_contains_job_id(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        await repo.write_job_completion(BIGQUERY_JOB_DATA)
        row = client.insert_rows_json.call_args[0][1][0]
        assert row["job_id"] == BIGQUERY_JOB_DATA["job_id"]

    async def test_row_contains_status(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        await repo.write_job_completion(BIGQUERY_JOB_DATA)
        row = client.insert_rows_json.call_args[0][1][0]
        assert row["status"] == "completed"

    async def test_none_values_excluded_from_row(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        data = dict(BIGQUERY_JOB_DATA)
        data["error_message"] = None
        await repo.write_job_completion(data)

        row = client.insert_rows_json.call_args[0][1][0]
        assert "error_message" not in row

    async def test_bigquery_errors_raise_storage_error(self):
        client = MagicMock()
        client.project = "test-project"
        repo = _make_repo(bq_client=client)
        # Set AFTER _make_repo, which unconditionally sets return_value=[].
        client.insert_rows_json.return_value = [
            {"errors": [{"reason": "quota exceeded"}]}
        ]

        with pytest.raises(StorageError, match="Failed to write"):
            await repo.write_job_completion(BIGQUERY_JOB_DATA)

    async def test_google_api_error_raises_storage_error(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.side_effect = GoogleAPIError("quota exceeded")
        repo = _make_repo(bq_client=client)

        with pytest.raises(StorageError, match="Failed to write"):
            await repo.write_job_completion(BIGQUERY_JOB_DATA)

    async def test_general_exception_raises_storage_error(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.side_effect = Exception("network error")
        repo = _make_repo(bq_client=client)

        with pytest.raises(StorageError):
            await repo.write_job_completion(BIGQUERY_JOB_DATA)

    async def test_output_gs_uris_serialized_as_json(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        await repo.write_job_completion(BIGQUERY_JOB_DATA)
        row = client.insert_rows_json.call_args[0][1][0]
        assert isinstance(row.get("output_gs_uris"), str)  # serialized JSON

    async def test_minimal_job_data_does_not_raise(self):
        """Verify that missing optional fields default gracefully."""
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        minimal = {"job_id": "min-job", "status": "completed"}
        await repo.write_job_completion(minimal)
        client.insert_rows_json.assert_called_once()

    @pytest.mark.parametrize("token_usage", [None, 0, 5000])
    async def test_token_usage_defaults(self, token_usage):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        data = dict(BIGQUERY_JOB_DATA)
        data["token_usage"] = token_usage
        await repo.write_job_completion(data)
        row = client.insert_rows_json.call_args[0][1][0]
        assert row.get("token_usage") == int(token_usage or 0)


# ---------------------------------------------------------------------------
# write_cost_attribution
# ---------------------------------------------------------------------------


class TestWriteCostAttribution:
    async def test_calls_insert_rows_json(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        data = {
            "job_id": "job-1",
            "user_id": "alice@example.com",
            "business_unit": "engineering",
            "model_id": "gpt-4o-mini",
            "cost_usd": 1.50,
            "input_tokens": 3000,
            "output_tokens": 2000,
        }
        await repo.write_cost_attribution(data)
        client.insert_rows_json.assert_called_once()

    async def test_row_contains_job_id(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        await repo.write_cost_attribution({"job_id": "job-42", "cost_usd": 0.50})
        row = client.insert_rows_json.call_args[0][1][0]
        assert row["job_id"] == "job-42"

    async def test_none_values_excluded(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.return_value = []
        repo = _make_repo(bq_client=client)

        data = {"job_id": "job-1", "user_id": None, "cost_usd": 1.0}
        await repo.write_cost_attribution(data)
        row = client.insert_rows_json.call_args[0][1][0]
        assert "user_id" not in row

    async def test_insert_errors_raise_storage_error(self):
        client = MagicMock()
        client.project = "test-project"
        repo = _make_repo(bq_client=client)
        # Set AFTER _make_repo so it isn't overwritten to [].
        client.insert_rows_json.return_value = [{"errors": [{"reason": "forbidden"}]}]

        with pytest.raises(StorageError):
            await repo.write_cost_attribution({"job_id": "j1"})

    async def test_api_error_raises_storage_error(self):
        client = MagicMock()
        client.project = "test-project"
        client.insert_rows_json.side_effect = GoogleAPIError("error")
        repo = _make_repo(bq_client=client)

        with pytest.raises(StorageError):
            await repo.write_cost_attribution({"job_id": "j1"})
