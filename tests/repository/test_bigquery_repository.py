"""Unit tests for repository.bigquery_repository."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from src.repository.bigquery_repository import BigQueryRepository
from src.repository.repository_exception import StorageError


def _make_repo(client=None) -> BigQueryRepository:
    mocked = client or MagicMock()
    mocked.project = "test-project"
    mocked.insert_rows_json.return_value = []
    query_job = MagicMock()
    query_job.result.return_value = []
    mocked.query.return_value = query_job
    return BigQueryRepository(client=mocked, dataset="translation")


class TestJobPersistence:
    async def test_upsert_translation_job_executes_query(self):
        repo = _make_repo()
        await repo.upsert_translation_job({"job_id": "j1", "status": "queued"})
        repo.client.query.assert_called_once()

    async def test_patch_translation_job_merges_existing_data(self):
        repo = _make_repo()
        repo.get_translation_job = AsyncMock(
            return_value={"job_id": "j1", "status": "queued", "submitted_at": None}
        )
        repo.upsert_translation_job = AsyncMock()
        await repo.patch_translation_job("j1", {"status": "processing"})
        repo.upsert_translation_job.assert_called_once()

    async def test_write_job_completion_inserts_rows(self):
        repo = _make_repo()
        await repo.write_job_completion({"job_id": "j1", "status": "completed"})
        repo.client.insert_rows_json.assert_called_once()


class TestCostAndDlpTables:
    async def test_write_cost_attribution_inserts_row(self):
        repo = _make_repo()
        await repo.write_cost_attribution({"job_id": "j1"})
        repo.client.insert_rows_json.assert_called_once()

    async def test_write_dlp_tokens_inserts_rows(self):
        repo = _make_repo()
        await repo.write_dlp_tokens(
            [
                {
                    "job_id": "j1",
                    "chunk_index": 0,
                    "token": "__DLP_TOKEN_0001__",
                    "original_value": "alice@example.com",
                }
            ]
        )
        repo.client.insert_rows_json.assert_called_once()

    async def test_insert_errors_raise_storage_error(self):
        client = MagicMock()
        client.project = "test-project"

        def insert_rows_json_errors(_table, _rows):
            return [{"error": "boom"}]

        client.insert_rows_json = insert_rows_json_errors
        repo = _make_repo(client=client)
        with pytest.raises(StorageError):
            await repo.write_cost_attribution({"job_id": "j1"})
