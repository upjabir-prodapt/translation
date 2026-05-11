import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.repository.bigquery_repository import BigQueryRepository
from src.repository.repository_exception import StorageError
from datetime import datetime, UTC

@pytest.fixture
def mock_bq_client():
    client = MagicMock()
    client.project = "test-project"
    return client

@pytest.fixture
def repo(mock_bq_client):
    with patch("src.repository.bigquery_repository.bigquery.Client", return_value=mock_bq_client):
        with patch("src.repository.bigquery_repository.settings") as mock_settings:
            mock_settings.GOOGLE_CLOUD_PROJECT_ID = "test-project"
            mock_settings.BIGQUERY_DATASET = "test_dataset"
            mock_settings.BIGQUERY_TABLE = "jobs"
            mock_settings.BIGQUERY_COST_TABLE = "cost"
            mock_settings.BIGQUERY_DLP_TABLE = "dlp"
            return BigQueryRepository()

class TestBigQueryRepository:
    def test_to_json_string(self, repo):
        assert repo._to_json_string(None) is None
        assert repo._to_json_string("string") == "string"
        assert repo._to_json_string({"a": 1}) == '{"a": 1}'

    async def test_insert_rows_json_success(self, repo, mock_bq_client):
        mock_bq_client.insert_rows_json.return_value = []
        await repo._insert_rows_json("table", [{"row": 1}])
        mock_bq_client.insert_rows_json.assert_called_once()

    async def test_insert_rows_json_error(self, repo, mock_bq_client):
        mock_bq_client.insert_rows_json.return_value = [{"error": "fail"}]
        with pytest.raises(StorageError, match="BigQuery insert errors"):
            await repo._insert_rows_json("table", [{"row": 1}])

    async def test_upsert_translation_job_missing_id(self, repo):
        with pytest.raises(StorageError, match="job_id is required"):
            await repo.upsert_translation_job({})

    async def test_upsert_translation_job_success(self, repo, mock_bq_client):
        mock_job = {
            "job_id": "job1",
            "status": "processing",
            "submitted_at": datetime.now(UTC).isoformat()
        }
        await repo.upsert_translation_job(mock_job)
        mock_bq_client.query.assert_called_once()

    async def test_get_translation_job_found(self, repo, mock_bq_client):
        mock_row = MagicMock()
        # Mocking row behavior is tricky, let's assume it returns a dict-like object
        mock_row.items.return_value = [("job_id", "job1"), ("status", "completed")]
        mock_row.get = lambda k, d=None: {"job_id": "job1", "status": "completed"}.get(k, d)
        
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [mock_row]
        mock_bq_client.query.return_value = mock_query_job
        
        res = await repo.get_translation_job("job1")
        assert res["job_id"] == "job1"

    async def test_get_translation_job_not_found(self, repo, mock_bq_client):
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = []
        mock_bq_client.query.return_value = mock_query_job
        
        res = await repo.get_translation_job("job1")
        assert res is None

    async def test_list_translation_jobs(self, repo, mock_bq_client):
        mock_row = MagicMock()
        mock_row.items.return_value = [("job_id", "1")]
        mock_row.get = lambda k, d=None: {"job_id": "1"}.get(k, d)
        
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [mock_row]
        mock_bq_client.query.return_value = mock_query_job
        
        res = await repo.list_translation_jobs(limit=10)
        assert len(res) == 1
        assert res[0]["job_id"] == "1"

    async def test_patch_translation_job(self, repo, mock_bq_client):
        # Mock get_translation_job (first call)
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = []
        mock_bq_client.query.return_value = mock_query_job
        
        await repo.patch_translation_job("job1", {"status": "cancelled"})
        assert mock_bq_client.query.call_count == 2


