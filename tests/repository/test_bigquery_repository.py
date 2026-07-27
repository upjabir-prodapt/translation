from datetime import UTC
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from google.api_core.exceptions import GoogleAPIError
from src.repository.bigquery_repository import BigQueryRepository
from src.repository.repository_exception import BigQueryError


@pytest.fixture
def mock_bq_client():
    client = MagicMock()
    client.project = "test-project"
    return client


@pytest.fixture
def repo(mock_bq_client):
    with patch(
        "src.repository.bigquery_repository.bigquery.Client",
        return_value=mock_bq_client,
    ):
        with patch("src.repository.bigquery_repository.settings") as mock_settings:
            mock_settings.GOOGLE_CLOUD_PROJECT = "test-project"
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

    @pytest.mark.asyncio
    async def test_insert_rows_json_success(self, repo, mock_bq_client):
        mock_bq_client.insert_rows_json.return_value = []
        await repo._insert_rows_json("table", [{"row": 1}])
        mock_bq_client.insert_rows_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_rows_json_error(self, repo, mock_bq_client):
        mock_bq_client.insert_rows_json.return_value = [{"error": "fail"}]
        with pytest.raises(BigQueryError, match="BigQuery insert errors"):
            await repo._insert_rows_json("table", [{"row": 1}])

    @pytest.mark.asyncio
    async def test_insert_rows_json_google_error(self, repo, mock_bq_client):
        mock_bq_client.insert_rows_json.side_effect = GoogleAPIError("Fail")
        with pytest.raises(BigQueryError, match="Failed to write to BigQuery"):
            await repo._insert_rows_json("table", [{"row": 1}])

    @pytest.mark.asyncio
    async def test_upsert_translation_job_missing_id(self, repo):
        with pytest.raises(BigQueryError, match="job_id is required"):
            await repo.upsert_translation_job({})

    @pytest.mark.asyncio
    async def test_upsert_translation_job_success(self, repo, mock_bq_client):
        mock_job = {
            "job_id": "job1",
            "status": "processing",
            "submitted_at": datetime.now(UTC).isoformat(),
        }
        mock_query_job = MagicMock()
        mock_bq_client.query.return_value = mock_query_job
        await repo.upsert_translation_job(mock_job)
        mock_bq_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_translation_job_includes_batch_metadata(
        self, repo, mock_bq_client
    ):
        mock_bq_client.query.return_value = MagicMock()
        await repo.upsert_translation_job(
            {
                "job_id": "job1",
                "batch_id": "batch1",
                "batch_index": 2,
            }
        )

        job_config = mock_bq_client.query.call_args.kwargs["job_config"]
        parameters = {
            parameter.name: parameter.value for parameter in job_config.query_parameters
        }
        assert parameters["batch_id"] == "batch1"
        assert parameters["batch_index"] == 2

    @pytest.mark.asyncio
    async def test_get_translation_job_found(self, repo, mock_bq_client):
        mock_row = MagicMock()
        mock_row.get = lambda k, d=None: {"job_id": "job1", "status": "completed"}.get(
            k, d
        )

        mock_query_job = MagicMock()
        mock_query_job.result = MagicMock(return_value=[mock_row])
        mock_bq_client.query.return_value = mock_query_job

        res = await repo.get_translation_job("job1")
        assert res["job_id"] == "job1"

    @pytest.mark.asyncio
    async def test_get_translation_jobs_by_ids_uses_array_parameter(
        self, repo, mock_bq_client
    ):
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = []
        mock_bq_client.query.return_value = mock_query_job

        result = await repo.get_translation_jobs_by_ids(["job1", "job2"])

        assert result == []
        query = mock_bq_client.query.call_args.args[0]
        assert "IN UNNEST(@job_ids)" in query
        job_config = mock_bq_client.query.call_args.kwargs["job_config"]
        parameter = job_config.query_parameters[0]
        assert parameter.name == "job_ids"
        assert parameter.values == ["job1", "job2"]

    @pytest.mark.asyncio
    async def test_list_translation_jobs(self, repo, mock_bq_client):
        mock_row = MagicMock()
        mock_row.get = lambda k, d=None: {"job_id": "1", "status": "queued"}.get(k, d)

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [mock_row]
        mock_bq_client.query.return_value = mock_query_job

        res = await repo.list_translation_jobs(status="queued", limit=10, offset=0)
        assert len(res) == 1
        assert res[0]["job_id"] == "1"

    @pytest.mark.asyncio
    async def test_patch_translation_job(self, repo, mock_bq_client):
        with patch.object(
            repo, "get_translation_job", return_value={"job_id": "j1", "status": "q"}
        ):
            with patch.object(repo, "upsert_translation_job") as mock_upsert:
                await repo.patch_translation_job("j1", {"status": "p"})
                mock_upsert.assert_called_once()
                args = mock_upsert.call_args[0][0]
                assert args["status"] == "p"

    def test_deserialize_json(self, repo):
        assert repo._deserialize_json(None) is None
        assert repo._deserialize_json('{"a": 1}') == {"a": 1}
        assert repo._deserialize_json("not json") == "not json"
        assert repo._deserialize_json({"already": "dict"}) == {"already": "dict"}

    def test_deserialize_job_row_includes_batch_metadata(self, repo):
        row = {
            "job_id": "job1",
            "status": "queued",
            "batch_id": "batch1",
            "batch_index": 2,
        }
        result = repo._deserialize_job_row(row)
        assert result["batch_id"] == "batch1"
        assert result["batch_index"] == 2

    @pytest.mark.asyncio
    async def test_write_dlp_tokens(self, repo, mock_bq_client):
        mock_bq_client.insert_rows_json.return_value = []
        rows = [{"job_id": "j1", "chunk_index": 0, "token": "t", "original_value": "o"}]
        await repo.write_dlp_tokens(rows)
        mock_bq_client.insert_rows_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_dlp_tokens(self, repo, mock_bq_client):
        mock_row = MagicMock()
        mock_data = {
            "job_id": "j1",
            "chunk_index": 0,
            "token": "t",
            "original_value": "o",
        }
        mock_row.get = lambda k, d=None: mock_data.get(k, d)

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [mock_row]
        mock_bq_client.query.return_value = mock_query_job

        res = await repo.read_dlp_tokens("j1")
        assert len(res) == 1
        assert res[0]["token"] == "t"  # noqa: S105

    @pytest.mark.asyncio
    async def test_write_cost_attribution(self, repo, mock_bq_client):
        mock_bq_client.insert_rows_json.return_value = []
        await repo.write_cost_attribution({"job_id": "j1", "cost_usd": 0.5})
        mock_bq_client.insert_rows_json.assert_called_once()
