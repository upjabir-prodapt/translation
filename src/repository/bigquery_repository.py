"""BigQuery repository for translation job persistence."""

import asyncio
import json
from datetime import UTC
from datetime import datetime
from typing import Any

from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery

from src.config.constants import settings
from src.repository.repository_exception import StorageError


class BigQueryRepository:
    """Repository for BigQuery operations."""

    def __init__(
        self,
        client: bigquery.Client | None = None,
        dataset: str | None = None,
    ):
        self.client = client or bigquery.Client(project=settings.GOOGLE_CLOUD_PROJECT_ID)
        self.dataset = dataset or settings.BIGQUERY_DATASET
        self.jobs_table = f"{self.client.project}.{self.dataset}.translation_jobs"
        self.cost_attribution_table = f"{self.client.project}.{self.dataset}.cost_attribution"
        self.dlp_tokens_table = f"{self.client.project}.{self.dataset}.dlp_tokens"

    def _to_json_string(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    async def _insert_rows_json(self, table: str, rows: list[dict[str, Any]]) -> None:
        try:
            errors = await asyncio.to_thread(self.client.insert_rows_json, table, rows)
            if errors:
                raise StorageError(
                    f"BigQuery insert errors: {errors}",
                    operation="insert",
                    path=table,
                )
        except GoogleAPIError as exc:
            raise StorageError(
                f"Failed to write to BigQuery: {exc}",
                operation="insert",
                path=table,
            ) from exc

    async def upsert_translation_job(self, job_data: dict[str, Any]) -> None:
        """Insert or update a translation job row."""
        if "job_id" not in job_data:
            raise StorageError(
                "job_id is required for translation_jobs upsert",
                operation="upsert",
                path=self.jobs_table,
            )

        submitted_at = job_data.get("submitted_at") or datetime.now(UTC)
        if isinstance(submitted_at, str):
            submitted_at = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
        completed_at = job_data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

        row = {
            "job_id": str(job_data["job_id"]),
            "status": str(job_data.get("status", "queued")),
            "source_document": self._to_json_string(job_data.get("source_document")),
            "translation_config": self._to_json_string(job_data.get("translation_config")),
            "cost_attribution": self._to_json_string(job_data.get("cost_attribution")),
            "result": self._to_json_string(job_data.get("result")),
            "error_message": job_data.get("error_message"),
            "source_hash": job_data.get("source_hash"),
            "submitted_at": submitted_at,
            "completed_at": completed_at,
        }

        query = f"""
        MERGE `{self.jobs_table}` T
        USING (
            SELECT
                @job_id AS job_id,
                @status AS status,
                @source_document AS source_document,
                @translation_config AS translation_config,
                @cost_attribution AS cost_attribution,
                @result AS result,
                @error_message AS error_message,
                @source_hash AS source_hash,
                @submitted_at AS submitted_at,
                @completed_at AS completed_at
        ) S
        ON T.job_id = S.job_id
        WHEN MATCHED THEN UPDATE SET
            status = S.status,
            source_document = S.source_document,
            translation_config = S.translation_config,
            cost_attribution = S.cost_attribution,
            result = S.result,
            error_message = S.error_message,
            source_hash = S.source_hash,
            submitted_at = S.submitted_at,
            completed_at = S.completed_at
        WHEN NOT MATCHED THEN
            INSERT (job_id, status, source_document, translation_config, cost_attribution, result, error_message, source_hash, submitted_at, completed_at)
            VALUES (S.job_id, S.status, S.source_document, S.translation_config, S.cost_attribution, S.result, S.error_message, S.source_hash, S.submitted_at, S.completed_at)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("job_id", "STRING", row["job_id"]),
                bigquery.ScalarQueryParameter("status", "STRING", row["status"]),
                bigquery.ScalarQueryParameter(
                    "source_document", "STRING", row["source_document"]
                ),
                bigquery.ScalarQueryParameter(
                    "translation_config", "STRING", row["translation_config"]
                ),
                bigquery.ScalarQueryParameter(
                    "cost_attribution", "STRING", row["cost_attribution"]
                ),
                bigquery.ScalarQueryParameter("result", "STRING", row["result"]),
                bigquery.ScalarQueryParameter(
                    "error_message", "STRING", row["error_message"]
                ),
                bigquery.ScalarQueryParameter("source_hash", "STRING", row["source_hash"]),
                bigquery.ScalarQueryParameter(
                    "submitted_at", "TIMESTAMP", row["submitted_at"]
                ),
                bigquery.ScalarQueryParameter(
                    "completed_at", "TIMESTAMP", row["completed_at"]
                ),
            ]
        )
        try:
            query_job = await asyncio.to_thread(
                self.client.query,
                query,
                job_config=job_config,
            )
            await asyncio.to_thread(query_job.result)
        except GoogleAPIError as exc:
            raise StorageError(
                f"Failed to upsert translation job: {exc}",
                operation="upsert",
                path=self.jobs_table,
            ) from exc

    async def get_translation_job(self, job_id: str) -> dict[str, Any] | None:
        query = f"""
        SELECT *
        FROM `{self.jobs_table}`
        WHERE job_id = @job_id
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("job_id", "STRING", job_id)]
        )
        try:
            query_job = await asyncio.to_thread(
                self.client.query, query, job_config=job_config
            )
            rows = list(await asyncio.to_thread(query_job.result))
            if not rows:
                return None
            row = rows[0]
            return self._deserialize_job_row(row)
        except GoogleAPIError as exc:
            raise StorageError(
                f"Failed to query translation job: {exc}",
                operation="select",
                path=self.jobs_table,
            ) from exc

    async def list_translation_jobs(
        self,
        status: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where_clause = "WHERE status = @status" if status else ""
        query = f"""
        SELECT *
        FROM `{self.jobs_table}`
        {where_clause}
        ORDER BY submitted_at DESC
        LIMIT @limit OFFSET @offset
        """
        params: list[bigquery.ScalarQueryParameter] = [
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("offset", "INT64", offset),
        ]
        if status:
            params.append(bigquery.ScalarQueryParameter("status", "STRING", status))

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        try:
            query_job = await asyncio.to_thread(
                self.client.query, query, job_config=job_config
            )
            rows = list(await asyncio.to_thread(query_job.result))
            return [self._deserialize_job_row(row) for row in rows]
        except GoogleAPIError as exc:
            raise StorageError(
                f"Failed to list translation jobs: {exc}",
                operation="select",
                path=self.jobs_table,
            ) from exc

    async def patch_translation_job(self, job_id: str, updates: dict[str, Any]) -> None:
        current = await self.get_translation_job(job_id)
        merged = {**(current or {"job_id": job_id}), **updates}
        merged["job_id"] = job_id
        if not merged.get("submitted_at"):
            merged["submitted_at"] = datetime.now(UTC)
        await self.upsert_translation_job(merged)

    def _deserialize_json(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _deserialize_job_row(self, row: Any) -> dict[str, Any]:
        return {
            "job_id": row.get("job_id"),
            "status": row.get("status"),
            "source_document": self._deserialize_json(row.get("source_document")),
            "translation_config": self._deserialize_json(row.get("translation_config")),
            "cost_attribution": self._deserialize_json(row.get("cost_attribution")),
            "result": self._deserialize_json(row.get("result")),
            "error_message": row.get("error_message"),
            "source_hash": row.get("source_hash"),
            "submitted_at": row.get("submitted_at"),
            "completed_at": row.get("completed_at"),
        }

    async def write_dlp_tokens(self, rows: list[dict[str, Any]]) -> None:
        """Insert DLP token mappings for one job."""
        if not rows:
            return
        prepared_rows = []
        for row in rows:
            prepared_rows.append(
                {
                    "job_id": row["job_id"],
                    "chunk_index": int(row["chunk_index"]),
                    "token": row["token"],
                    "original_value": row["original_value"],
                    "info_type": row.get("info_type"),
                    "masked_at": row.get("masked_at", datetime.now(UTC).isoformat()),
                }
            )
        await self._insert_rows_json(self.dlp_tokens_table, prepared_rows)

    async def read_dlp_tokens(self, job_id: str) -> list[dict[str, Any]]:
        query = f"""
        SELECT job_id, chunk_index, token, original_value, info_type, masked_at
        FROM `{self.dlp_tokens_table}`
        WHERE job_id = @job_id
        ORDER BY chunk_index ASC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("job_id", "STRING", job_id)]
        )
        try:
            query_job = await asyncio.to_thread(
                self.client.query, query, job_config=job_config
            )
            rows = list(await asyncio.to_thread(query_job.result))
            return [
                {
                    "job_id": row.get("job_id"),
                    "chunk_index": row.get("chunk_index"),
                    "token": row.get("token"),
                    "original_value": row.get("original_value"),
                    "info_type": row.get("info_type"),
                    "masked_at": row.get("masked_at"),
                }
                for row in rows
            ]
        except GoogleAPIError as exc:
            raise StorageError(
                f"Failed to read DLP tokens: {exc}",
                operation="select",
                path=self.dlp_tokens_table,
            ) from exc

    async def write_cost_attribution(self, data: dict[str, Any]) -> None:
        """Write cost attribution record for a completed job."""
        row = {
            "job_id": data["job_id"],
            "user_id": data.get("user_id"),
            "business_unit": data.get("business_unit"),
            "organization": data.get("organization"),
            "model_id": data.get("model_id"),
            "intent": data.get("intent"),
            "input_tokens": int(data.get("input_tokens", 0) or 0),
            "output_tokens": int(data.get("output_tokens", 0) or 0),
            "cost_usd": float(data.get("cost_usd", 0.0) or 0.0),
            "timestamp": data.get("timestamp", datetime.now(UTC).isoformat()),
        }
        await self._insert_rows_json(self.cost_attribution_table, [row])

    async def write_job_completion(self, job_data: dict[str, Any]) -> None:
        """Backward-compatible alias for legacy call sites."""
        row = {
            "job_id": job_data["job_id"],
            "status": job_data.get("status", "completed"),
            "source_document": self._to_json_string(job_data.get("source_document")),
            "translation_config": self._to_json_string(job_data.get("translation_config")),
            "cost_attribution": self._to_json_string(job_data.get("cost_attribution")),
            "result": self._to_json_string(job_data.get("result")),
            "error_message": job_data.get("error_message"),
            "source_hash": job_data.get("source_hash"),
            "submitted_at": job_data.get("submitted_at", datetime.now(UTC)),
            "completed_at": job_data.get("completed_at", datetime.now(UTC)),
        }
        row = {k: v for k, v in row.items() if v is not None}
        await self._insert_rows_json(self.jobs_table, [row])
