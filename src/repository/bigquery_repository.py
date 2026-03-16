"""BigQuery repository for analytics and reporting."""

import json
import logging
from datetime import datetime
from typing import Any

from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery

from config.constants import settings
from repository.repository_exception import StorageError

logger = logging.getLogger(__name__)


class BigQueryRepository:
    """Repository for BigQuery operations."""

    def __init__(
        self, client: bigquery.Client | None = None, dataset: str | None = None
    ):
        """Initialize BigQuery repository with cached client."""
        self.client = client or bigquery.Client(
            project=settings.GOOGLE_CLOUD_PROJECT_ID
        )
        self.dataset = dataset or settings.BIGQUERY_DATASET
        self.jobs_table = f"{self.client.project}.{self.dataset}.translation_jobs"
        self.report_table = f"{self.client.project}.{self.dataset}.translation_report"

    async def write_job_completion(self, job_data: dict[str, Any]) -> None:
        """Write job completion analytics."""
        try:
            # Prepare row with required fields
            row = {
                "job_id": job_data["job_id"],
                "document_id": job_data.get("document_id", job_data["job_id"]),
                "status": job_data["status"],
                "domain": job_data.get("domain"),
                "lang_in": job_data.get("lang_in"),
                "lang_out": job_data.get("lang_out"),
                "file_size_bytes": job_data.get("file_size_bytes"),
                "processing_seconds": job_data.get("processing_seconds"),
                "pages_processed": job_data.get("pages_processed"),
                "created_at": job_data.get("created_at", datetime.utcnow()),
                "completed_at": job_data.get("completed_at", datetime.utcnow()),
                "updated_at": datetime.utcnow(),
                "output_gs_uris": json.dumps(job_data.get("output_gs_uris", {})),
                "error_message": job_data.get("error_message"),
            }

            # Remove None values
            row = {k: v for k, v in row.items() if v is not None}

            errors = self.client.insert_rows_json(self.jobs_table, [row])
            if errors:
                raise Exception(f"BigQuery insert errors: {errors}")

            logger.info(f"Wrote job analytics to BigQuery: {job_data['job_id']}")

        except GoogleAPIError as e:
            logger.error(f"BigQuery API error: {e}")
            raise StorageError(
                f"Failed to write to BigQuery: {e}",
                operation="insert",
                path=self.jobs_table,
            ) from e
        except Exception as e:
            logger.error(f"Failed to write job analytics: {e}")
            raise StorageError(
                f"Failed to write job analytics: {e}",
                operation="insert",
                path=self.jobs_table,
            ) from e

    async def write_translation_report(
        self, job_id: str, translations: list[dict[str, Any]]
    ) -> None:
        """Write detailed translation reports."""
        try:
            rows = []
            now = datetime.utcnow()

            for translation in translations:
                row = {
                    "document_id": job_id,
                    "lang_in": translation["lang_in"],
                    "lang_out": translation["lang_out"],
                    "translate_engine": translation.get("engine", "openai"),
                    "translate_engine_params": json.dumps(
                        translation.get("params", {})
                    ),
                    "original_text": translation["original_text"],
                    "translated_text": translation["translated_text"],
                    "created_at": now,
                }
                rows.append(row)

            if rows:
                errors = self.client.insert_rows_json(self.report_table, rows)
                if errors:
                    raise Exception(f"BigQuery insert errors: {errors}")

                logger.info(f"Wrote {len(rows)} translation reports to BigQuery")

        except GoogleAPIError as e:
            logger.error(f"BigQuery API error: {e}")
            raise StorageError(
                f"Failed to write translation reports: {e}",
                operation="insert",
                path=self.report_table,
            ) from e
        except Exception as e:
            logger.error(f"Failed to write translation reports: {e}")
            raise StorageError(
                f"Failed to write translation reports: {e}",
                operation="insert",
                path=self.report_table,
            ) from e

    async def get_job_analytics(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:  # noqa: S608
        """Get job analytics with optional filtering."""
        try:
            query = f"""
            SELECT
                status,
                domain,
                lang_in,
                lang_out,
                COUNT(*) as job_count,
                AVG(processing_seconds) as avg_processing_time,
                AVG(pages_processed) as avg_pages,
                SUM(file_size_bytes) as total_file_size
            FROM `{self.jobs_table}`
            WHERE 1=1
            """

            params = {}

            if start_date:
                query += " AND created_at >= @start_date"
                params["start_date"] = start_date

            if end_date:
                query += " AND created_at <= @end_date"
                params["end_date"] = end_date

            if domain:
                query += " AND domain = @domain"
                params["domain"] = domain

            query += " GROUP BY status, domain, lang_in, lang_out"

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(k, "STRING", v)
                    for k, v in params.items()
                ]
            )

            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()

            return [dict(row) for row in results]

        except GoogleAPIError as e:
            logger.error(f"BigQuery API error: {e}")
            raise StorageError(
                f"Failed to get job analytics: {e}",
                operation="query",
                path=self.jobs_table,
            ) from e

    async def get_daily_stats(self, days: int = 30) -> list[dict[str, Any]]:  # noqa: S608
        """Get daily statistics for the last N days."""
        try:
            query = f"""
            SELECT
                DATE(created_at) as date,
                COUNT(*) as total_jobs,
                COUNTIF(status = 'completed') as completed_jobs,
                COUNTIF(status = 'failed') as failed_jobs,
                AVG(processing_seconds) as avg_processing_time,
                SUM(file_size_bytes) as total_file_size
            FROM `{self.jobs_table}`
            WHERE created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("days", "INT64", days)]
            )

            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()

            return [dict(row) for row in results]

        except GoogleAPIError as e:
            logger.error(f"BigQuery API error: {e}")
            raise StorageError(
                f"Failed to get daily stats: {e}",
                operation="query",
                path=self.jobs_table,
            ) from e
