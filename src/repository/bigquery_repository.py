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

    async def write_job_completion(self, job_data: dict[str, Any]) -> None:
        """Write job completion analytics."""
        try:
            # Prepare row with required fields
            row = {
                "job_id": job_data["job_id"],
                "status": job_data["status"],
                "domain": job_data.get("domain"),
                "lang_in": job_data.get("lang_in"),
                "lang_out": job_data.get("lang_out"),
                "user": job_data.get("user"),
                "department": job_data.get("department"),
                "file_size_bytes": job_data.get("file_size_bytes"),
                "processing_seconds": job_data.get("processing_seconds"),
                "pages_processed": job_data.get("pages_processed"),
                "created_at": job_data.get("created_at", datetime.utcnow()),
                "completed_at": job_data.get("completed_at", datetime.utcnow()),
                "updated_at": datetime.utcnow(),
                "output_gs_uris": json.dumps(job_data.get("output_gs_uris", {})),
                "quality_report": json.dumps(job_data.get("quality_report", {})),
                "token_usage": int(job_data.get("token_usage", 0) or 0),
                "total_cost_usd": float(job_data.get("total_cost_usd", 0.0) or 0.0),
                "iteration_details": json.dumps(
                    job_data.get("iteration_details", job_data.get("attempts", []))
                ),
                "selected_model": job_data.get("selected_model"),
                "attempt_count": int(job_data.get("attempt_count", 0) or 0),
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
