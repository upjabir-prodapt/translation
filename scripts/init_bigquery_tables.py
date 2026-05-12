"""
Initialize BigQuery tables for TranslateDoc.

Run this script once to create the required BigQuery dataset and tables.

Usage:
    python scripts/init_bigquery_tables.py
    python scripts/init_bigquery_tables.py --project-id YOUR_PROJECT_ID --dataset YOUR_DATASET
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery
from src.config.constants import settings

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()


def create_dataset(client: bigquery.Client, dataset_id: str, location: str = "US"):
    """Create BigQuery dataset if it doesn't exist."""
    dataset_ref = f"{client.project}.{dataset_id}"

    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_ref} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"Created dataset {dataset_ref}")


def create_translation_jobs_table(client: bigquery.Client, dataset_id: str) -> None:
    """Create translation jobs table (name from settings)."""
    table_id = f"{client.project}.{dataset_id}.{settings.BIGQUERY_TABLE}"

    schema = [
        bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("source_document", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("translation_config", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("cost_attribution", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("result", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("submitted_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("completed_at", "TIMESTAMP", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="submitted_at",
    )
    table.clustering_fields = ["status", "job_id"]

    try:
        table = client.create_table(table)
        print(f"Created table {table_id}")
    except Exception as e:
        print(f"Table {table_id} already exists or error: {e}")


def create_cost_attribution_table(client: bigquery.Client, dataset_id: str) -> None:
    """Create cost attribution table (name from settings)."""
    table_id = f"{client.project}.{dataset_id}.{settings.BIGQUERY_COST_TABLE}"
    schema = [
        bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("business_unit", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("organization", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("model_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("intent", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("input_tokens", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("output_tokens", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("cost_usd", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",
    )
    try:
        client.create_table(table)
        print(f"Created table {table_id}")
    except Exception as e:
        print(f"Table {table_id} already exists or error: {e}")


def create_dlp_tokens_table(client: bigquery.Client, dataset_id: str) -> None:
    """Create DLP tokens table (name from settings)."""
    table_id = f"{client.project}.{dataset_id}.{settings.BIGQUERY_DLP_TABLE}"
    schema = [
        bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk_index", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("token", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("original_value", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("info_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("masked_at", "TIMESTAMP", mode="NULLABLE"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="masked_at",
    )
    table.clustering_fields = ["job_id", "chunk_index"]
    try:
        client.create_table(table)
        print(f"Created table {table_id}")
    except Exception as e:
        print(f"Table {table_id} already exists or error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize BigQuery tables for TranslateDoc"
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="GCP Project ID (default: GOOGLE_CLOUD_PROJECT_ID from settings)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="BigQuery dataset name (default: BIGQUERY_DATASET from settings)",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="BigQuery dataset location (default: BIGQUERY_LOCATION from settings)",
    )

    args = parser.parse_args()

    project_id = args.project_id or settings.GOOGLE_CLOUD_PROJECT_ID
    dataset = args.dataset or settings.BIGQUERY_DATASET
    location = args.location or settings.BIGQUERY_LOCATION

    client = bigquery.Client(project=project_id)

    print(f"Initializing BigQuery tables in project {project_id}...")

    create_dataset(client, dataset, location)
    create_translation_jobs_table(client, dataset)
    create_cost_attribution_table(client, dataset)
    create_dlp_tokens_table(client, dataset)

    print("\n✅ BigQuery initialization complete!")
    print(f"\nDataset: {project_id}.{dataset}")
    print("Tables created:")
    print(f"  - {dataset}.{settings.BIGQUERY_TABLE}")
    print(f"  - {dataset}.{settings.BIGQUERY_COST_TABLE}")
    print(f"  - {dataset}.{settings.BIGQUERY_DLP_TABLE}")


if __name__ == "__main__":
    main()
