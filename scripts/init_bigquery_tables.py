"""
Initialize BigQuery tables for TranslateDoc.

Run this script once to create the required BigQuery dataset and tables.

Usage:
    python scripts/init_bigquery_tables.py --project-id YOUR_PROJECT_ID
"""

import argparse
import os

from dotenv import load_dotenv
from google.cloud import bigquery

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


def create_translation_report_table(client: bigquery.Client, dataset_id: str):
    """Create translation_report table."""
    table_id = f"{client.project}.{dataset_id}.translation_report"

    schema = [
        bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lang_in", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lang_out", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("translate_engine", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("translate_engine_params", "JSON", mode="REQUIRED"),
        bigquery.SchemaField("original_text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("translated_text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_id, schema=schema)

    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="created_at",
    )

    table.clustering_fields = ["document_id", "translate_engine"]

    try:
        table = client.create_table(table)
        print(f"Created table {table_id}")
    except Exception as e:
        print(f"Table {table_id} already exists or error: {e}")


def create_translation_jobs_table(client: bigquery.Client, dataset_id: str):
    """Create translation_jobs table."""
    table_id = f"{client.project}.{dataset_id}.translation_jobs"

    schema = [
        bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("output_file_gcs_path", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_id, schema=schema)

    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="created_at",
    )

    table.clustering_fields = ["document_id", "status"]

    try:
        table = client.create_table(table)
        print(f"Created table {table_id}")
    except Exception as e:
        print(f"Table {table_id} already exists or error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize BigQuery tables for TranslateDoc"
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
        help="GCP Project ID",
    )
    parser.add_argument(
        "--dataset", default=os.getenv("BIGQUERY_DATASET"), help="BigQuery dataset name"
    )
    parser.add_argument(
        "--location", default=os.getenv("BIGQUERY_LOCATION"), help="BigQuery location"
    )

    args = parser.parse_args()

    client = bigquery.Client(project=args.project_id)

    print(f"Initializing BigQuery tables in project {args.project_id}...")

    create_dataset(client, args.dataset, args.location)
    create_translation_report_table(client, args.dataset)
    create_translation_jobs_table(client, args.dataset)

    print("\n✅ BigQuery initialization complete!")
    print(f"\nDataset: {args.project_id}.{args.dataset}")
    print("Tables created:")
    print(f"  - {args.dataset}.translation_report")
    print(f"  - {args.dataset}.translation_jobs")
    print(f"  - {args.dataset}.glossaries")


if __name__ == "__main__":
    main()
