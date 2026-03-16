"""BigQuery Translation Cache Repository.

Optimized two-tier caching system:
- L1: In-memory dict for O(1) cache lookups
- L2: Async background thread batching writes to BigQuery
"""

import json
import logging
import os
import queue
import threading
import time

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

logger = logging.getLogger(__name__)

# BigQuery Configuration
# Set these via environment variables in your .env file
BQ_PROJECT = os.environ.get("BABELDOC_BQ_PROJECT")  # If None, uses default auth project
BQ_DATASET = os.environ.get("BABELDOC_BQ_DATASET", "babeldoc")
BQ_TABLE = os.environ.get("BABELDOC_BQ_TABLE", "translation_cache")

# Global BigQuery client and writer
_bq_client = None
_bq_writer = None


def get_bq_client():
    """Get or create BigQuery client."""
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=BQ_PROJECT)
    return _bq_client


def get_table_ref():
    """Get full table reference string."""
    client = get_bq_client()
    project = client.project or BQ_PROJECT
    return f"{project}.{BQ_DATASET}.{BQ_TABLE}"


class BQWriterThread(threading.Thread):
    """Background thread to asynchronously batch write translations to BigQuery."""

    def __init__(self):
        super().__init__(daemon=True, name="BQCacheWriter")
        self.queue = queue.Queue()
        self.batch_size = 200
        self.flush_interval = 2.0  # seconds

    def run(self):
        """Main writer thread loop."""
        batch = []
        last_flush_time = time.time()

        # Wait a moment for auth to settle before starting loop
        time.sleep(1)
        client = get_bq_client()
        table_ref = get_table_ref()

        while True:
            try:
                item = self.queue.get(timeout=0.5)
                if item is None:  # Sentinel to shutdown
                    self._flush(client, table_ref, batch)
                    break
                batch.append(item)
            except queue.Empty:
                pass

            now = time.time()
            if len(batch) >= self.batch_size or (
                batch and now - last_flush_time >= self.flush_interval
            ):
                self._flush(client, table_ref, batch)
                batch = []
                last_flush_time = now

    def _flush(self, client: bigquery.Client, table_ref: str, batch: list):
        """Flush batch to BigQuery."""
        if not batch:
            return
        try:
            # Deduplicate within the batch (keep the latest)
            unique_batch = {}
            for row in batch:
                key = (
                    row["translate_engine"],
                    row["translate_engine_params"],
                    row["original_text"],
                )
                unique_batch[key] = row

            rows_to_insert = list(unique_batch.values())
            errors = client.insert_rows_json(table_ref, rows_to_insert)
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.debug(f"Inserted {len(rows_to_insert)} rows to BigQuery")
        except Exception as e:
            logger.error(f"Failed to flush translation cache to BigQuery: {e}")


def get_bq_writer():
    """Get or create BigQuery writer thread."""
    global _bq_writer
    if _bq_writer is None:
        _bq_writer = BQWriterThread()
        _bq_writer.start()
    return _bq_writer


class TranslationCache:
    """
    Optimized BigQuery Cache:
    - Lazy loads all relevant cached items into an in-memory dict on first use (L1 Cache)
    - All get() calls are instant O(1) local dict lookups
    - All set() calls instantly update the local dict and queue the BigQuery insert (L2 Cache)
    """

    @staticmethod
    def _sort_dict_recursively(obj):
        """Recursively sort dictionary keys for consistent JSON serialization."""
        if isinstance(obj, dict):
            return {
                k: TranslationCache._sort_dict_recursively(v)
                for k in sorted(obj.keys())
                for v in [obj[k]]
            }
        elif isinstance(obj, list):
            return [TranslationCache._sort_dict_recursively(item) for item in obj]
        return obj

    def __init__(self, translate_engine: str, translate_engine_params: dict = None):
        self.translate_engine = translate_engine
        self._local_cache = {}
        self._loaded = False
        self._lock = threading.Lock()
        self.replace_params(translate_engine_params)

    def replace_params(self, params: dict = None):
        """Replace all engine parameters."""
        with self._lock:
            if params is None:
                params = {}
            self.params = params
            params = self._sort_dict_recursively(params)
            self.translate_engine_params = json.dumps(params)

            # Reset L1 cache because params changed
            self._loaded = False
            self._local_cache.clear()

    def update_params(self, params: dict = None):
        """Update engine parameters."""
        if params is None:
            params = {}
        self.params.update(params)
        self.replace_params(self.params)

    def add_params(self, k: str, v):
        """Add a single parameter."""
        self.params[k] = v
        self.replace_params(self.params)

    def _load_from_bq(self):  # noqa: S608
        """Lazy load translations for current engine+params from BigQuery into memory."""
        with self._lock:
            if self._loaded:
                return
            try:
                logger.info(
                    f"Loading translation cache from BigQuery for {self.translate_engine}..."
                )
                client = get_bq_client()

                query = f"""
                    SELECT original_text, translation 
                    FROM `{get_table_ref()}`
                    WHERE translate_engine = @engine 
                    AND translate_engine_params = @params
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter(
                            "engine", "STRING", self.translate_engine
                        ),
                        bigquery.ScalarQueryParameter(
                            "params", "STRING", self.translate_engine_params
                        ),
                    ]
                )
                query_job = client.query(query, job_config=job_config)
                rows = query_job.result()

                for row in rows:
                    self._local_cache[row.original_text] = row.translation

                logger.info(
                    f"Loaded {len(self._local_cache)} cache entries from BigQuery."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load cache from BigQuery (might be empty/missing): {e}"
                )
            finally:
                # Mark as loaded even if it failed, so we don't spam BigQuery on every miss
                self._loaded = True

    def get(self, original_text: str) -> str | None:
        """Get translation from cache."""
        if not self._loaded:
            self._load_from_bq()
        return self._local_cache.get(original_text)

    def set(self, original_text: str, translation: str):
        """Set translation in cache."""
        if not self._loaded:
            self._load_from_bq()

        # 1. Update fast in-memory L1 cache immediately
        self._local_cache[original_text] = translation

        # 2. Queue for async BigQuery streaming insert
        writer = get_bq_writer()
        writer.queue.put(
            {
                "translate_engine": self.translate_engine,
                "translate_engine_params": self.translate_engine_params,
                "original_text": original_text,
                "translation": translation,
            }
        )


def init_db(remove_exists=False):
    """Initialize BigQuery dataset and table."""
    try:
        client = get_bq_client()
        project = client.project or BQ_PROJECT

        # Create Dataset
        dataset_id = f"{project}.{BQ_DATASET}"
        try:
            client.get_dataset(dataset_id)
        except NotFound:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"  # Modify if your bucket is elsewhere
            client.create_dataset(dataset, timeout=30)
            logger.info(f"Created BigQuery dataset: {dataset_id}")

        # Create Table
        table_id = get_table_ref()
        schema = [
            bigquery.SchemaField("translate_engine", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("translate_engine_params", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("original_text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("translation", "STRING", mode="REQUIRED"),
        ]

        if remove_exists:
            client.delete_table(table_id, not_found_ok=True)
            logger.info(f"Deleted BigQuery table: {table_id}")

        try:
            client.get_table(table_id)
        except NotFound:
            table = bigquery.Table(table_id, schema=schema)
            # Partitioning configuration
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="created_at"
                if "created_at" in [field.name for field in schema]
                else None,
            )
            # Clustering for performance
            table.clustering_fields = ["translate_engine", "translate_engine_params"]
            client.create_table(table)
            logger.info(f"Created BigQuery table: {table_id}")

    except Exception as e:
        logger.error(f"Failed to initialize BigQuery: {e}")
        raise
