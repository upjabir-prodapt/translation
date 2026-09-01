import json
from pathlib import Path

from google.cloud import bigquery
from src.config.constants import settings

client = bigquery.Client(
    project=settings.GOOGLE_CLOUD_PROJECT, location=settings.BIGQUERY_LOCATION
)
dataset = settings.BIGQUERY_DATASET
job_id = "3411fac9-f681-4f72-aa5c-43258f2cfd4c"

jobs_query = f"""
    SELECT job_id, status, submitted_at, completed_at, result
    FROM `{settings.GOOGLE_CLOUD_PROJECT}.{dataset}.{settings.BIGQUERY_TABLE}`
    WHERE job_id = '{job_id}'
"""  # noqa: S608
job_rows = list(client.query(jobs_query).result())
for r in job_rows:
    row = dict(r.items())
    print("Status:", row.get("status"))
    print("Submitted:", row.get("submitted_at"))
    print("Completed:", row.get("completed_at"))
    res = json.loads(row.get("result", "{}"))
    print("Result structure:")
    print(json.dumps(res, indent=2))

print("\n--- Cost records ---")
costs_query = f"""
    SELECT *
    FROM `{settings.GOOGLE_CLOUD_PROJECT}.{dataset}.{settings.BIGQUERY_COST_TABLE}`
    WHERE job_id = '{job_id}'
"""  # noqa: S608
cost_rows = list(client.query(costs_query).result())
for c in cost_rows:
    row = dict(c.items())
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            row[k] = v.isoformat()
    print(json.dumps(row))

out_dir = Path(
    "/home/jabir_mohammed_colt_net/Translation/.local-tmp/e2e/live_test_run_promptengg"
)
out_file = out_dir / "usage_summary.json"
summary = {
    "job_id": job_id,
    "job_rows": [
        {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in dict(r.items()).items()
        }
        for r in job_rows
    ],
    "cost_rows": [
        {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in dict(c.items()).items()
        }
        for c in cost_rows
    ],
}
out_file.write_text(json.dumps(summary, indent=2, default=str))
print(f"\nWrote usage summary to {out_file}")
