"""Test script to submit documents to the live translation server and track usage."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from google.cloud import bigquery
from src.api.core.security import create_access_token
from src.config.constants import settings

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8002/api/v1")


def get_auth_token() -> str:
    return create_access_token(
        {
            "sub": "jabir.mohammed@colt.net",
            "business_unit": "Engineering",
            "organization": "Colt",
        }
    )


def submit_document(
    file_path: Path,
    target_languages: list[str],
    domain: str = "general",
) -> str:
    token = get_auth_token()
    headers = {"x-app-auth": f"Bearer {token}"}

    print("\n=======================================================")
    print(f"Submitting: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
    print(f"Target languages: {target_languages}, Domain: {domain}")
    print("=======================================================")

    with file_path.open("rb") as f:
        file_bytes = f.read()

    files = {"file": (file_path.name, file_bytes, "application/pdf")}
    data = {
        "domain": domain,
        "target_languages": target_languages,
        "enable_dlp": "false",
        "enable_chunking": "false",
        "priority": "standard",
    }

    resp = httpx.post(
        f"{API_BASE_URL}/translate",
        headers=headers,
        files=files,
        data=data,
        timeout=1800.0,
    )

    if resp.status_code not in (200, 202):
        print(f"Submission failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    result = resp.json()
    print("Submission response:", json.dumps(result, indent=2))
    job_ids = result.get("job_ids", [])
    if not job_ids and "jobs" in result:
        job_ids = [j["job_id"] for j in result["jobs"]]
    if not job_ids and "job_id" in result:
        job_ids = [result["job_id"]]

    job_id = job_ids[0]
    print(f"Job ID: {job_id}")
    return job_id


def poll_job(job_id: str, max_timeout_seconds: int = 1800) -> dict:
    token = get_auth_token()
    headers = {"x-app-auth": f"Bearer {token}"}
    start_time = time.time()
    last_status = None

    print(f"Waiting for job {job_id} to complete...")
    while time.time() - start_time < max_timeout_seconds:
        elapsed = int(time.time() - start_time)
        try:
            resp = httpx.get(
                f"{API_BASE_URL}/jobs/{job_id}",
                headers=headers,
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status != last_status:
                    print(f"[{elapsed}s] Job status: {status}")
                    last_status = status
                if status in ("completed", "failed", "cancelled"):
                    print(f"[{elapsed}s] Final status: {status}")
                    return data
        except Exception as e:
            print(f"[{elapsed}s] Polling error: {e}")

        time.sleep(5)

    print(f"Timed out after {max_timeout_seconds}s waiting for job {job_id}")
    return {}


def query_bigquery_usage(job_id: str, out_dir: Path | None = None):
    print(f"\n--- BigQuery Usage & Cost Tracking for {job_id} ---")
    report = {}
    try:
        client = bigquery.Client(project=settings.GOOGLE_CLOUD_PROJECT)
        dataset = settings.BIGQUERY_DATASET

        # Query translation_jobs
        jobs_query = f"""
            SELECT job_id, status, translation_config, cost_attribution, result,
                   submitted_at, completed_at
            FROM `{settings.GOOGLE_CLOUD_PROJECT}.{dataset}.{settings.BIGQUERY_TABLE}`
            WHERE job_id = '{job_id}'
        """  # noqa: S608
        job_rows = list(client.query(jobs_query).result())
        if job_rows:
            row = dict(job_rows[0].items())
            t_cfg = json.loads(row.get("translation_config") or "{}")
            cost_attr = json.loads(row.get("cost_attribution") or "{}")
            res_dict = json.loads(row.get("result") or "{}")
            report["job"] = {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in row.items()
            }
            print(f"Status:             {row.get('status')}")
            print(
                f"Language Pair:      {t_cfg.get('source_language')} -> {t_cfg.get('target_language')}"
            )
            print(f"Model Used:         {cost_attr.get('selected_model')}")
            print(f"Attempts:           {len(res_dict.get('attempts', []))}")
            print(
                f"Quality Score:      {res_dict.get('quality_report', {}).get('final_score')}"
            )
            print(f"Prompt Tokens:      {cost_attr.get('prompt_tokens')}")
            print(f"Completion Tokens:  {cost_attr.get('completion_tokens')}")
            print(f"Total Tokens:       {cost_attr.get('total_tokens')}")
            print(f"Total Cost (USD):   ${cost_attr.get('total_cost_usd', 0):.6f}")
        else:
            print("No BigQuery job record found yet.")

        # Query translation_costs for breakdown
        costs_query = f"""
            SELECT model_id, intent, input_tokens, output_tokens,
                   cost_usd, timestamp
            FROM `{settings.GOOGLE_CLOUD_PROJECT}.{dataset}.{settings.BIGQUERY_COST_TABLE}`
            WHERE job_id = '{job_id}'
        """  # noqa: S608
        cost_rows = list(client.query(costs_query).result())
        if cost_rows:
            report["costs"] = [dict(c.items()) for c in cost_rows]
            print(f"\nCost Breakdown ({len(cost_rows)} entries):")
            for c in cost_rows:
                cd = dict(c.items())
                print(
                    f"  - [{cd.get('intent')}] model={cd.get('model_id')} "
                    f"tokens(in/out)={cd.get('input_tokens')}/{cd.get('output_tokens')} "
                    f"cost=${cd.get('cost_usd', 0):.6f}"
                )

        if out_dir:
            out_file = out_dir / f"usage_{job_id}.json"
            out_file.write_text(json.dumps(report, indent=2, default=str))
            print(f"Saved usage report to {out_file}")
    except Exception as e:
        print(f"Error querying BigQuery: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=Path)
    parser.add_argument("--target-lang", default="de")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    job_id = submit_document(
        args.file_path,
        target_languages=[args.target_lang],
        domain=args.domain,
    )
    job_result = poll_job(job_id)
    print("\nJob Detail Result:")
    print(json.dumps(job_result, indent=2, default=str))

    if args.out_dir:
        (args.out_dir / f"job_result_{job_id}.json").write_text(
            json.dumps(job_result, indent=2, default=str)
        )

    query_bigquery_usage(job_id, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
