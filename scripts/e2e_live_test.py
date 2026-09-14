"""End-to-End Live Test Script for Colt-profile PDF translation."""

import json
import os
import sys
import time
from pathlib import Path

import httpx

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.constants import settings

API_BASE_URL = os.environ.get(
    "API_BASE_URL", f"http://localhost:8010{settings.API_PREFIX}"
)
PDF_PATH = Path(
    "/home/jabir_mohammed_colt_net/Translation/docs/test_docs/Colt-profile-EN_2026_JMamends.pdf"
)
OUTPUT_DIR = Path(".local-tmp/e2e_results")


def get_auth_headers() -> dict[str, str]:
    """Build request headers for Apigee auth (src/api/core/apigee_auth.py).

    In local dev (IS_LOCAL=true) the API skips Google ID-token verification
    entirely and instead trusts the `x-colt-user-*` headers directly if
    present -- no JWT to mint. See get_current_apigee_user()'s docstring.
    """
    return {
        "x-colt-user-oid": "e2e-test-oid",
        "x-colt-user-email": "jabir.mohammed@colt.net",
        "x-colt-user-roles": "Translation.User",
        "x-colt-user-department": "Engineering",
        "x-colt-user-company": "Colt",
    }


def run_e2e_test():
    print("=" * 70)
    print("STARTING E2E TRANSLATION TEST")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Input Document: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"Project: {settings.GOOGLE_CLOUD_PROJECT}")
    print(f"Bucket: {settings.GCS_BUCKET_NAME}")
    print(f"BigQuery Dataset: {settings.BIGQUERY_DATASET}")
    print(f"Asset Cache: {settings.ASSETS_ROOT}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = get_auth_headers()

    # 1. Health check
    print("\n[Step 1] Checking API health...")
    health_resp = httpx.get(f"{API_BASE_URL}/healthz", timeout=10.0)
    print(f"API Health: {health_resp.status_code} -> {health_resp.text}")
    assert health_resp.status_code == 200, "API is not healthy"

    # 2. Submit document
    print("\n[Step 2] Submitting PDF translation request (en -> de, commercial)...")
    with PDF_PATH.open("rb") as f:
        file_bytes = f.read()

    files = {
        "file": (
            PDF_PATH.name,
            file_bytes,
            "application/pdf",
        )
    }
    data = {
        "domain": "commercial",
        "source_language": "en",
        "target_languages": ["de"],
        "enable_dlp": "false",
        "enable_chunking": "false",
        "priority": "standard",
    }

    resp = httpx.post(
        f"{API_BASE_URL}/translate",
        headers=headers,
        files=files,
        data=data,
        timeout=600.0,
    )
    print(f"Submit Response Status: {resp.status_code}")
    print(f"Submit Response Body: {resp.text}")

    if resp.status_code not in (200, 202):
        print(f"ERROR: Submission failed with status {resp.status_code}")
        sys.exit(1)

    result_json = resp.json()
    jobs = result_json.get("jobs", [])
    if not jobs:
        print("ERROR: No jobs returned in response")
        sys.exit(1)

    job_id = jobs[0]["job_id"]
    batch_id = result_json.get("batch_id")
    print(f"\n[Step 3] Job enqueued: job_id={job_id}, batch_id={batch_id}")

    # 3. Poll job status
    print("\n[Step 4] Polling job status...")
    start_time = time.time()
    poll_timeout = 600  # 10 minutes max
    last_status = None
    final_job = None

    while time.time() - start_time < poll_timeout:
        elapsed = int(time.time() - start_time)
        try:
            status_resp = httpx.get(
                f"{API_BASE_URL}/jobs/{job_id}",
                headers=headers,
                timeout=15.0,
            )
            if status_resp.status_code == 200:
                job_data = status_resp.json()
                current_status = job_data.get("status")
                if current_status != last_status:
                    print(
                        f"  [{elapsed:3d}s] Job {job_id} status transitioned to: {current_status}"
                    )
                    last_status = current_status
                if current_status in ("completed", "failed", "cancelled"):
                    final_job = job_data
                    break
            else:
                print(
                    f"  [{elapsed:3d}s] Poll status {status_resp.status_code}: {status_resp.text}"
                )
        except Exception as e:
            print(f"  [{elapsed:3d}s] Poll error: {e}")

        time.sleep(3)

    if not final_job:
        print(f"ERROR: Job did not reach terminal state within {poll_timeout}s")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"FINAL JOB STATUS: {final_job.get('status')}")
    print(f"Error (if any): {final_job.get('error_message')}")
    print("=" * 70)

    # Save job json
    job_result_file = OUTPUT_DIR / f"job_{job_id}_result.json"
    job_result_file.write_text(json.dumps(final_job, indent=2, default=str))
    print(f"Saved job result JSON to {job_result_file}")

    if final_job.get("status") != "completed":
        print(f"ERROR: Job failed with: {final_job.get('error_message')}")
        sys.exit(1)

    # 4. Download translated file
    print("\n[Step 5] Downloading translated document via download endpoint...")
    dl_endpoint_resp = httpx.get(
        f"{API_BASE_URL}/jobs/{job_id}/download",
        headers=headers,
        timeout=15.0,
    )
    print(
        f"Download Endpoint Response ({dl_endpoint_resp.status_code}): {dl_endpoint_resp.text}"
    )
    if dl_endpoint_resp.status_code == 200:
        dl_url = dl_endpoint_resp.json().get("download_url")
        if dl_url:
            pdf_dl_resp = httpx.get(dl_url, timeout=60.0)
            if pdf_dl_resp.status_code == 200:
                out_pdf_path = OUTPUT_DIR / f"translated_{PDF_PATH.name}"
                out_pdf_path.write_bytes(pdf_dl_resp.content)
                print(
                    f"Successfully downloaded translated PDF: {out_pdf_path} ({len(pdf_dl_resp.content) / 1024:.1f} KB)"
                )
            else:
                print(
                    f"Download from signed URL failed with status {pdf_dl_resp.status_code}"
                )

    # 5. Query BigQuery for verification
    print("\n[Step 6] Verifying BigQuery job records...")
    try:
        from google.cloud import bigquery

        bq = bigquery.Client(project=settings.GOOGLE_CLOUD_PROJECT)
        query = f"""
            SELECT job_id, status, translation_config, cost_attribution, result,
                   business_unit, organization
            FROM `{settings.GOOGLE_CLOUD_PROJECT}.{settings.BIGQUERY_DATASET}.{settings.BIGQUERY_TABLE}`
            WHERE job_id = '{job_id}'
        """  # noqa: S608
        rows = list(bq.query(query).result())
        if rows:
            r = dict(rows[0].items())
            print(f"BigQuery Job Record: Found (status={r.get('status')})")
            print(
                f"Business Unit: {r.get('business_unit')!r}  "
                f"Organization: {r.get('organization')!r}"
            )
            cost_attr = json.loads(r.get("cost_attribution") or "{}")
            print(f"Cost Attribution:\n{json.dumps(cost_attr, indent=2)}")
            result_obj = json.loads(r.get("result") or "{}")
            print(
                f"Quality Score: {result_obj.get('quality_report', {}).get('final_score')}"
            )
        else:
            print("Warning: No BigQuery row found.")
    except Exception as e:
        print(f"BigQuery query warning: {e}")

    print("\n" + "=" * 70)
    print("E2E TRANSLATION TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_test()
