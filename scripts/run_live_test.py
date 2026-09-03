"""Live E2E test script for input consistency guards against running API/Worker servers."""

import json
import os
import sys
import time
from pathlib import Path

import httpx

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.core.security import create_access_token

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8002/api/v1")
LOG_OUTPUT_FILE = Path(".local-tmp/logs/live_test_results.json")


def get_auth_token() -> str:
    return create_access_token(
        {
            "sub": "jabir.mohammed@colt.net",
            "email": "jabir.mohammed@colt.net",
            "business_unit": "Engineering",
            "organization": "Colt",
            "scopes": ["translation"],
        }
    )


def submit_translation(
    file_path: Path,
    target_languages: list[str],
    source_language: str | None = None,
    domain: str = "operations",
    enable_dlp: bool = False,
) -> tuple[int, dict]:
    token = get_auth_token()
    headers = {"x-app-auth": f"Bearer {token}"}

    with file_path.open("rb") as f:
        file_bytes = f.read()

    files = {
        "file": (
            file_path.name,
            file_bytes,
            "application/pdf"
            if file_path.suffix == ".pdf"
            else "application/octet-stream",
        )
    }
    data = {
        "domain": domain,
        "target_languages": target_languages,
        "enable_dlp": str(enable_dlp).lower(),
        "enable_chunking": "false",
        "priority": "standard",
    }
    if source_language is not None:
        data["source_language"] = source_language

    resp = httpx.post(
        f"{API_BASE_URL}/translate",
        headers=headers,
        files=files,
        data=data,
        timeout=180.0,
    )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    return resp.status_code, body


def poll_job_status(job_id: str, timeout_seconds: int = 120) -> dict:
    token = get_auth_token()
    headers = {"x-app-auth": f"Bearer {token}"}
    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout_seconds:
        elapsed = int(time.time() - start_time)
        resp = httpx.get(
            f"{API_BASE_URL}/translate/{job_id}",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status")
            if status != last_status:
                print(f"  [{elapsed}s] Job {job_id} status: {status}")
                last_status = status
            if status in ("completed", "failed", "cancelled"):
                return data
        else:
            print(f"  [{elapsed}s] Poll failed ({resp.status_code}): {resp.text}")
        time.sleep(2)

    raise TimeoutError(
        f"Job {job_id} did not reach terminal status within {timeout_seconds}s"
    )


def main():
    print("=" * 60)
    print("LIVE E2E TEST: Translation API & Input Consistency Guards")
    print(f"Target API: {API_BASE_URL}")
    print("=" * 60)

    results = []

    # Verify API health first
    try:
        health_resp = httpx.get("http://localhost:8002/api/v1/healthz", timeout=5.0)
        print(f"API Health check: {health_resp.status_code} -> {health_resp.text}")
        assert health_resp.status_code == 200
    except Exception as e:
        print(f"FAILED to connect to API: {e}")
        sys.exit(1)

    # Test file paths
    pdf_small = Path(".local-tmp/testfiles/dlp_test.pdf")
    pdf_hr = Path("docs/test_docs/HR Policy Manual 2023.pdf")
    docx_sample = Path(".local-tmp/testfiles/sample.docx")

    # -------------------------------------------------------------
    # Scenario 1: Missing source_language (API rejects with 422)
    # -------------------------------------------------------------
    print("\n--- Scenario 1: Missing source_language (API validation) ---")
    status_code, body = submit_translation(
        pdf_small,
        target_languages=["fr"],
        source_language=None,
        domain="operations",
    )
    print(f"HTTP Status: {status_code}")
    print(f"Response: {json.dumps(body, indent=2)}")
    passed = status_code == 422
    print(f"Scenario 1 Result: {'PASS' if passed else 'FAIL'}")
    results.append(
        {
            "scenario": "1_missing_source_language",
            "status_code": status_code,
            "passed": passed,
            "response": body,
        }
    )

    # -------------------------------------------------------------
    # Scenario 2: source_language == target_language (API rejects with 422)
    # -------------------------------------------------------------
    print("\n--- Scenario 2: source_language == target_language (API validation) ---")
    status_code, body = submit_translation(
        pdf_small,
        target_languages=["de"],
        source_language="de",
        domain="operations",
    )
    print(f"HTTP Status: {status_code}")
    print(f"Response: {json.dumps(body, indent=2)}")
    passed = status_code == 422
    print(f"Scenario 2 Result: {'PASS' if passed else 'FAIL'}")
    results.append(
        {
            "scenario": "2_source_equals_target",
            "status_code": status_code,
            "passed": passed,
            "response": body,
        }
    )

    # -------------------------------------------------------------
    # Scenario 3: Language Mismatch Guard (Worker fails job)
    # -------------------------------------------------------------
    print("\n--- Scenario 3: Language Mismatch Guard (Worker execution) ---")
    print(
        f"Submitting English PDF {pdf_small.name} declaring source_language='German' (de)..."
    )
    status_code, body = submit_translation(
        pdf_small,
        target_languages=["fr"],
        source_language="de",
        domain="operations",
    )
    print(f"Submit HTTP Status: {status_code}")
    if status_code == 202 and "jobs" in body and body["jobs"]:
        job_id = body["jobs"][0]["job_id"]
        print(f"Job enqueued: {job_id}. Polling worker execution...")
        job_result = poll_job_status(job_id, timeout_seconds=60)
        error_msg = job_result.get("error_message") or ""
        print(f"Final status: {job_result.get('status')}")
        print(f"Error message: {error_msg}")
        passed = (
            job_result.get("status") == "failed"
            and "German" in error_msg
            and "English" in error_msg
        )
        print(f"Scenario 3 Result: {'PASS' if passed else 'FAIL'}")
        results.append(
            {
                "scenario": "3_language_mismatch_guard",
                "job_id": job_id,
                "final_status": job_result.get("status"),
                "error_message": error_msg,
                "passed": passed,
            }
        )
    else:
        print(f"Submission failed: {body}")
        results.append(
            {"scenario": "3_language_mismatch_guard", "passed": False, "response": body}
        )

    # -------------------------------------------------------------
    # Scenario 4: Domain Mismatch Guard (Worker fails job)
    # -------------------------------------------------------------
    print("\n--- Scenario 4: Domain Mismatch Guard (Worker execution) ---")
    print(
        f"Submitting HR Policy document {pdf_hr.name} declaring domain='commercial'..."
    )
    status_code, body = submit_translation(
        pdf_hr,
        target_languages=["fr"],
        source_language="en",
        domain="commercial",
    )
    print(f"Submit HTTP Status: {status_code}")
    if status_code == 202 and "jobs" in body and body["jobs"]:
        job_id = body["jobs"][0]["job_id"]
        print(f"Job enqueued: {job_id}. Polling worker execution...")
        job_result = poll_job_status(job_id, timeout_seconds=90)
        error_msg = job_result.get("error_message") or ""
        print(f"Final status: {job_result.get('status')}")
        print(f"Error message: {error_msg}")
        passed = (
            job_result.get("status") == "failed"
            and "commercial" in error_msg
            and "hr" in error_msg
        )
        print(f"Scenario 4 Result: {'PASS' if passed else 'FAIL'}")
        results.append(
            {
                "scenario": "4_domain_mismatch_guard",
                "job_id": job_id,
                "final_status": job_result.get("status"),
                "error_message": error_msg,
                "passed": passed,
            }
        )
    else:
        print(f"Submission failed: {body}")
        results.append(
            {"scenario": "4_domain_mismatch_guard", "passed": False, "response": body}
        )

    # -------------------------------------------------------------
    # Scenario 5: Multi-target Batch (matching source language & domain)
    # -------------------------------------------------------------
    print("\n--- Scenario 5: Multi-target Batch with Matching Language & Domain ---")
    print(
        f"Submitting {pdf_small.name} declaring source_language='en', domain='operations', targets=['fr', 'de']..."
    )
    status_code, body = submit_translation(
        pdf_small,
        target_languages=["fr", "de"],
        source_language="en",
        domain="operations",
    )
    print(f"Submit HTTP Status: {status_code}")
    print(f"Batch response: {json.dumps(body, indent=2)}")
    if status_code == 202 and "jobs" in body and len(body["jobs"]) == 2:
        jobs = body["jobs"]
        batch_passed = True
        batch_details = []
        for j in jobs:
            jid = j["job_id"]
            tgt = j["target_language"]
            print(f"Polling job {jid} (target={tgt})...")
            jres = poll_job_status(jid, timeout_seconds=90)
            status = jres.get("status")
            err = jres.get("error_message")
            print(f"  -> Job {jid} ({tgt}) reached status: {status} (error: {err})")
            batch_details.append(
                {"job_id": jid, "target": tgt, "status": status, "error": err}
            )
            # It should either complete or proceed past consistency guards without mismatch error
            if err and ("You selected" in err or "more than one language" in err):
                batch_passed = False
        print(f"Scenario 5 Result: {'PASS' if batch_passed else 'FAIL'}")
        results.append(
            {
                "scenario": "5_multitarget_matching",
                "passed": batch_passed,
                "batch_jobs": batch_details,
            }
        )
    else:
        print(f"Submission failed: {body}")
        results.append(
            {"scenario": "5_multitarget_matching", "passed": False, "response": body}
        )

    # Save results to file
    LOG_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_OUTPUT_FILE.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll live test results written to {LOG_OUTPUT_FILE}")

    all_passed = all(r.get("passed", False) for r in results)
    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
    for r in results:
        print(f"  * {r['scenario']}: {'PASS' if r.get('passed') else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
