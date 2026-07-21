"""Cutover / rollback notes for API + Worker Cloud Run split.

## Prerequisites
1. Create Cloud Tasks queue:
   PROJECT=... REGION=... ./scripts/create_cloud_tasks_queue.sh
2. Create two Secret Manager secrets from templates:
   - translation-api-env    ← .env.example.api
   - translation-worker-env ← .env.example.worker
3. IAM:
   - API SA: roles/cloudtasks.enqueuer on the queue
   - API SA: iam.serviceAccountUser on CLOUD_TASKS_OIDC_SERVICE_ACCOUNT
   - OIDC SA: roles/run.invoker on the worker Cloud Run service
   - Worker SA: existing GCS / BQ / Vertex / DLP roles

## Cutover sequence
1. Deploy **worker** (min-instances≥1, FUSE mount, APP_ROLE=worker).
2. Smoke: authenticated POST /internal/tasks/translate with a queued job_id (or OIDC test token).
3. Deploy **api** (no FUSE, API_USE_BACKGROUND_PIPELINE=false, Cloud Tasks env).
4. Point IAP / ILB / DNS at the new API service; keep old monolith running.
5. Drain: wait for in-flight monolith jobs to finish.
6. Scale down / delete the old single Cloud Run service.

## Rollback
- Prefer: traffic back to the previous **monolith** Cloud Run revision/image.
- Do **not** expect the new API image to run in-process translation
  (API_USE_BACKGROUND_PIPELINE=false and worker deps are not in the API image).
- If only the worker is broken: keep API, fix worker; queued jobs will retry via Tasks.

## Local development
- Combined: API_USE_BACKGROUND_PIPELINE=true and `uv sync --extra worker`
  (in-process pipeline; single process).
- Production-like: run `python main_api.py` + `python main_worker.py` with Tasks
  (or HTTP to worker) and API_USE_BACKGROUND_PIPELINE=false.
