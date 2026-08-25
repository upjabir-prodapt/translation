# Local Testing Guide — Running the API + Worker Servers Yourself

This guide explains how to spin up the API and Worker servers locally
against the **aicoesandox** GCP project, submit a real translation job
(PDF or DOCX), and find every file/credential/cookie produced along the way.

---

## 1. Prerequisites

- `gcloud` CLI authenticated as a user/service-account with access to the
  `aicoeprod` Secret Manager secrets and the `aicoesandox` project's GCS
  bucket + BigQuery dataset.
  ```bash
  gcloud auth login
  gcloud auth application-default login
  ```
- `uv` installed (already present in this environment at
  `~/.local/bin/uv`).
- Repo dependencies installed:
  ```bash
  cd /home/jabir_mohammed_colt_net/Translation
  uv sync --extra worker --group dev
  ```

---

### For step 2, 3 it is already exist. so no need to do anything if it exists

## 2. Build local `.env` files from Secret Manager

The real Cloud Run secrets (`translation-service-env` for the API,
`translation-worker-env` for the worker) live in the **aicoeprod** Secret
Manager. For local testing we pull them down and rewrite every
`aicoeprod` reference to `aicoesandox` (a real, working GCP project with
the same GCS bucket layout and BigQuery schema).

```bash
cd /home/jabir_mohammed_colt_net/Translation

# 1. Fetch the two secrets
gcloud secrets versions access latest --secret=translation-service-env --project=aicoeprod > /tmp/api_secret.env
gcloud secrets versions access latest --secret=translation-worker-env --project=aicoeprod > /tmp/worker_secret.env

# 2. Rewrite aicoeprod -> aicoesandox and adjust for local/dev
cp /tmp/api_secret.env .env.api.local
cp /tmp/worker_secret.env .env.worker.local
sed -i 's/aicoeprod/aicoesandox/g' .env.api.local .env.worker.local
sed -i 's/^IS_LOCAL=false/IS_LOCAL=true/' .env.api.local .env.worker.local

# 3. API-specific local overrides: run in Cloud-Tasks-bypass mode,
#    pointing directly at your local worker instead of real Cloud Tasks
sed -i 's/^API_USE_BACKGROUND_PIPELINE=true/API_USE_BACKGROUND_PIPELINE=false/' .env.api.local
sed -i '/^APP_ROLE/d' .env.api.local
cat >> .env.api.local << 'EOF'

# --- Cloud Tasks (local dev: bypasses real Cloud Tasks, POSTs directly to worker) ---
CLOUD_TASKS_PROJECT=aicoesandox
CLOUD_TASKS_LOCATION=europe-west1
CLOUD_TASKS_QUEUE=translation-jobs
CLOUD_TASKS_WORKER_URL=http://localhost:8001/internal/tasks/translate
CLOUD_TASKS_OIDC_SERVICE_ACCOUNT=
CLOUD_TASKS_DISPATCH_DEADLINE_SECONDS=3600
WORKER_OIDC_AUDIENCE=
WORKER_SKIP_OIDC_VERIFICATION=true
APP_ROLE=
EOF

# 4. Worker-specific local overrides: same worker URL, OIDC bypass
sed -i 's|^CLOUD_TASKS_WORKER_URL=.*|CLOUD_TASKS_WORKER_URL=http://localhost:8001/internal/tasks/translate|' .env.worker.local
sed -i 's/^WORKER_SKIP_OIDC_VERIFICATION=.*/WORKER_SKIP_OIDC_VERIFICATION=true/' .env.worker.local

# 5. Point assets/temp storage at an in-repo directory (see Section 3)
REPO_TMP="$(pwd)/.local-tmp"
sed -i "s|^ASSETS_ROOT=.*|ASSETS_ROOT=${REPO_TMP}/assets-cache|" .env.api.local .env.worker.local
sed -i "s|^TEMP_DIR=.*|TEMP_DIR=${REPO_TMP}/tmp|" .env.api.local .env.worker.local

# 6. Local Redis for the LLM translation cache (see "Local Redis" section
#    below for how to start it). Points at a plain local container instead
#    of the real Memorystore/PSC endpoint used in Cloud Run.
sed -i '/^REDIS_HOST=/d;/^REDIS_PORT=/d;/^REDIS_TLS_ENABLED=/d;/^REDIS_PASSWORD=/d' .env.worker.local
cat >> .env.worker.local << 'EOF'
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TLS_ENABLED=false
REDIS_PASSWORD=
EOF
```

### Local Redis (LLM translation cache)

The worker's per-text-batch LLM translation cache
(`src/worker/doctranslator/translator/translation_cache.py`) is backed
by Redis in every environment, including local dev. Start a throwaway
local Redis container:

```bash
docker run -d --name translation-redis -p 6379:6379 redis:7-alpine
```

If Docker isn't available in your environment, either install/run
`redis-server` directly, or point `REDIS_HOST`/`REDIS_PORT` at a real
dev Memorystore instance reachable through an SSH tunnel/bastion (ask
infra for a PSC-reachable dev instance — see
`docs/infra/redis-memorystore-psc-setup.md`). If `REDIS_HOST` is left
empty, the worker simply disables the cache and falls through to
calling the LLM directly every time — translation still works, just
without cache hits.

**Files created:** `.env.api.local` and `.env.worker.local` in the repo
root. Both match the `.env.*.local` pattern already in `.gitignore`, so
they will never be committed.

---

## 3. Where files are stored locally ( check if it alreday exists or not, if it exist donot modify it)

Everything is kept inside the repo under `.local-tmp/` so it's easy to
find, inspect, and clean up (also gitignored via the `.local-tmp/` entry
in `.gitignore`). PLease check

```
.local-tmp/
├── assets-cache/          # ASSETS_ROOT — downloaded model/glossary/font assets
│   ├── models/            # ONNX layout model (PDF pipeline only)
│   ├── glossaries/        # cached domain glossary JSON files
│   ├── metadata/          # font/cmap metadata indexes
│   ├── fonts/, cmap/, tiktoken/
│   └── model_selection.json   # model routing table (see Section 5 note)
├── tmp/                   # TEMP_DIR — per-job scratch workspace
│   └── jobs/
│       ├── <job_id>/
│       │   ├── input/     # downloaded source document
│       │   ├── attempts/  # per-model-attempt translated output (before GCS upload)
│       │   ├── masked/, verification/, assembly/, final/, logs/
│       └── _shared_prep/  # shared download/prep cache for multi-language batches
├── worker_server.log       # worker process stdout/stderr
├── worker.pid              # worker process PID
├── api_server.log          # API process stdout/stderr
├── api.pid                 # API process PID
└── cookies.txt             # curl cookie jar holding your session cookie (see Section 5)
```

> **Note:** `TempWorkspaceService.cleanup()` deletes each job's
> `.local-tmp/tmp/jobs/<job_id>/` directory automatically once the
> pipeline finishes (success or failure). If you want to inspect
> intermediate files, add a `time.sleep()` breakpoint or copy the
> directory out mid-run.

---

## 4. Start the two servers

Open two terminals (or run both in the background as shown):

```bash
cd /home/jabir_mohammed_colt_net/Translation

# Terminal 1 — Worker (port 8001)
DOTENV_PATH=.env.worker.local PORT=8001 uv run python main_worker.py

# Terminal 2 — API (port 8000)
DOTENV_PATH=.env.api.local PORT=8000 uv run python main_api.py
```

Or run both in the background and tail their logs:

```bash
mkdir -p .local-tmp
DOTENV_PATH=.env.worker.local PORT=8001 nohup uv run python main_worker.py > .local-tmp/worker_server.log 2>&1 &
echo $! > .local-tmp/worker.pid

DOTENV_PATH=.env.api.local PORT=8000 nohup uv run python main_api.py > .local-tmp/api_server.log 2>&1 &
echo $! > .local-tmp/api.pid

tail -f .local-tmp/worker_server.log   # in one terminal
tail -f .local-tmp/api_server.log      # in another
```

**Health check:**
```bash
curl -s http://localhost:8000/docs -o /dev/null -w "API: %{http_code}\n"
curl -s http://localhost:8001/       # worker root -> {"service":"translation-worker",...}
```

**Stop the servers:**
```bash
kill "$(cat .local-tmp/api.pid)" "$(cat .local-tmp/worker.pid)"
```

---

## 5. One critical manual step: `model_selection.json`

The worker's `STARTUP_WARMUP_ENABLED=false` in the fetched secrets means
the worker does **not** auto-download `model_selection.json` (the
language-pair → LLM model routing table) at startup. Without it, every
job fails immediately with:
```
Configuration file not found: /path/assets-cache/model_selection.json
```

**Fix — download it once manually before submitting jobs:**
```bash
gsutil cat gs://aicoesandox-vxai-translation-app-001/assets/model_selection.json \
  > .local-tmp/assets-cache/model_selection.json
```
(No restart needed — the worker reads this file fresh on each request.)

---

## 6. Authenticate locally (no real IAP available)

Since `IS_LOCAL=true`, the app exposes a **local-dev IAP bypass**: instead
of a real Google IAP JWT, you set two headers with your identity and
group membership directly. Find your `TRANSLATION_REQUIRED_GROUP` value:

```bash
grep "^TRANSLATION_REQUIRED_GROUP" .env.api.local
# TRANSLATION_REQUIRED_GROUP=98be0302-982b-465c-a633-baf4f8ef2da0
```

Then request a session token/cookie:

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "x-dev-iap-user-email: your.name@colt.net" \
  -H "x-dev-iap-user-groups: 98be0302-982b-465c-a633-baf4f8ef2da0" \
  -H "Content-Type: application/json" \
  -d '{"business_unit":"AI-COE","organization":"Colt"}' \
  -c .local-tmp/cookies.txt
```

**Where the credential is stored:** the response sets an httpOnly session
cookie named `colt_session` (a self-signed JWT, `JWT_SECRET_KEY` from the
`.env.*.local` file), saved by curl's `-c` flag into
**`.local-tmp/cookies.txt`**. Every subsequent request just needs
`-b .local-tmp/cookies.txt` to reuse it (default expiry:
`JWT_ACCESS_TOKEN_EXPIRE_MINUTES`, typically 60 minutes — re-run the
`auth/token` call to get a fresh one if it expires).

---

## 7. Submit a translation job

Use one of the two sample files already in this repo under
`docs/test docs/` (copy them somewhere without spaces in the path first,
since some shells/tools mishandle spaces in multipart file uploads):

```bash
mkdir -p .local-tmp/testfiles
cp "docs/test docs/Colt_AICOE_Azure_SSO_IAP_WIF_Implementation_Guide_v1.0_1.docx" .local-tmp/testfiles/sample.docx
cp "docs/test docs/Research_Report_Microsoft.pdf" .local-tmp/testfiles/sample.pdf
```

**Submit a DOCX job (English → French):**
```bash
curl -s -X POST http://localhost:8000/api/v1/translate \
  -b .local-tmp/cookies.txt \
  -F "file=@.local-tmp/testfiles/sample.docx" \
  -F "domain=commercial" \
  -F "target_languages=French" \
  -F "source_language=English"
```
Response:
```json
{"batch_id":"...","jobs":[{"job_id":"<JOB_ID>","target_language":"fr","status":"queued","status_url":"/api/v1/translate/<JOB_ID>"}]}
```

**Submit a PDF job** — identical command, just point `-F "file=@..."` at
`sample.pdf` instead.

**Multiple target languages in one call** (exercises the
`shared_document_prep` optimization — see
`docs/architecture/pdf-vs-docx-translation-architecture.md`):
```bash
curl -s -X POST http://localhost:8000/api/v1/translate \
  -b .local-tmp/cookies.txt \
  -F "file=@.local-tmp/testfiles/sample.docx" \
  -F "domain=commercial" \
  -F "target_languages=French" \
  -F "target_languages=German" \
  -F "source_language=English"
```

---

## 8. Poll job status

```bash
curl -s -b .local-tmp/cookies.txt http://localhost:8000/api/v1/translate/<JOB_ID>
```
```json
{"job_id":"...","status":"completed","submitted_at":"...","completed_at":"...",
 "result":{"output_gcs_uri":"gs://.../output/....docx","cost_usd":0.36,"confidence_score":0.69,...}}
```

Statuses progress: `queued` → `processing` → `completed` | `failed`.
A single-language DOCX/PDF translation typically takes **1–5 minutes**
depending on document size and how many model-attempt retries the
quality judge triggers.

While it's running, watch progress live:
```bash
tail -f .local-tmp/worker_server.log
```

---

## 9. Download and inspect the translated output

The `result.output_gcs_uri` field gives you the exact GCS path:

```bash
gsutil cp "gs://aicoesandox-vxai-translation-app-001/translation-service/<JOB_ID>/output/<FILENAME>" \
  .local-tmp/translated_output.docx
```

**Verify the DOCX content with python-docx:**
```bash
uv run python -c "
from docx import Document
d = Document('.local-tmp/translated_output.docx')
paras = [p.text for p in d.paragraphs if p.text.strip()]
print('Paragraphs:', len(paras))
for p in paras[:15]:
    print('-', p[:120])
"
```

**Verify a translated PDF:** open it directly, or extract text with
`pymupdf`:
```bash
uv run python -c "
import pymupdf
doc = pymupdf.open('.local-tmp/translated_output.pdf')
print(doc[0].get_text()[:1000])
"
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Configuration file not found: .../model_selection.json` | Startup warmup disabled | Manually download it (Section 5) |
| `401 Not authenticated` on `/translate` | Missing/expired session cookie | Re-run the `auth/token` curl command (Section 6) |
| `403 You do not have access` | Wrong `x-dev-iap-user-groups` value | Confirm `TRANSLATION_REQUIRED_GROUP` in `.env.api.local` matches the header |
| `address already in use` on port 8000/8001 | A previous server instance is still running | `lsof -i :8000` / `:8001`, then `kill <pid>` |
| Job stuck in `processing` for a long time | Normal for large documents — many small LLM batch/fallback calls | Check `worker_server.log` for active `do_llm_translate`/`do_translate` lines; it is progressing, not hung |
| `PermissionDenied` on GCS/BigQuery calls | ADC not authenticated for `aicoesandox` | Re-run `gcloud auth application-default login` |

---

## 11. Cleaning up

```bash
kill "$(cat .local-tmp/api.pid)" "$(cat .local-tmp/worker.pid)" 2>/dev/null
rm -rf .local-tmp
```
`.env.api.local` / `.env.worker.local` are safe to leave in place for
next time — regenerate them from Secret Manager again if the underlying
secrets rotate.
