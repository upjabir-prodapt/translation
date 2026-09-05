# GitLab CI/CD Variables & Deployment Configuration Guide

This document details the GitLab CI/CD variables and Secret Manager payload keys for deploying
`translation-api` and `translation-worker` into **`gclt-aicoe-dev-st`**, scoped to the **`dev`**
GitLab environment, for the Apigee/AI-Hub integration built on the `gitlab-sync-branch-shared-dev`
branch. Modelled on
[`aihub-ui`'s `GITLAB_CI_VARIABLES.md`](../shared_ui/aihub-ui/GITLAB_CI_VARIABLES.md).

**Read this before touching either mechanism below — mixing them is the most common mistake in
this file:**

- **Section 2** = GitLab CI/CD pipeline variables (Settings → CI/CD → Variables). The `.gitlab-ci.yml`
  *script* reads these with `${VAR}` to build `gcloud` commands — project ids, regions, network
  names, service-account emails the pipeline itself must know to run `gcloud run deploy`.
- **Section 3** = keys inside the `/secrets/.env` payload — a Secret Manager secret's *content*,
  mounted into the container at `/secrets/.env` via `--set-secrets`. The **application** reads
  these at boot, not the pipeline. `API_PREFIX`, `APIGEE_RUNTIME_SA_EMAIL`,
  `CLOUD_RUN_SERVICE_URL` and `LLM_GATEWAY_*` all belong here — they are **not** GitLab CI/CD
  variables and must **not** be added to `--set-env-vars` or to Section 2. Every credential in
  Section 3 lives only in Secret Manager, never in Git.

---

## 1. Summary

| Category | Mechanism | Notes |
|---|---|---|
| WIF & CI runner identity | GitLab CI/CD variable | Shared across all 8 platform repos |
| GCP project / region / registry | GitLab CI/CD variable | `gclt-aicoe-dev-st`, `europe-west3` |
| Cloud Run network / per-service SA | GitLab CI/CD variable | `CLOUD_RUN_API_SA` / `CLOUD_RUN_WORKER_SA`, new — see WS-F1 |
| Cloud Tasks queue / OIDC identity | GitLab CI/CD variable | `translation-jobs`, `worker-invoker-sa` |
| App runtime config (prefix, Apigee trust, LLM gateway) | `/secrets/.env` payload | New in this pass — WS-D/WS-E settings |
| Pipeline access tokens | GitLab CI/CD variable (masked, protected) | Existing, unchanged |

---

## 2. GitLab CI/CD Variables (scope `dev`)

### A. Workload Identity Federation & Authentication

| Variable | Value | Source |
|---|---|---|
| `WORKLOAD_IDENTITY_PROJECT_NUMBER` | `538669417284` | `aicoe-sharedwif` project number |
| `WORKLOAD_IDENTITY_POOL` | `gitlab-pool` | Stage 0 |
| `WORKLOAD_IDENTITY_PROVIDER` | `gitlab-provider` | Stage 0 |
| `SERVICE_ACCOUNT` | `tf-deployer@gclt-aicoe-dev-st.iam.gserviceaccount.com` | Stage 0; `translation` is in `allowed_repositories` (verified, WS-B8) |

### B. GCP Infrastructure

| Variable | Value | Source |
|---|---|---|
| `GCP_PROJECT_ID` | `gclt-aicoe-dev-st` (number `499193286543`) | `terraform/2-foundations/gclt-aicoe-dev-st.tf` |
| `GCP_REGION` | `europe-west3` | Platform-wide convention |
| `ARTIFACT_REPO` | `containers` | Verified CMEK Docker repo in `gclt-aicoe-dev-st` |
| `IMAGE_NAME` | `translation` | |

### C. Cloud Run Service, Network & Runtime Identity

| Variable | Value | Source |
|---|---|---|
| `CLOUD_RUN_API_SERVICE` | `translation-api` | WS-B1 rename — was `translation-api-service`. **Required, no fallback** — there is no shared `CLOUD_RUN_SERVICE` base name; set both this and `CLOUD_RUN_WORKER_SERVICE` explicitly |
| `CLOUD_RUN_WORKER_SERVICE` | `translation-worker` | WS-B1 rename — was `translation-worker-service`. **Required, no fallback** |
| `CLOUD_RUN_NETWORK` | `projects/gclt-aicoe-dev-network/global/networks/gclt-aicoe-dev-vpc` | Stage 3 |
| `CLOUD_RUN_SUBNET` | `projects/gclt-aicoe-dev-network/regions/europe-west3/subnetworks/gclt-aicoe-dev-cloudrun-ew3` | Stage 3; `st`'s serverless-robot SA now holds `compute.networkUser` here (WS-B7) |
| `CLOUD_RUN_API_SA` | `translation-api-sa@gclt-aicoe-dev-st.iam.gserviceaccount.com` | New — Terraform stage 2 now provisions API and worker identities separately. **Required, no fallback** — `.gitlab-ci.yml` no longer has a shared `CLOUD_RUN_SA` to fall back to; always set both this and `CLOUD_RUN_WORKER_SA` |
| `CLOUD_RUN_WORKER_SA` | `translation-worker-sa@gclt-aicoe-dev-st.iam.gserviceaccount.com` | New, same as above. **Required, no fallback** |

### D. App-Config Secrets & Storage

| Variable | Value | Source |
|---|---|---|
| `APP_CONFIG_API_SECRET_NAME` | `translation-api-env` | Terraform stage 2, CMEK `st-ew3/secrets` |
| `APP_CONFIG_WORKER_SECRET_NAME` | `translation-worker-env` | Terraform stage 2, CMEK `st-ew3/secrets` |
| `APP_CONFIG_API_SECRET_VERSION` | `latest` | Optional, defaults to `latest` if unset. Independent from the worker's version below — no shared `APP_CONFIG_SECRET_VERSION` exists anymore, so pinning the API to an older secret version (e.g. while testing a payload change on the worker only) doesn't affect the worker |
| `APP_CONFIG_WORKER_SECRET_VERSION` | `latest` | Optional, defaults to `latest` if unset. Independent from the API's version above |
| `GCS_BUCKET_NAME` | `gclt-aicoe-dev-st-translation` | Terraform stage 2. **Renamed 2026-09-05** from a bucket shared with Sales-Agent (`gclt-aicoe-dev-st-artifacts`) — that bucket granted both apps' service accounts `storage.objectAdmin` on the whole bucket with no prefix scoping, a real cross-app blast-radius gap (Sales-Agent's SA could read/write/delete Translation's objects and vice versa). Split into dedicated per-app buckets, matching the old `aicoeprod` platform's convention. `aihub-bff-sa` still has `storage.objectAdmin` on this bucket (it uploads source documents here) |
| `ASSETS_MOUNT_PATH` | `/mnt/assets-cache` | |
| `GCS_ASSETS_DIR` | `assets` | |

### E. Cloud Tasks

| Variable | Value | Source |
|---|---|---|
| `CLOUD_TASKS_QUEUE` | `translation-jobs` | Terraform stage 6b (unchanged name) |
| `CLOUD_TASKS_OIDC_SERVICE_ACCOUNT` | `worker-invoker-sa@gclt-aicoe-dev-st.iam.gserviceaccount.com` | Terraform stage 2; **do not** invent a `translation`-specific invoker SA name — this identity is shared across both backends by platform convention |

### F. Pipeline Access & Runner Environment (masked, protected — unchanged)

| Variable | Notes |
|---|---|
| `AZURE_PAT` | Azure DevOps PAT, Code (Read), for `sync-from-azure` |
| `GITLAB_PUSH_TOKEN` | `write_repository`, for promotion jobs |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | Corporate runner egress |

---

## 3. `/secrets/.env` Payload Keys (Secret Manager — never in Git, never a GitLab CI/CD variable)

These are the contents of the `translation-api-env` and `translation-worker-env` Secret Manager
secrets (CMEK `st-ew3/secrets`, created by Terraform stage 2; **populating the actual values is a
manual, out-of-band step** — `gcloud secrets versions add`, same pattern as
`docs/20-apigee-manual-proxy-product-app-guide.md` Part 3 in `AICOE-Terraform`).

| Key | Applies to | Value / Note |
|---|---|---|
| `API_PREFIX` | API | `/api/translation/v1` (D6 — changed from the old prefix; Apigee does no path rewrite, see `AICOE-Terraform/docs/20` §5.6) |
| `APIGEE_RUNTIME_SA_EMAIL` | API | `apigee-int-runtime@gclt-aicoe-dev-apigee.iam.gserviceaccount.com` — the identity `apigee_auth.py` requires the verified Google ID token's `email` claim to equal |
| `CLOUD_RUN_SERVICE_URL` | API | This service's own `https://...run.app` URL — the ID-token audience `apigee_auth.py` verifies against. **Bootstrapping note:** unknown before the first deploy. Deploy once, resolve the real URL with `gcloud run services describe translation-api --project=gclt-aicoe-dev-st --region=europe-west3 --format='value(status.url)'`, add it as a new secret version, then redeploy (or restart) so the running revision picks it up. The service will 401 every real Apigee call — and, per the `IS_LOCAL=false` boot validator, refuse to start at all — until this is set correctly |
| `LLM_GATEWAY_BASE_URL` | API + worker | `https://llm.aicoedev-int.colt.net` (WS-E2) — not usable until the `llm` Apigee proxy is built (GAP-REGISTER R-08); leave `LLM_GATEWAY_ENABLED=false` until then |
| `LLM_GATEWAY_API_KEY_SECRET` | API + worker | Despite the `_SECRET` suffix (kept to match the plan's naming), this holds the **raw** `translation-api` developer app's Apigee consumer key value itself (`docs/21-apigee-llm-gateway-manual-setup-guide.md` in `AICOE-Terraform`) — appended directly into this same `translation-api-env`/`translation-worker-env` payload, not a pointer to a separate Secret Manager resource. This repo has no existing "fetch a named secret live at runtime" client (unlike `aihub-bff`'s `apigee_api_key_secret`), so it follows the simpler mounted-dotenv convention already used for everything else in this file |
| `LLM_GATEWAY_ENABLED` | API + worker | `false` until WS-E1's `llm` proxy is deployed and WS-E4 (`GoogleSearch`/grounding-equivalent) is verified working through it. This is the escape hatch back to calling Vertex directly — but note D-30 still forbids granting this SA `roles/aiplatform.user`, so `false` only makes sense in an environment where direct Vertex access is otherwise possible (i.e. not in `gclt-aicoe-dev-st`) |
| `CLAUDE_MODEL` / `CLAUDE_VERTEX_REGION` | worker | Existing keys, unchanged. Translation-only — Sales-Agent does not call Claude. Also routed through the gateway when `LLM_GATEWAY_ENABLED=true` (`src/worker/doctranslator/translator/providers/claude.py`'s `AnthropicVertex` client takes `base_url`/`default_headers` directly, not `HttpOptions` like the Gemini clients — see `gateway_anthropic_vertex_kwargs()` in `src/config/llm_gateway.py`). Per the "everything LLM-related goes through the gateway" decision, this is **not optional** once the gateway is live — there is no separate decision to make Claude an exception |
| `LLM_GATEWAY_VERTEX_PROJECT` | worker | `gclt-aicoe-dev-llm` — **only** consumed by the Claude/`AnthropicVertex` client. Unlike the Gemini clients (which omit `project`/`location` entirely when the gateway is enabled, letting the gateway's own target supply them), `AnthropicVertex` always embeds a project id in its request path with no "omit it" mode, so this workload's own project id must be explicitly overridden to the central inference project. See `gateway_anthropic_vertex_kwargs()` in `src/config/llm_gateway.py` for the full explanation. Leave empty while `LLM_GATEWAY_ENABLED=false` |
| `BIGQUERY_*` | API + worker | Existing keys, unchanged — dataset is `translation_jobs` |
| `GCS_ASSETS_PREFIX` | API + worker | `assets` — static reference data (fonts, cmap tables, ONNX models, `pricing_catalog.json`) inside the dedicated bucket (`GCS_BUCKET_NAME`, §2C), FUSE-mounted read-only via `ASSETS_MOUNT_PATH`/`GCS_ASSETS_DIR` (also §2C — same prefix, two different mechanisms for reaching it). Copied byte-for-byte from the old `aicoesandox-vxai-translation-app-001/assets` 2026-09-05 |
| `GCS_GLOSSARIES_PREFIX` | API + worker | `glossaries` — a **separate top-level prefix**, not nested under `assets/` despite looking related. Also copied from the old bucket's own top-level `glossaries/` (170 KB, distinct from — and a superset of — `assets/glossaries/` in the old bucket, which was leftover/unused duplicate data not referenced by this setting) |
| `GCS_TRANSLATION_PREFIX` | API + worker | `translation-service` — the **job-data** parent prefix, e.g. `translation-service/{job_id}/...` (`src/repository/*_storage_repository.py`). Starts empty in the new bucket by design — job outputs are per-request, not asset config, so the old bucket's `translation-service/` (73 MB of real historical job data) was deliberately **not** copied |
| `GCS_INPUT_FOLDER` / `GCS_OUTPUT_FOLDER` | API + worker | `input` / `output` — sub-paths within each job's own folder under `GCS_TRANSLATION_PREFIX` |

All of the above (`GCS_ASSETS_PREFIX`, `GCS_GLOSSARIES_PREFIX`, `GCS_TRANSLATION_PREFIX`, `GCS_INPUT_FOLDER`, `GCS_OUTPUT_FOLDER`) are prefixes **inside the single dedicated bucket** named by `GCS_BUCKET_NAME` (§2C) — there is one bucket per app, not one bucket per prefix.

**Removed in this pass (WS-D3) — delete these keys from the secret payload, do not carry them forward:**
`IAP_AUDIENCE`, `HUB_IAP_AUDIENCE`, `TRANSLATION_REQUIRED_GROUP`, `JWT_SECRET_KEY`/`SECRET_KEY`,
`JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`, `REQUIRE_SCOPE_CLAIM`,
`SESSION_ABSOLUTE_MAX_MINUTES`. Leaving `IAP_AUDIENCE` in the payload is harmless by itself, but if
the code-side startup validator for it is not also deleted (WS-D3), the service will not boot at
all once it's gone from here — see `Translation/src/config/constants.py`.

---

## 4. Order of Operations

```
[Terraform stage 2 + 3 applied]           (done this pass — see AICOE-Terraform)
              │
              ▼
[Populate translation-api-env / translation-worker-env secret content — minus CLOUD_RUN_SERVICE_URL]
              │
              ▼
[GitLab CI: lint → test → build (build-and-push)]
              │
              ▼
[GitLab CI: deploy-cloud-run — worker, then API]
              │
              ▼
[Resolve real Cloud Run URLs, add CLOUD_RUN_SERVICE_URL to each secret, redeploy/restart]
              │
              ▼
[Terraform stage 6b applied]               (blocked until the two Cloud Run services above exist — GAP-REGISTER B-03)
              │
              ▼
[Apigee `int` proxy target URLs updated to the real Cloud Run URLs — docs/20 §5.1/§5.6]
              │
              ▼
[WS-G validation]
```

This branch (`gitlab-sync-branch-shared-dev`) does not run any pipeline until merged to `dev`
(D9) — populating the variables above ahead of that merge is safe and recommended, since nothing
consumes them until then.
