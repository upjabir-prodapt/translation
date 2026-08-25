# Implementation Plan — Dual Cloud Tasks Queues (high-priority `.txt`)

> Scope: route `.txt` translation jobs onto a dedicated high-priority Cloud
> Tasks queue. Everything (queue names, enabling, which formats qualify) is
> configuration-driven so it can be reproduced in `aicoeprod` without code
> changes.

## [Overview]

Give plain-text translation jobs their own Cloud Tasks queue so they are never stuck behind long-running PDF jobs, with the format-to-priority mapping applied server-side and driven entirely by configuration.

**Why a second queue at all.** Cloud Tasks has **no per-task priority**. Verified
against the installed SDK — `tasks_v2.Task` exposes exactly
`app_engine_http_request, create_time, dispatch_count, dispatch_deadline,
first_attempt, http_request, last_attempt, name, response_count,
schedule_time, view`. There is no `priority` field, no reorder API
(`create_task, delete_task, get_task, list_tasks, run_task`), and no
lease/pull API in v2. Separate queues with independent dispatch budgets are
the only supported mechanism.

**What "priority" does and does not buy.** Two queues give each tier its own
`maxConcurrentDispatches` budget, so a text job is dispatched from a queue
that PDFs cannot saturate. It does **not** preempt an in-flight job: with
`containerConcurrency=1` and `maxScale=30`, if all 30 worker instances are
busy, a high-priority task still waits for a free instance. Documented here
so nobody expects preemption later.

**Server-side mapping, not client-controlled.** Per the decision on this
ticket, the user never chooses priority. `ProcessingOptions.priority` already
exists and is already persisted to BigQuery, but is **never read at enqueue
time** — dead config today. This plan wires it up and has the API *override*
whatever the client sends, based on document format.

**Relevant UI behaviour.** `TranslationPage.tsx:222` wraps pasted text into
`new File([sourceText], 'pasted-text.txt', { type: 'text/plain' })`. So the
"paste some text" path already arrives as `.txt` and will be auto-promoted
with no UI change at all. The comment at `TranslationPage.tsx:213` explicitly
notes `priority` is intentionally omitted from the FormData.

## [Types]

**`src/api/schemas/requests.py`** — no shape change. `ProcessingOptions.priority`
already exists as `Literal["standard", "high"]` defaulting to `"standard"`.
Only its docstring changes, to record that the server may override it.

**`src/shared/schemas/tasks.py`** — extend `TranslateTaskPayload`:

```python
class TranslateTaskPayload(BaseModel):
    job_id: str = Field(..., min_length=1)
    traceparent: str | None = None
    tracestate: str | None = None
    # Routing/observability metadata. Deliberately NON-PII: the worker
    # re-reads the full job (including cost_attribution) from BigQuery in
    # translate_task_handler.py:65, so user identity must NOT be duplicated
    # into the task body.
    priority: str | None = None
    doc_format: str | None = None
```

Both new fields are optional so in-flight tasks created by the previous
revision still deserialise during a rolling deploy.

**`src/config/constants.py`** — new `Settings` fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `CLOUD_TASKS_QUEUE_HIGH` | `str` | `""` | High-priority queue name. Empty = feature off, everything uses `CLOUD_TASKS_QUEUE`. |
| `HIGH_PRIORITY_FORMATS` | `list[str]` | `["txt"]` | Document formats auto-promoted to high priority. |
| `HIGH_PRIORITY_ROUTING_ENABLED` | `bool` | `True` | Master kill-switch for auto-promotion. |

Making the format list a setting (rather than hardcoding `txt`) is what lets
`aicoeprod` later promote, say, small DOCX without a code deploy.

## [Files]

### New files

| Path | Purpose |
|---|---|
| `docs/plans/high-priority-queue-plan.md` | This plan |
| `docs/cloud-tasks-queues.md` | Runbook: create/configure both queues in any project, verify routing, roll back |
| `tests/api/services/test_cloud_tasks_priority.py` | Queue selection + fallback tests |

### Modified files

| Path | Change |
|---|---|
| `src/config/constants.py` | 3 new settings above |
| `src/shared/schemas/tasks.py` | `priority` + `doc_format` on `TranslateTaskPayload` |
| `src/api/services/cloud_tasks_service.py` | `_queue_path(priority)`, `_resolve_queue_name(priority)`, `enqueue_translate(..., priority, doc_format)`; log the chosen queue |
| `src/api/services/translation_service.py` | `_resolve_effective_priority()`; thread priority through `_schedule_background_pipeline` from both `submit_translations` and `_do_submit` |
| `src/api/schemas/requests.py` | docstring only — note server-side override |
| `scripts/create_cloud_tasks_queue.sh` | Provision both queues; per-tier rate limits |
| `.env.example.api`, `.env.api.local`, `.env.api.prod`, `tests/test.env` | New settings |

### AI-Hub-UI (separate repo)

| Path | Change |
|---|---|
| `src/pages/TranslationPage.tsx` | Update the `priority` comment at ~213 to state the server auto-promotes `.txt`; **no FormData change** |
| `src/types/translation.ts` | Comment on `TranslateRequest.priority` noting it is server-controlled |

The UI change is documentation-only. Sending `priority` from the browser would
be ignored anyway, and adding a control would imply user choice we do not want.

### Not changed, deliberately

- `ALLOWED_EXTENSIONS` — **verified dead config.** `grep -rn ALLOWED_EXTENSIONS src/`
  returns only the `Settings` declaration at `constants.py:334`; nothing reads
  it. The real gate is `DocumentInput.validate_filename`, which already accepts
  `.txt`. Adding `.txt` to the env list would imply an enforcement that does
  not exist. Flagged in the runbook instead.

## [Functions]

### New

| Function | File | Signature | Purpose |
|---|---|---|---|
| `_resolve_queue_name` | `cloud_tasks_service.py` | `(priority: str \| None) -> str` | Returns `CLOUD_TASKS_QUEUE_HIGH` for `"high"` when set, else `CLOUD_TASKS_QUEUE`. Falls back with a WARNING if high is requested but unconfigured. |
| `_resolve_effective_priority` | `translation_service.py` | `(request: TranslateRequest) -> str` | Server-side mapping: `"high"` when `HIGH_PRIORITY_ROUTING_ENABLED` and `document.format in HIGH_PRIORITY_FORMATS`, else `"standard"`. Ignores the client value. |

### Modified

| Function | File | Change |
|---|---|---|
| `CloudTasksService._queue_path` | `cloud_tasks_service.py` | Accept `priority: str \| None = None`; resolve via `_resolve_queue_name` |
| `CloudTasksService.enqueue_translate` | `cloud_tasks_service.py` | Add `priority`/`doc_format` (both defaulted); include in payload; log `queue=` + `priority=`; keep the `CLOUD_TASKS_QUEUE` guard |
| `CloudTasksService._enqueue_local_http` | `cloud_tasks_service.py` | Accept and forward the same fields so local dev exercises the identical payload |
| `TranslationService._schedule_background_pipeline` | `translation_service.py` | Add `priority`/`doc_format` params; pass to `enqueue_translate` |
| `TranslationService.submit_translations` | `translation_service.py` | Compute effective priority once; store it in `processing_options` so BigQuery records what was actually used |
| `TranslationService._do_submit` | `translation_service.py` | Same |

### Removed

None.

## [Classes]

- **Modified:** `TranslateTaskPayload` (+2 optional fields); `CloudTasksService`
  (queue resolution); `TranslationService` (priority resolution).
- **New / removed:** none.

## [Dependencies]

**No new packages.** `google-cloud-tasks` is already vendored; `queue_path()`
takes the queue name as a plain argument, so a second queue needs no new API.

**External prerequisite:** the high-priority queue must exist in the target
project before `CLOUD_TASKS_QUEUE_HIGH` is set. Ordering is deliberate — if the
setting is empty the code silently uses the standard queue, so the config can
safely land before the queue is created.

## [Testing]

Framework: `pytest` + `pytest-asyncio` (`asyncio_mode = auto`), env from
`tests/test.env`. **Baseline to preserve: 784 passed.**

| Test | Coverage |
|---|---|
| `test_cloud_tasks_priority.py::test_high_priority_uses_high_queue` | `priority="high"` → `CLOUD_TASKS_QUEUE_HIGH` in the queue path |
| `::test_standard_priority_uses_default_queue` | `priority="standard"` → `CLOUD_TASKS_QUEUE` |
| `::test_high_falls_back_when_unset` | `CLOUD_TASKS_QUEUE_HIGH=""` → standard queue + WARNING, no crash |
| `::test_priority_none_defaults_to_standard` | Back-compat for callers not passing priority |
| `::test_payload_carries_priority_and_format` | Task body includes both fields |
| `::test_payload_has_no_user_pii` | **Asserts** `user_id`/`email`/`organization` absent from the task body |
| `test_translation_service.py::test_txt_is_auto_promoted` | `.txt` upload → `enqueue_translate(priority="high")` |
| `::test_pdf_stays_standard` | `.pdf` → `"standard"` |
| `::test_client_high_priority_is_ignored_for_pdf` | Client sends `high`, format `pdf` → server forces `standard` |
| `::test_routing_disabled_forces_standard` | `HIGH_PRIORITY_ROUTING_ENABLED=False` → `.txt` stays standard |
| `::test_configurable_formats` | `HIGH_PRIORITY_FORMATS=["docx"]` → `.docx` promoted, `.txt` not |
| existing `test_cloud_tasks_service.py` | Unchanged — new params default, so all 4 tests still pass |

**Gates:** `uv run pytest -q` ≥ 784 passed; `ruff check src tests` at the
pre-existing baseline of 11 findings (none added).

## [Implementation Order]

1. **Settings** — add the 3 fields to `constants.py` with safe defaults
   (`CLOUD_TASKS_QUEUE_HIGH=""` keeps the feature inert until configured).
2. **Payload** — extend `TranslateTaskPayload` with the two optional fields.
3. **Queue selection** — `_resolve_queue_name` + `_queue_path(priority)`;
   thread through `enqueue_translate` and `_enqueue_local_http`; add the
   chosen-queue log line.
4. **Priority resolution** — `_resolve_effective_priority`; wire both submit
   paths; persist the effective value into `processing_options`.
5. **Tests** — new `test_cloud_tasks_priority.py` + service routing tests;
   confirm the existing 4 Cloud Tasks tests still pass untouched.
6. **Provisioning script** — extend `create_cloud_tasks_queue.sh` to create
   both queues with per-tier limits, idempotently.
7. **Env files** — `.env.example.api`, `.env.api.local`, `.env.api.prod`,
   `tests/test.env`.
8. **Runbook** — `docs/cloud-tasks-queues.md`.
9. **UI** — comment-only updates in `TranslationPage.tsx` / `translation.ts`.
10. **Validate** — full suite + ruff; local E2E asserting a `.txt` upload logs
    the high queue and a `.pdf` logs the standard one.
11. **Memory bank** — update `activeContext.md` / `progress.md`.

## [Cancel hardening] — in scope

Two defects found while verifying the AI-Hub-UI contract, now folded into this
change because the second one directly degrades the new high-priority queue.

### C1 — `cancel_job` has no ownership check (security)

`jobs_handler.py:32` calls `job_service.cancel_job(job_id, request)` with no
`user_id`, unlike `list_jobs` and `get_jobs_status` which both pass
`current_user.email`. **Any authenticated user can cancel any other user's job
by guessing/observing a job ID.**

Fix, mirroring the existing pattern in `get_jobs_status`
(`job_service.py`, which compares
`(job.get("cost_attribution") or {}).get("user_id") != user_id` and raises
`JobNotFoundError`):

- `cancel_job(job_id, request, user_id)` — new required `user_id` param.
- Raise `JobNotFoundError` (**not** `403`) on mismatch, so the endpoint cannot
  be used to probe which job IDs exist. This matches `get_jobs_status`.
- Route passes `current_user.email`.

`GET /jobs/{job_id}` and `GET /jobs/{job_id}/download` have the same gap.
Fixing those changes read behaviour for the Job Tracker page, so they are
listed as follow-ups rather than bundled here — cancel is the destructive one.

### C2 — cancel does not delete the Cloud Task

`grep -rn delete_task src/` returns nothing. Cancelling only flips BigQuery to
`cancelled`; the task still dispatches, and the worker no-ops at
`translate_task_handler.py:77`. Wasteful today — and on the new high-priority
queue a cancelled text job would consume one of only ~10 reserved dispatch
slots to do nothing.

Fix:

- New `CloudTasksService.delete_translate_task(job_id, priority=None) -> bool`.
- `task_id_for_job()` already yields a deterministic name, so the task path is
  reconstructable without storing it.
- **Best-effort:** `NotFound` (already dispatched, or never created in local
  mode) is logged at DEBUG and treated as success. Any other failure logs a
  WARNING but must **not** fail the cancel — the BigQuery status is the source
  of truth and the worker still no-ops.
- Deletion is attempted **after** the status patch, so a Cloud Tasks outage
  cannot leave a job un-cancelled.
- Must try **both** queues when the priority is unknown: the stored
  `processing_options.priority` is read first, falling back to trying standard
  then high.

### Additional tests

| Test | Coverage |
|---|---|
| `test_job_service.py::test_cancel_rejects_other_users_job` | Job owned by user A, cancelled by B → `JobNotFoundError`, status unchanged |
| `::test_cancel_allows_owner` | Owner cancels → status becomes `cancelled` |
| `::test_cancel_missing_cost_attribution_is_rejected` | Legacy row with no `cost_attribution` → rejected, not crashed |
| `::test_cancel_deletes_cloud_task` | `delete_translate_task` called with the job id |
| `::test_cancel_survives_task_delete_failure` | Cloud Tasks raises → job still marked `cancelled` |
| `::test_cancel_tolerates_task_not_found` | `NotFound` → no warning-level noise, cancel succeeds |
| `test_cloud_tasks_priority.py::test_delete_uses_correct_queue` | Priority routes deletion to the matching queue |

## Follow-ups (still out of scope)

- `GET /jobs/{job_id}` and `GET /jobs/{job_id}/download` have the same missing
  ownership check as C1. Lower severity (read, not destructive) but they change
  Job Tracker behaviour, so they deserve their own change and UI regression pass.

## Endpoint parity with AI-Hub-UI (verified, no gaps)

All seven endpoints `src/api/translationApi.ts` calls exist server-side:

| UI call | Backend route | Status |
|---|---|---|
| `POST /translate` | `translate.py:32` | OK |
| `GET /translate/{id}` | `translate.py:105` | OK |
| `POST /jobs/status` | `jobs.py:24` | OK |
| `GET /jobs` | `jobs.py:44` | OK |
| `GET /jobs/{id}/download` | `jobs.py:71` | OK |
| `DELETE /jobs/{id}` | `jobs.py:60` | OK |
| `POST /reviews/{id}` | `reviews.py:20` | OK |

`GET /reviews/{job_id}` exists server-side (`reviews.py:36`) but is unused by
the UI — not a gap.

## User information in Cloud Tasks — answered

**No user data is sent to Cloud Tasks today, and this plan keeps it that way.**
`TranslateTaskPayload` carries only `job_id`, `traceparent`, `tracestate`. The
worker re-reads the job from BigQuery (`translate_task_handler.py:65`), which
is where `cost_attribution` (`user_id`, `business_unit`, `organization`)
already lives.

That is the correct design and is preserved deliberately: task bodies are
stored by Cloud Tasks and surface in logs, so duplicating PII into them would
widen exposure for no benefit. The two fields added here (`priority`,
`doc_format`) are non-PII routing metadata, and a test asserts no user fields
leak into the payload.
