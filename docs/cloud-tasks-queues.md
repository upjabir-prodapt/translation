# Cloud Tasks Queues — standard + high priority

Runbook for provisioning and operating the translation service's two Cloud
Tasks queues in any GCP project. Everything is configuration-driven; no code
change is required to add, rename, disable or re-tune a queue.

## Why two queues

Cloud Tasks has **no per-task priority**. The `Task` proto has no `priority`
field, there is no reorder API, and pull/lease queues were removed in v2.
Verified against the installed SDK:

```
Task fields: app_engine_http_request, create_time, dispatch_count,
             dispatch_deadline, first_attempt, http_request, last_attempt,
             name, response_count, schedule_time, view
Client task methods: create_task, delete_task, get_task, list_tasks, run_task
```

Separate queues with independent dispatch budgets are the only supported way
to prioritise. Each queue has its own `maxConcurrentDispatches`, so
high-priority work cannot be starved by a backlog of standard work.

**What this does not do:** it does not preempt a running job. With
`containerConcurrency=1` and `maxScale=30`, if every worker instance is busy a
high-priority task still waits for a free instance. Two queues guarantee
*dispatch* fairness, not *execution* preemption.

## Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `CLOUD_TASKS_QUEUE` | API env | `translation-job` | Standard queue. Required. |
| `CLOUD_TASKS_QUEUE_HIGH` | API env | `""` | High-priority queue. **Empty disables the feature** — all jobs use the standard queue. |
| `HIGH_PRIORITY_FORMATS` | API env | `["txt"]` | Formats auto-promoted to high priority. |
| `HIGH_PRIORITY_ROUTING_ENABLED` | API env | `true` | Master switch for auto-promotion. |
| `CLOUD_TASKS_LOCATION` | API env | `europe-west1` | Region for both queues. |
| `CLOUD_TASKS_PROJECT` | API env | falls back to `GOOGLE_CLOUD_PROJECT` | Project holding the queues. |

Priority is decided **server-side** from the document format. Clients cannot
request it; anything a client sends in `processing_options.priority` is
overridden. The AI-Hub-UI deliberately does not send the field.

## Provision the queues

```bash
PROJECT=aicoeprod \
REGION=europe-west1 \
QUEUE=translation-job \
QUEUE_HIGH=translation-job-high \
./scripts/create_cloud_tasks_queue.sh
```

The script is idempotent — existing queues are reported and skipped, so it is
safe to re-run. Defaults mirror the current production standard queue
(`maxConcurrentDispatches=30`, `maxDispatchesPerSecond=10`). Omit `QUEUE_HIGH`
to provision only the standard queue.

### Recommended per-tier settings

| Setting | Standard | High | Why |
|---|---|---|---|
| `--max-concurrent-dispatches` | 30 | 10 | High gets a *reserved* budget, not a bigger one. Text jobs are short (~32 s); 10 concurrent is ample and leaves worker capacity for PDFs. |
| `--max-dispatches-per-second` | 10 | 10 | Same burst rate. |
| `--min-backoff` | 10s | 5s | Retry a short job sooner. |
| `--max-attempts` | 5 | 5 | Unchanged. |

The combined budget (30 + 10 = 40) exceeds worker `maxScale=30`. That is
intentional: Cloud Run queues the excess, and it guarantees the high tier
always has slots even when the standard tier is saturated.

## Enable

Order matters — create the queue **before** setting the env var:

1. Provision `translation-job-high` (above).
2. Add to the `translation-service-env` secret (API only; the worker does not
   enqueue):

   ```
   CLOUD_TASKS_QUEUE_HIGH=translation-job-high
   HIGH_PRIORITY_FORMATS=["txt"]
   HIGH_PRIORITY_ROUTING_ENABLED=true
   ```

3. Redeploy the API service.

If the env var is set but the queue does not exist, enqueue fails with
`NOT_FOUND` and the job is marked failed. If the env var is *empty*, the code
falls back to the standard queue and logs a WARNING — safe by design, which is
why config can safely land before provisioning.

## Verify

```bash
# 1. Queues exist
gcloud tasks queues list --location=europe-west1 --project=aicoeprod

# 2. Submit a .txt job, then confirm routing in the API log
gcloud run services logs read translation-api-service \
  --region=europe-west1 --project=aicoeprod --limit=50 \
  | grep 'Enqueued Cloud Task'
```

Expected: `queue=translation-job-high priority=high doc_format=txt` for a
`.txt` upload, and `queue=translation-job priority=standard doc_format=pdf`
for a PDF.

```bash
# 3. Per-queue depth
gcloud tasks queues describe translation-job-high \
  --location=europe-west1 --project=aicoeprod
```

## Roll back

Any one of these; no code deploy needed:

| Goal | Action |
|---|---|
| Disable auto-promotion, keep the queue | `HIGH_PRIORITY_ROUTING_ENABLED=false` |
| Route everything to the standard queue | `CLOUD_TASKS_QUEUE_HIGH=""` |
| Change which formats qualify | `HIGH_PRIORITY_FORMATS=["txt","docx"]` |
| Pause the high queue without touching config | `gcloud tasks queues pause translation-job-high --location=... --project=...` |

Pausing holds tasks rather than dropping them; resume with
`gcloud tasks queues resume`.

## Reproducing in another project

The only project-specific values are `CLOUD_TASKS_PROJECT`,
`CLOUD_TASKS_LOCATION` and the two queue names. To stand this up in e.g.
`aicoesandox`:

```bash
PROJECT=aicoesandox REGION=europe-west1 \
QUEUE=translation-job QUEUE_HIGH=translation-job-high \
./scripts/create_cloud_tasks_queue.sh
```

then set the same three env vars in that project's API secret. IAM is
unchanged: the API service account needs `roles/cloudtasks.enqueuer` on
**both** queues, and the OIDC service account needs `roles/run.invoker` on the
worker (already required for the standard queue).

## Local development

`CloudTasksService._is_local_http_target()` bypasses Cloud Tasks entirely when
`IS_LOCAL=true` and `CLOUD_TASKS_WORKER_URL` starts with `http://` — it POSTs
straight to the worker. Priority is still computed and included in the payload,
and the chosen queue name is still logged, so routing can be verified locally
without provisioning anything.

## Known gotchas

- **`ALLOWED_EXTENSIONS` is dead config.** Declared in `Settings`
  (`constants.py:334`) but read nowhere. The real upload gate is
  `DocumentInput.validate_filename`, which already accepts `.pdf`, `.docx` and
  `.txt`. Do not rely on `ALLOWED_EXTENSIONS` to block a format.
- **Cancelled jobs still occupy a dispatch slot.** Cancelling sets the
  BigQuery status; it does not delete the Cloud Task, so the task is still
  dispatched and the worker no-ops. On the high-priority queue that wastes a
  reserved slot. Tracked as a separate ticket (`delete_task` on cancel).
- **Pasted text is already `.txt`.** AI-Hub-UI wraps pasted input as
  `pasted-text.txt` (`TranslationPage.tsx:222`), so it is auto-promoted with
  no UI change.
