#!/usr/bin/env bash
# Bootstrap Cloud Tasks queues (standard + optional high priority).
# Everything is configuration-driven; defaults mirror production settings.
#
# Usage:
#   # Standard queue only (default):
#   PROJECT=aicoeprod REGION=europe-west1 QUEUE=translation-job \
#     ./scripts/create_cloud_tasks_queue.sh
#
#   # Both queues:
#   PROJECT=aicoeprod REGION=europe-west1 QUEUE=translation-job \
#     QUEUE_HIGH=translation-job-high ./scripts/create_cloud_tasks_queue.sh
set -euo pipefail

PROJECT="${PROJECT:?Set PROJECT}"
REGION="${REGION:-europe-west1}"
QUEUE="${QUEUE:-translation-job}"
QUEUE_HIGH="${QUEUE_HIGH:-}"

# Standard tier defaults (mirrors current production)
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
MAX_RATE="${MAX_RATE:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"

# High-priority tier defaults (smaller reserved budget, faster retry)
HIGH_MAX_CONCURRENT="${HIGH_MAX_CONCURRENT:-10}"
HIGH_MAX_RATE="${HIGH_MAX_RATE:-10}"
HIGH_MIN_BACKOFF="${HIGH_MIN_BACKOFF:-5s}"

create_one_queue() {
  local name="$1"
  local concurrent="$2"
  local rate="$3"
  local min_backoff="$4"

  if gcloud tasks queues describe "$name" --location="$REGION" --project="$PROJECT" >/dev/null 2>&1; then
    echo "Queue $name already exists in $REGION (skipped)."
    return 0
  fi

  gcloud tasks queues create "$name" \
    --location="$REGION" \
    --project="$PROJECT" \
    --max-concurrent-dispatches="$concurrent" \
    --max-dispatches-per-second="$rate" \
    --max-attempts="$MAX_ATTEMPTS" \
    --min-backoff="$min_backoff" \
    --max-backoff=300s \
    --max-retry-duration=3600s

  echo "Created queue $name (max_concurrent=$concurrent, rate=${rate}/s)."
}

echo "=== Provisioning standard queue ==="
create_one_queue "$QUEUE" "$MAX_CONCURRENT" "$MAX_RATE" "10s"

if [ -n "$QUEUE_HIGH" ]; then
  echo "=== Provisioning high-priority queue ==="
  create_one_queue "$QUEUE_HIGH" "$HIGH_MAX_CONCURRENT" "$HIGH_MAX_RATE" "$HIGH_MIN_BACKOFF"
fi

echo
echo "Done. Remember to grant the API service account roles/cloudtasks.enqueuer"
echo "on both queues, and OIDC SA roles/run.invoker on the worker."

