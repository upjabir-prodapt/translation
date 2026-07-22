#!/usr/bin/env bash
# Bootstrap a Cloud Tasks queue for the translation worker.
# Usage:
#   PROJECT=... REGION=europe-west1 QUEUE=translation-jobs \
#   MAX_CONCURRENT=4 ./scripts/create_cloud_tasks_queue.sh
set -euo pipefail

PROJECT="${PROJECT:?Set PROJECT}"
REGION="${REGION:-europe-west1}"
QUEUE="${QUEUE:-translation-jobs}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
MAX_RATE="${MAX_RATE:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"

gcloud tasks queues describe "$QUEUE" --location="$REGION" --project="$PROJECT" >/dev/null 2>&1 && {
  echo "Queue $QUEUE already exists in $REGION"
  exit 0
}

gcloud tasks queues create "$QUEUE" \
  --location="$REGION" \
  --project="$PROJECT" \
  --max-concurrent-dispatches="$MAX_CONCURRENT" \
  --max-dispatches-per-second="$MAX_RATE" \
  --max-attempts="$MAX_ATTEMPTS" \
  --min-backoff=10s \
  --max-backoff=300s \
  --max-retry-duration=3600s

echo "Created queue $QUEUE (max concurrent=$MAX_CONCURRENT)."
echo "Remember: grant API SA roles/cloudtasks.enqueuer and OIDC SA roles/run.invoker on the worker."
