#!/usr/bin/env bash
# Creates dataset/tables when missing. When a table exists:
#   - adds missing columns via `bq update`
#   - if any shared column has a type/mode mismatch, or live table has extra
#     columns not in the schema file, deletes the table and recreates it
#     (destructive — all rows in that table are lost)
#
# Table names are identical in every project; only PROJECT (and derived dataset id) change.
#
# Usage:
#   ./scripts/create_bigquery_tables.sh sandbox
#   ./scripts/create_bigquery_tables.sh dev prod
#   ./scripts/create_bigquery_tables.sh all
#   ./scripts/create_bigquery_tables.sh --project my-gcp-project
#   ./scripts/create_bigquery_tables.sh --dry-run sandbox
#
# Override default project ids (optional):
#   BQ_PROJECT_SANDBOX=aicoesandox BQ_PROJECT_DEV=aicoedev BQ_PROJECT_PROD=aicoeprod \
#     ./scripts/create_bigquery_tables.sh all
#
# Requires: gcloud (authenticated) and bq on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCHEMA_DIR="${SCRIPT_DIR}/bigquery_schemas"

readonly LOCATION="${BQ_LOCATION:-europe-west1}"

readonly TABLE_TRANSLATION_JOBS="translation_jobs"
readonly TABLE_TRANSLATION_COSTS="translation_costs"
readonly TABLE_DLP_MAPPINGS="dlp_mappings"
readonly TABLE_TRANSLATION_REVIEWS="translation_reviews"

readonly PROJECT_SANDBOX="${BQ_PROJECT_SANDBOX:-aicoesandox}"
readonly PROJECT_DEV="${BQ_PROJECT_DEV:-aicoedev}"
readonly PROJECT_PROD="${BQ_PROJECT_PROD:-aicoeprod}"

readonly SCHEMA_TRANSLATION_JOBS="${SCHEMA_DIR}/translation_jobs.json"
readonly SCHEMA_TRANSLATION_COSTS="${SCHEMA_DIR}/translation_costs.json"
readonly SCHEMA_DLP_MAPPINGS="${SCHEMA_DIR}/dlp_mappings.json"
readonly SCHEMA_TRANSLATION_REVIEWS="${SCHEMA_DIR}/translation_reviews.json"

DRY_RUN=0
CUSTOM_PROJECT=""
CUSTOM_DATASET=""

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

log() {
  printf '==> %s\n' "$*"
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

require_tools() {
  command -v gcloud >/dev/null 2>&1 || {
    echo "ERROR: gcloud not found on PATH" >&2
    exit 1
  }
  command -v bq >/dev/null 2>&1 || {
    echo "ERROR: bq not found on PATH (install Google Cloud SDK BigQuery component)" >&2
    exit 1
  }
  command -v python3 >/dev/null 2>&1 || {
    echo "ERROR: python3 not found on PATH" >&2
    exit 1
  }
  local schema_file
  for schema_file in \
    "$SCHEMA_TRANSLATION_JOBS" \
    "$SCHEMA_TRANSLATION_COSTS" \
    "$SCHEMA_DLP_MAPPINGS" \
    "$SCHEMA_TRANSLATION_REVIEWS"; do
    [[ -f "$schema_file" ]] || {
      echo "ERROR: Missing schema file: $schema_file" >&2
      exit 1
    }
  done
}

dataset_for_project() {
  local project="$1"
  if [[ -n "$CUSTOM_DATASET" ]]; then
    printf '%s' "$CUSTOM_DATASET"
  else
    printf '%s_translation_dataset' "$project"
  fi
}

project_for_env() {
  case "$1" in
    sandbox) printf '%s' "$PROJECT_SANDBOX" ;;
    dev) printf '%s' "$PROJECT_DEV" ;;
    prod) printf '%s' "$PROJECT_PROD" ;;
    *)
      echo "ERROR: Unknown environment '$1' (expected sandbox, dev, prod, or all)" >&2
      exit 1
      ;;
  esac
}

table_exists() {
  local project="$1" dataset="$2" table="$3"
  bq show "${project}:${dataset}.${table}" >/dev/null 2>&1
}

dataset_exists() {
  local project="$1" dataset="$2"
  bq show "${project}:${dataset}" >/dev/null 2>&1
}

ensure_dataset() {
  local project="$1" dataset="$2"
  local ref="${project}:${dataset}"

  if dataset_exists "$project" "$dataset"; then
    log "Dataset already exists: ${ref}"
    return 0
  fi

  log "Creating dataset ${ref} (location=${LOCATION})"
  run bq mk --location="$LOCATION" "$ref"
}

create_table() {
  local ref="$1" schema_file="$2" partition_field="$3" clustering_fields="${4:-}"

  log "Creating table ${ref} (partition=${partition_field}, schema=${schema_file})"
  if [[ -n "$clustering_fields" ]]; then
    run bq mk --table \
      --time_partitioning_type=DAY \
      --time_partitioning_field="$partition_field" \
      --clustering_fields="$clustering_fields" \
      "$ref" \
      "$schema_file"
  else
    run bq mk --table \
      --time_partitioning_type=DAY \
      --time_partitioning_field="$partition_field" \
      "$ref" \
      "$schema_file"
  fi
}

sync_table() {
  local project="$1" dataset="$2" table="$3" schema_file="$4" partition_field="$5" clustering_fields="${6:-}"
  local ref="${project}:${dataset}.${table}"

  if ! table_exists "$project" "$dataset" "$table"; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "Would create table ${ref} (partition=${partition_field}, schema=${schema_file})"
    else
      create_table "$ref" "$schema_file" "$partition_field" "$clustering_fields"
    fi
    return 0
  fi

  local tmp_missing tmp_drift
  tmp_missing="$(mktemp)"
  tmp_drift="$(mktemp)"

  python3 - "$ref" "$schema_file" "$tmp_missing" "$tmp_drift" <<'PY'
import json
import subprocess
import sys

TYPE_ALIASES = {
    "INTEGER": "INT64",
    "INT64": "INT64",
    "FLOAT": "FLOAT64",
    "FLOAT64": "FLOAT64",
    "BOOLEAN": "BOOL",
    "BOOL": "BOOL",
}


def norm_type(value: str) -> str:
    return TYPE_ALIASES.get(value, value)


ref, schema_file, missing_path, drift_path = sys.argv[1:5]
desired = json.load(open(schema_file, encoding="utf-8"))
raw = subprocess.check_output(
    ["bq", "show", "--format=prettyjson", ref],
    text=True,
)
existing = json.loads(raw)["schema"]["fields"]
by_name = {field["name"]: field for field in existing}

missing = []
drift = []
desired_names = {field["name"] for field in desired}
for field in desired:
    name = field["name"]
    current = by_name.get(name)
    if current is None:
        missing.append(field)
        continue
    same_type = norm_type(current.get("type", "")) == norm_type(field.get("type", ""))
    same_mode = current.get("mode", "NULLABLE") == field.get("mode", "NULLABLE")
    if not (same_type and same_mode):
        drift.append(
            f"{name}: live={current.get('type')}/{current.get('mode', 'NULLABLE')} "
            f"desired={field.get('type')}/{field.get('mode', 'NULLABLE')}"
        )

for name in sorted(by_name):
    if name not in desired_names:
        drift.append(f"{name}: extra column in live table not in schema")

json.dump(missing, open(missing_path, "w", encoding="utf-8"))
open(drift_path, "w", encoding="utf-8").write("\n".join(drift))
PY

  local missing_count drift_count
  missing_count="$(python3 -c "import json; print(len(json.load(open('$tmp_missing'))))")"
  drift_count="$(grep -c . "$tmp_drift" 2>/dev/null || true)"
  drift_count="${drift_count:-0}"

  if [[ "$drift_count" -gt 0 ]]; then
    while IFS= read -r line; do
      [[ -n "$line" ]] && echo "WARNING: Schema drift on ${ref}: ${line}" >&2
    done < "$tmp_drift"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "Would delete and recreate ${ref} (all existing rows would be lost)"
    else
      log "Schema drift on ${ref} — deleting table and recreating (all rows will be lost)"
      run bq rm -f -t "$ref"
      create_table "$ref" "$schema_file" "$partition_field" "$clustering_fields"
    fi
    rm -f "$tmp_missing" "$tmp_drift"
    return 0
  fi

  if [[ "$missing_count" -eq 0 ]]; then
    log "Schema up to date: ${ref}"
    rm -f "$tmp_missing" "$tmp_drift"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would update schema for ${ref} (add ${missing_count} column(s))"
  else
    log "Updating schema for ${ref} (adding ${missing_count} column(s))"
    run bq update "$ref" "$tmp_missing"
  fi
  rm -f "$tmp_missing" "$tmp_drift"
}

provision_project() {
  local project="$1"
  local dataset
  dataset="$(dataset_for_project "$project")"

  log "Provisioning BigQuery for project=${project} dataset=${dataset}"
  run gcloud config set project "$project" >/dev/null

  ensure_dataset "$project" "$dataset"
  sync_table "$project" "$dataset" "$TABLE_TRANSLATION_JOBS" "$SCHEMA_TRANSLATION_JOBS" "submitted_at" "status,job_id"
  sync_table "$project" "$dataset" "$TABLE_TRANSLATION_COSTS" "$SCHEMA_TRANSLATION_COSTS" "timestamp"
  sync_table "$project" "$dataset" "$TABLE_DLP_MAPPINGS" "$SCHEMA_DLP_MAPPINGS" "masked_at" "job_id,chunk_index"
  sync_table "$project" "$dataset" "$TABLE_TRANSLATION_REVIEWS" "$SCHEMA_TRANSLATION_REVIEWS" "created_at" "job_id"

  log "Done: ${project}:${dataset}"
  echo "  - ${TABLE_TRANSLATION_JOBS}"
  echo "  - ${TABLE_TRANSLATION_COSTS}"
  echo "  - ${TABLE_DLP_MAPPINGS}"
  echo "  - ${TABLE_TRANSLATION_REVIEWS}"
}

main() {
  local targets=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help)
        usage 0
        ;;
      -n | --dry-run)
        DRY_RUN=1
        shift
        ;;
      --project)
        [[ $# -ge 2 ]] || {
          echo "ERROR: --project requires a value" >&2
          exit 1
        }
        CUSTOM_PROJECT="$2"
        shift 2
        ;;
      --dataset)
        [[ $# -ge 2 ]] || {
          echo "ERROR: --dataset requires a value" >&2
          exit 1
        }
        CUSTOM_DATASET="$2"
        shift 2
        ;;
      all)
        targets+=(sandbox dev prod)
        shift
        ;;
      sandbox | dev | prod)
        targets+=("$1")
        shift
        ;;
      *)
        echo "ERROR: Unknown argument '$1'" >&2
        usage 1
        ;;
    esac
  done

  if [[ -n "$CUSTOM_PROJECT" ]]; then
    if [[ ${#targets[@]} -gt 0 ]]; then
      echo "ERROR: Use either --project or environment names, not both" >&2
      exit 1
    fi
    require_tools
    provision_project "$CUSTOM_PROJECT"
    return 0
  fi

  if [[ ${#targets[@]} -eq 0 ]]; then
    echo "ERROR: Specify sandbox, dev, prod, all, or --project" >&2
    usage 1
  fi

  require_tools

  local env
  for env in "${targets[@]}"; do
    provision_project "$(project_for_env "$env")"
    echo
  done
}

main "$@"
