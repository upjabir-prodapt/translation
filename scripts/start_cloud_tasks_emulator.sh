#!/bin/bash
# Start Cloud Tasks Emulator using Podman
# This emulator allows local testing of Cloud Tasks without connecting to GCP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="cloud-practice-dev-2"
LOCATION="us-central1"
QUEUE_NAME="translation_service_queue"
EMULATOR_PORT=9022
TARGET_HOST="host.docker.internal"  # Podman-compatible host reference
TARGET_PORT=8081

echo "🚀 Starting Cloud Tasks Emulator with Podman..."

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo "❌ Error: Podman is not installed"
    echo "Install with: brew install podman"
    exit 1
fi

# Stop and remove existing container if running
podman stop cloud-tasks-emulator 2>/dev/null || true
podman rm cloud-tasks-emulator 2>/dev/null || true

# Create configs directory in project root
mkdir -p "${PROJECT_ROOT}/configs"

# Create queue.yaml configuration
cat > "${PROJECT_ROOT}/configs/queue.yaml" <<EOF
queue:
- name: ${QUEUE_NAME}
  rate: 10/s
  bucket_size: 100
  max_concurrent_requests: 100
  retry_parameters:
    task_retry_limit: 3
    task_age_limit: 3600s
    min_backoff: 1s
    max_backoff: 10s
    max_doublings: 5
EOF

echo "📋 Created queue configuration at ${PROJECT_ROOT}/configs/queue.yaml"

# Run the emulator with Podman
# Using spine3/cloudtasks-emulator (well-maintained, available on Docker Hub)
echo "🐳 Starting Podman container..."
podman run -d \
  --name cloud-tasks-emulator \
  -p ${EMULATOR_PORT}:9090 \
  -e GCP_PROJECT=${PROJECT_ID} \
  -e QUEUE_YAML_LOCATION=${LOCATION} \
  -e TARGET_HOST=${TARGET_HOST} \
  -e TARGET_PORT=${TARGET_PORT} \
  -e EMULATOR_PORT=9090 \
  -v "${PROJECT_ROOT}/configs:/configs:ro,z" \
  docker.io/spine3/cloudtasks-emulator:latest

echo ""
echo "✅ Cloud Tasks Emulator started successfully!"
echo ""
echo "📊 Emulator Details:"
echo "   Container: cloud-tasks-emulator"
echo "   Port: ${EMULATOR_PORT}"
echo "   Project: ${PROJECT_ID}"
echo "   Location: ${LOCATION}"
echo "   Queue: ${QUEUE_NAME}"
echo "   Worker Target: http://${TARGET_HOST}:${TARGET_PORT}"
echo ""
echo "🔍 View logs:"
echo "   podman logs -f cloud-tasks-emulator"
echo ""
echo "🛑 Stop emulator:"
echo "   podman stop cloud-tasks-emulator"
echo ""
echo "💡 Set environment variable in your app:"
echo "   export CLOUD_TASKS_EMULATOR_HOST=localhost:${EMULATOR_PORT}"
echo ""
