#!/bin/bash
# Stop Cloud Tasks Emulator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🛑 Stopping Cloud Tasks Emulator..."

if podman ps | grep -q cloud-tasks-emulator; then
    podman stop cloud-tasks-emulator
    podman rm cloud-tasks-emulator
    echo "✅ Cloud Tasks Emulator stopped and removed"
else
    echo "ℹ️  Cloud Tasks Emulator is not running"
fi

# Clean up config directory (optional - you may want to keep it)
# rm -rf "${PROJECT_ROOT}/configs"
