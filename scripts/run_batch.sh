#!/usr/bin/env bash
set -euo pipefail

# Configuration:
#   BATCH_PORT: port to bind (default 8011)
# Batch server MUST run with a single worker to keep JOB_STATUS in one process.

BATCH_PORT="${BATCH_PORT:-8011}"

echo "[batch] Starting batch server on :$BATCH_PORT with 1 worker"
exec uvicorn wordlist_generation.app_batch:app \
  --host 0.0.0.0 \
  --port "$BATCH_PORT" \
  --workers 1
