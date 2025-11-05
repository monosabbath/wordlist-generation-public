#!/usr/bin/env bash
set -euo pipefail

# Configuration:
#   PORT: port to bind (default 8010)
#   UVICORN_WORKERS: number of workers for chat (default 1)

PORT="${PORT:-8010}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

echo "[chat] Starting chat server on :$PORT with $UVICORN_WORKERS workers"
exec uvicorn wordlist_generation.app_chat:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$UVICORN_WORKERS"
