#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Start FastAPI server (OpenAI-compatible)
uv run --active uvicorn server:app --host 0.0.0.0 --port 8010
