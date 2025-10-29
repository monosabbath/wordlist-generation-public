#!/usr/bin/env bash
set -euo pipefail

cd /app

# If a .env exists and we want to use it for defaults, load it without overriding explicitly-set envs
if [[ -f "/app/.env" ]]; then
  # Export any variables in .env that are not already set
  set -a
  # shellcheck disable=SC1091
  . /app/.env
  set +a
fi

# Show selected model and mode
echo "MODEL_NAME=${MODEL_NAME:-unset}"
echo "PARALLEL_MODE=${PARALLEL_MODE:-device_map}"
echo "TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}"
echo "HF_HOME=${HF_HOME:-/opt/huggingface-cache}"

# Activate venv
. /app/.venv/bin/activate

# Optional: ensure baked model is present; if not (e.g., you changed MODEL_NAME at runtime),
# attempt to download it once here. Comment this block out if you want truly offline-only runs.
if [[ -n "${MODEL_NAME:-}" ]]; then
  python - << 'PY' || true
from huggingface_hub import snapshot_download
import os
repo = os.environ.get("MODEL_NAME")
if repo:
    print(f"Verifying HF cache for {repo}...")
    try:
        snapshot_download(repo_id=repo, local_dir=None, local_dir_use_symlinks=False, token=os.environ.get("HF_TOKEN") or None)
        print("HF cache OK")
    except Exception as e:
        print("Warning: could not prefetch at runtime:", e)
PY
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"

if [[ "${PARALLEL_MODE:-device_map}" == "tp" ]]; then
  # Tensor-parallel using torchrun; rank 0 will run uvicorn (see tp_launcher.py) [1][6]
  NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
  if [[ -z "${NPROC}" || "${NPROC}" -lt 1 ]]; then NPROC=1; fi
  MASTER_PORT="${MASTER_PORT:-29500}"
  echo "Starting tensor-parallel server with ${NPROC} processes on ${HOST}:${PORT} ..."
  exec torchrun --standalone --nproc-per-node="${NPROC}" --master-port="${MASTER_PORT}" tp_launcher.py
else
  # Single-process device_map path [6]
  echo "Starting single-process server on ${HOST}:${PORT} ..."
  exec uvicorn server:app --host "${HOST}" --port "${PORT}"
fi
