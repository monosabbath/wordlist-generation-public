# Choose a Python 3.12 base to satisfy requires-python (>=3.12,<3.13) [2]
FROM python:3.12-slim

# System deps for building wheels and running libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Use uv to create/manage the virtual environment (matches your setup script) [5]
RUN pip install --no-cache-dir uv

# Workdir and copy project files
WORKDIR /app
# Copy only project metadata first to leverage Docker layer caching during installs
COPY pyproject.toml .
COPY README.md .
# Copy the rest of the repo
COPY . .

# Ensure .env is present inside the image (you can overwrite at runtime)
# If you plan to supply .env at runtime via RunPod env UI, this is optional.
# Otherwise, copy your local .env:
# COPY .env /app/.env

# Create the virtual environment and install project dependencies [5][2]
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install . && \
    # Show what got installed (helpful for debugging image builds)
    python -c "import pkgutil; print('Installed:', [m.name for m in pkgutil.iter_modules()])"

# Set HF cache paths inside the image so the baked model files live in image layers [3]
ENV HF_HOME=/opt/huggingface-cache
ENV HUGGINGFACE_HUB_CACHE=/opt/huggingface-cache
# Optional: also set XDG cache dirs to keep compilers’ caches local (matches your .env) [3]
ENV XDG_CACHE_HOME=/opt/.cache
ENV TRITON_CACHE_DIR=/opt/.cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/opt/.cache/torchinductor

# Build-time args to control which model to prefetch; defaults align with your .env [3]
ARG MODEL_NAME=zai-org/GLM-4.6
ARG TRUST_REMOTE_CODE=true
# Optional token for private models (avoid putting long-lived tokens in image history)
# Pass with: --build-arg HF_TOKEN=...
ARG HF_TOKEN

# Bake the model into the image by resolving and downloading it into HF cache.
# We use huggingface_hub.snapshot_download so runtime won't need network volume access.
# Note: If TRUST_REMOTE_CODE=true in runtime, model code is executed then (not here) [4].
RUN . .venv/bin/activate && python - << 'PY'
from huggingface_hub import snapshot_download
import os, sys
repo_id = os.environ.get("MODEL_NAME") or sys.argv[1] if len(sys.argv) > 1 else "zai-org/GLM-4.6"
token = os.environ.get("HF_TOKEN") or None
# Download core model assets. local_dir not set -> uses HF_HOME/HUGGINGFACE_HUB_CACHE.
snapshot_download(
    repo_id=repo_id,
    token=token,
    local_dir=None,
    local_dir_use_symlinks=False,
    allow_patterns=[
        "*.safetensors", "*.bin", "*.pt",
        "config.json", "generation_config.json",
        "tokenizer.*", "tokenizer_config.json", "special_tokens_map.json",
        "*.model", "*.vocab", "*.spm", "*.json",
        "*.py"
    ]
)
print(f"Prefetched model for {repo_id} into HF cache at:", os.environ.get("HF_HOME"))
PY

# Make sure entrypoint is available
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Network port used by uvicorn (server.py binds to 0.0.0.0 when rank 0 in TP mode) [1][4]
EXPOSE 8010

# Default environment values; can be overridden at runtime (RunPod env UI or docker run -e)
# These mirror your .env defaults [3]
ENV PARALLEL_MODE=tp
ENV TP_FALLBACK_TO_DEVICE_MAP=false
ENV DEVICE_MAP=auto
ENV SECRET_TOKEN=changeme
ENV ATTN_IMPLEMENTATION=sdpa
ENV TOKENIZER_PADDING_SIDE=left
ENV PAD_TO_MULTIPLE_OF=64
ENV MAX_INPUT_TOKENS=512
ENV ALLOWED_MAX_NEW_TOKENS=64,128,256,512
ENV STATIC_KV_CACHE=false
ENV PREBUILD_PREFIX=true
ENV PREBUILD_WORD_COUNTS=3000
ENV PREBUILD_LANGS=es
ENV BATCH_JOB_PIPELINE_SIZE=8
ENV MODEL_NAME=$MODEL_NAME
ENV TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE

# Start the server based on PARALLEL_MODE [6][1]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
