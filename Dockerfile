# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc1

# System prep
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Optional: set HF caches to persistent paths (override via env at runtime)
ENV HF_HOME=/workspace/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/workspace/huggingface-cache \
    XDG_CACHE_HOME=/workspace/.cache \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor

# Create workspace
WORKDIR /workspace

# Install uv and Python deps
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml /workspace/
COPY server.py /workspace/
COPY start.sh /workspace/
COPY parallel_config.example.yaml /workspace/parallel_config.yaml
COPY .env /workspace/.env
# Optional: word lists (put your <lang>.txt files in container or mount at runtime)
# COPY es.txt /workspace/es.txt
# COPY en.txt /workspace/en.txt
# ...

# Install Python dependencies via uv (from pyproject)
RUN uv venv && . .venv/bin/activate && uv pip install . --system

EXPOSE 8010

# Default startup
CMD ["/bin/bash", "/workspace/start.sh"]
