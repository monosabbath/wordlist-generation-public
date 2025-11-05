# Wordlist‑Constrained Generation Server

FastAPI server for vocabulary‑constrained text generation using Hugging Face Transformers and lm‑format‑enforcer. It exposes an OpenAI‑compatible chat completions endpoint and a simple batch job system. The code is modular for readability and maintainability.

- Language: Python 3.12
- Inference stack: transformers, torch, accelerate
- Constraints: trie → regex with lm‑format‑enforcer
- Auth: simple bearer token or `?token=...` query param

## Contents

- Quickstart
- Configuration (environment variables)
- Endpoints
  - GET /v1/models
  - POST /v1/chat/completions (request/response examples)
- Constrained vocabulary (how it works, how to enable)
- Batching (clear, step‑by‑step guide with examples)
- Operational and performance tips
- Troubleshooting

---

## Quickstart

1) Create a virtual environment (Python 3.12):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2) Install:

```bash
pip install -e .
```

3) Create and edit your .env (example shown below):

```bash
cp .env.example .env
# then edit .env with your settings
```

4) Place any wordlists (e.g., `es.txt`) in the `wordlists/` directory. The repo includes `wordlists/es.txt` as an example.

5) Run the server:

```bash
uvicorn wordlist_generation.main:app --reload --host 0.0.0.0 --port 8010
```

6) Test authentication:

```bash
curl -H "Authorization: Bearer <your-secret-token>" http://127.0.0.1:8010/v1/models
```

If you didn’t change it, the default token is `changeme` (don’t use this in production).

---

## Configuration

All configuration is environment‑driven (loaded via `.env`). Key settings and safe defaults:

```dotenv
# Auth
SECRET_TOKEN=changeme          # REQUIRED in production (use a strong value)

# Model/runtime
MODEL_NAME=google/gemma-3-27b-it
TRUST_REMOTE_CODE=false        # set true only for trusted model repos
TORCH_DTYPE=bf16               # "auto", "bf16", or "fp16"
DEVICE_MAP=auto                # transformers dispatch (can shard/offload)

# Attention implementation: sdpa | flash_attention_2 | eager
ATTN_IMPLEMENTATION=sdpa

# Tokenization/padding
TOKENIZER_PADDING_SIDE=left
PAD_TO_MULTIPLE_OF=16          # 8 or 16 is a good default; 64 can over‑pad
MAX_INPUT_TOKENS=512
ALLOWED_MAX_NEW_TOKENS=64,128,256,512

# Optional optimization
STATIC_KV_CACHE=false          # enable only if measured faster on your HW

# Constrained vocab prebuild (optional)
PREBUILD_PREFIX=true
PREBUILD_WORD_COUNTS=3000
PREBUILD_LANGS=es
WORDLIST_DIR=wordlists

# Batch jobs
BATCH_JOB_PIPELINE_SIZE=8
# BATCH_JOB_TEMP_DIR=/tmp      # defaults to OS temp dir

# Logging
LOG_LEVEL=INFO
```

Notes:
- Keep `TOKENIZER_PADDING_SIDE=left` for decoder‑only models and batching.
- Prefer `PAD_TO_MULTIPLE_OF=8` or `16` unless you measure a benefit with `64`.
- `STATIC_KV_CACHE` can improve throughput on some setups if memory allows.
- If `TRUST_REMOTE_CODE=true`, ensure the model repo is trusted.

---

## Endpoints

### GET /v1/models

Lists the single configured model.

```bash
curl -H "Authorization: Bearer $SECRET_TOKEN" http://127.0.0.1:8010/v1/models
```

Response:

```json
{
  "object": "list",
  "data": [
    { "id": "google/gemma-3-27b-it", "object": "model", "owned_by": "owner" }
  ]
}
```

You can also pass the token via query param:

```bash
curl "http://127.0.0.1:8010/v1/models?token=$SECRET_TOKEN"
```

### POST /v1/chat/completions

- OpenAI‑compatible shape for messages.
- Optional constrained vocabulary via `vocab_lang` and `vocab_n_words`.
- Beam search defaults to `num_beams=10` (higher quality but slower) — adjust for your latency/quality balance.

Request:

```json
{
  "model": "google/gemma-3-27b-it",
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Write a short greeting in Spanish." }
  ],
  "max_tokens": 128,
  "vocab_lang": "es",
  "vocab_n_words": 3000,
  "num_beams": 4,
  "length_penalty": 1.0
}
```

Curl examples (auth as header or query):

```bash
# Header auth
curl -H "Authorization: Bearer $SECRET_TOKEN" \
     -H "Content-Type: application/json" \
     -d @request.json \
     http://127.0.0.1:8010/v1/chat/completions

# Query auth
curl -H "Content-Type: application/json" \
     -d @request.json \
     "http://127.0.0.1:8010/v1/chat/completions?token=$SECRET_TOKEN"
```

Response (abridged):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "google/gemma-3-27b-it",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 60,
    "total_tokens": 102
  }
}
```

---

## Constrained vocabulary

The server can constrain generation to a wordlist using lm‑format‑enforcer.

How it works:
- At startup (or on demand), the server reads `WORDLIST_DIR/<lang>.txt`.
- It normalizes words (NFC + lowercase), builds a trie, and converts it to a compact regex.
- During generation, a prefix function filters next tokens so outputs match the allowed words (with flexible punctuation/whitespace between words).

How to enable in a request:
- Include both `vocab_lang` and `vocab_n_words`.
- Example uses the top 3,000 words from `wordlists/es.txt`:

```json
{
  "messages": [...],
  "vocab_lang": "es",
  "vocab_n_words": 3000
}
```

Prebuilding:
- If `PREBUILD_PREFIX=true`, the server pre‑compiles prefix functions for all combinations in `PREBUILD_LANGS` × `PREBUILD_WORD_COUNTS` at startup to reduce first‑use latency.

Wordlist guidance:
- Ensure one tokenized word per line; keep words normalized (lowercased, NFC).
- Deduplicate the list; sort by frequency rank (most frequent first).

---

## Batching

The batch system lets you upload a list of chat completion requests as a JSON file. The server processes them in the background and provides a downloadable results file.

Important behavior:
- Authentication is required (token header or `?token=`).
- The JSON file must be a list of objects matching the per‑request schema (at minimum, `model` and `messages`).
- Generation parameters for the entire job (e.g., `max_tokens`, `num_beams`, `length_penalty`, `vocab_lang`, `vocab_n_words`) are supplied in the POST query string and apply to all prompts in the file. Any per‑item values for these fields in the uploaded JSON are ignored.
- Status is kept in‑memory (process‑local). If you run multiple Uvicorn workers, each has its own queue/state.

### 1) Prepare the input file

Example `my_requests.json` (a list of ChatCompletionRequest objects):

```json
[
  {
    "model": "google/gemma-3-27b-it",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ]
  },
  {
    "model": "google/gemma-3-27b-it",
    "messages": [
      { "role": "user", "content": "Translate 'hello world' to Spanish." }
    ]
  },
  {
    "model": "google/gemma-3-27b-it",
    "messages": [
      { "role": "user", "content": "Write a one-sentence poem." }
    ]
  }
]
```

Place your file anywhere (e.g., `/path/to/my_requests.json`).

### 2) Create a batch job

You upload the file and set job‑wide parameters via query string. Either pass the token as a header or as `?token=...`.

- Header auth:

```bash
curl -X POST "http://127.0.0.1:8010/v1/batch/jobs?max_tokens=128&num_beams=4&length_penalty=1.0&vocab_lang=es&vocab_n_words=1000" \
     -H "Authorization: Bearer $SECRET_TOKEN" \
     -F "file=@/path/to/my_requests.json"
```

- Query auth:

```bash
curl -X POST "http://127.0.0.1:8010/v1/batch/jobs?token=$SECRET_TOKEN&max_tokens=128&num_beams=4&length_penalty=1.0&vocab_lang=es&vocab_n_words=1000" \
     -F "file=@/path/to/my_requests.json"
```

Response:

```json
{
  "job_id": "6c1f1b0e-...-....",
  "status": "pending",
  "message": "Batch job accepted and queued for processing."
}
```

Job‑wide parameters (all optional; shown with defaults):
- `max_tokens`: default 512 (clamped to `ALLOWED_MAX_NEW_TOKENS`)
- `num_beams`: default 10
- `length_penalty`: default 1.0
- `vocab_lang`: optional (e.g., `es`)
- `vocab_n_words`: optional (e.g., `3000`)

Notes:
- For non‑Cohere models, batching uses the transformers text‑generation pipeline under the hood with `batch_size=BATCH_JOB_PIPELINE_SIZE`.
- Inputs are truncated to `MAX_INPUT_TOKENS`.
- If constrained vocab is enabled and the language file is missing/empty, the job fails with an error.

### 3) Check job status

```bash
curl -H "Authorization: Bearer $SECRET_TOKEN" \
     "http://127.0.0.1:8010/v1/batch/jobs/<job_id>"
```

Example response:

```json
{
  "job_id": "6c1f1b0e-...-....",
  "status": "processing",
  "submitted_at": 1730000100,
  "error": null
}
```

Possible statuses: `pending`, `processing`, `completed`, `failed`.

### 4) Download results

When status is `completed`, fetch the results file:

```bash
curl -L -H "Authorization: Bearer $SECRET_TOKEN" \
     -o results.json \
     "http://127.0.0.1:8010/v1/batch/jobs/<job_id>/results"
```

The file is an array of OpenAI‑shaped chat completion objects (token counts are `null` in batch):

```json
[
  {
    "id": "chatcmpl-batch-<job_id>-0",
    "object": "chat.completion",
    "created": 1730000123,
    "model": "google/gemma-3-27b-it",
    "choices": [
      {
        "index": 0,
        "message": { "role": "assistant", "content": "..." },
        "finish_reason": "stop"
      }
    ],
    "usage": { "prompt_tokens": null, "completion_tokens": null, "total_tokens": null }
  },
  { "... second item ..." }
]
```

Implementation details:
- Uploaded files are stored in `BATCH_JOB_TEMP_DIR` (defaults to OS temp) and removed after processing.
- The output file remains available for download after completion (path is tracked in memory).
- `BATCH_JOB_PIPELINE_SIZE` controls how many prompts the pipeline feeds per forward pass.

---

## Operational and performance tips

- Beam search: `num_beams=10` is high‑quality but slow. For lower latency, try `num_beams=1–4`.
- Inputs: keep `MAX_INPUT_TOKENS` realistic; longer prompts increase prefill cost.
- Padding: left padding pairs well with KV cache schemes and batching. Use `PAD_TO_MULTIPLE_OF=8` or `16` unless measured otherwise.
- Static KV cache: only enable if you’ve measured throughput gains and have VRAM headroom.
- Security: keep `TRUST_REMOTE_CODE=false` unless you control the model repo; use a strong `SECRET_TOKEN`.

---

## Troubleshooting

- 403 Forbidden:
  - Missing or incorrect token. Pass `Authorization: Bearer ...` header or `?token=...`.
- 404 on batch results:
  - Wrong `job_id` or the job hasn’t completed yet (check status first).
- Job failed (batch):
  - Check `error` in `/v1/batch/jobs/{job_id}`. Common causes: invalid JSON file (must be a JSON list), missing wordlist for `vocab_lang`, or OOM on large batches.
- Slow responses:
  - Reduce `num_beams`, lower `max_tokens`, or lower `BATCH_JOB_PIPELINE_SIZE` if you see GPU OOM.
- Device offload:
  - With large models, `DEVICE_MAP=auto` may offload to CPU and increase latency. Consider quantization (4‑bit) and/or different hardware.

---

## Development notes

- Auth helper accepts either `Authorization: Bearer <token>` or `?token=<token>`.
- Startup optionally prebuilds constrained vocab prefix functions for configured languages and word counts.
- Batch status is in‑memory per process. If you run multiple workers, stick to one worker for batch routes or swap to a shared store (e.g., Redis) for production.

---
