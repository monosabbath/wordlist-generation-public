# Wordlist‑Constrained Generation Server

FastAPI server for vocabulary‑constrained text generation using Hugging Face Transformers and lm‑format‑enforcer. It exposes:
- OpenAI‑compatible chat completions on /v1/chat/completions (supports SSE streaming).
- A simple batch job system on /v1/batch for running many prompts offline.

New setup overview:
- Split deployment is recommended for production:
  - Chat service (multi‑worker, low latency): wordlist_generation.app_chat:app on port 8010 (example).
  - Batch service (single worker, consistent state): wordlist_generation.app_batch:app on port 8011 (example).
- Rationale: Batch job state (JOB_STATUS) is in‑memory and must live in one process; chat can scale independently.

- Language: Python 3.12
- Inference stack: transformers, torch, accelerate
- Constraints: trie → regex with lm‑format‑enforcer
- Auth: simple bearer token or ?token=... query param

---

## Deployment modes

You can run a unified server (main.py) for local dev, or split services in production for correctness and performance isolation.

- Local dev (single process, both routers on one port):
  - uvicorn wordlist_generation.main:app --host 0.0.0.0 --port 8010 --reload

- Production (recommended):
  - Chat service (multi‑worker): uvicorn wordlist_generation.app_chat:app --host 0.0.0.0 --port 8010 --workers 4
  - Batch service (single worker): uvicorn wordlist_generation.app_batch:app --host 0.0.0.0 --port 8011 --workers 1

If you added the convenience scripts:
- ./scripts/run_chat.sh (configurable workers and port)
- ./scripts/run_batch.sh (single worker, separate port)

Optional reverse proxy: route /v1/batch/* to the batch port and everything else to the chat port.

---

## Quickstart (local dev)

1) Create a virtual environment (Python 3.12):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2) Install:

```bash
pip install -e .
```

3) Create and edit your .env:

```bash
cp .env.example .env
# then edit .env with your settings
```

4) Place any wordlists (e.g., es.txt) in wordlists/. The repo includes wordlists/es.txt as an example.

5) Run the unified server (dev):

```bash
uvicorn wordlist_generation.main:app --reload --host 0.0.0.0 --port 8010
```

6) Test authentication:

```bash
curl -H "Authorization: Bearer <your-secret-token>" http://127.0.0.1:8010/v1/models
```

If you didn’t change it, the default token is changeme (don’t use this in production).

---

## Production run (split services)

Start chat (multi‑worker):

```bash
uvicorn wordlist_generation.app_chat:app --host 0.0.0.0 --port 8010 --workers 4
```

Start batch (single worker):

```bash
uvicorn wordlist_generation.app_batch:app --host 0.0.0.0 --port 8011 --workers 1
```

Example nginx map:

```nginx
location /v1/batch/ {
  proxy_pass http://127.0.0.1:8011;
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}

location / {
  proxy_pass http://127.0.0.1:8010;
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

---

## Endpoints

- Chat service (default port 8010):
  - GET /v1/models
  - POST /v1/chat/completions
- Batch service (default port 8011):
  - POST /v1/batch/jobs
  - GET /v1/batch/jobs/{job_id}
  - GET /v1/batch/jobs/{job_id}/results

You can authenticate with a bearer header or a token query param.

Header auth:

```bash
curl -H "Authorization: Bearer $SECRET_TOKEN" http://127.0.0.1:8010/v1/models
```

Query auth:

```bash
curl "http://127.0.0.1:8010/v1/models?token=$SECRET_TOKEN"
```

---

## Chat completions

OpenAI‑compatible request/response. Optional constrained vocabulary via vocab_lang and vocab_n_words.
Beam search defaults to num_beams=10 (higher quality but slower) — adjust for latency/quality.

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

Curl examples:

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
- At startup (or on demand), the server reads WORDLIST_DIR/<lang>.txt.
- It normalizes words (NFC + lowercase), builds a trie, and converts it to a compact regex.
- During generation, a prefix function filters next tokens so outputs match the allowed words (with flexible punctuation/whitespace between words).

How to enable in a request:
- Include both vocab_lang and vocab_n_words.
- Example uses the top 3,000 words from wordlists/es.txt:

```json
{
  "messages": [...],
  "vocab_lang": "es",
  "vocab_n_words": 3000
}
```

Prebuilding:
- If PREBUILD_PREFIX=true, the server pre‑compiles prefix functions for all combinations in PREBUILD_LANGS × PREBUILD_WORD_COUNTS at startup to reduce first‑use latency.

Wordlist guidance:
- Ensure one tokenized word per line; keep words normalized (lowercased, NFC).
- Deduplicate the list; sort by frequency rank (most frequent first).

---

## Batching

The batch system lets you upload a list of chat completion requests as a JSON file. The server processes them in the background and provides a downloadable results file.

Important behavior:
- Authentication is required (token header or ?token=).
- The JSON file must be a list of objects matching the per‑request schema (at minimum, model and messages).
- Generation parameters for the entire job (e.g., max_tokens, num_beams, length_penalty, vocab_lang, vocab_n_words) are supplied in the POST query string and apply to all prompts in the file (per‑item values are ignored).
- Status is kept in‑memory (process‑local). Run the batch service with exactly one worker.
- Default ports in examples assume chat on 8010 and batch on 8011.

### 1) Prepare the input file

Example my_requests.json (a list of ChatCompletionRequest objects):

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

Place your file anywhere (e.g., /path/to/my_requests.json).

### 2) Create a batch job

Upload the file and set job‑wide parameters via query string. Use the batch service port.

- Header auth:

```bash
curl -X POST "http://127.0.0.1:8011/v1/batch/jobs?max_tokens=128&num_beams=4&length_penalty=1.0&vocab_lang=es&vocab_n_words=1000" \
     -H "Authorization: Bearer $SECRET_TOKEN" \
     -F "file=@/path/to/my_requests.json"
```

- Query auth:

```bash
curl -X POST "http://127.0.0.1:8011/v1/batch/jobs?token=$SECRET_TOKEN&max_tokens=128&num_beams=4&length_penalty=1.0&vocab_lang=es&vocab_n_words=1000" \
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

Job‑wide parameters (all optional; defaults shown):
- max_tokens: default 512 (clamped to ALLOWED_MAX_NEW_TOKENS)
- num_beams: default 10
- length_penalty: default 1.0
- vocab_lang: optional (e.g., es)
- vocab_n_words: optional (e.g., 3000)

Notes:
- For non‑Cohere models, batching uses the transformers text‑generation pipeline with batch_size=BATCH_JOB_PIPELINE_SIZE.
- Inputs are truncated to MAX_INPUT_TOKENS.
- If constrained vocab is enabled and the language file is missing/empty, the job fails with an error.

### 3) Check job status

```bash
curl -H "Authorization: Bearer $SECRET_TOKEN" \
     "http://127.0.0.1:8011/v1/batch/jobs/<job_id>"
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

Possible statuses: pending, processing, completed, failed.

### 4) Download results

When status is completed, fetch the results file:

```bash
curl -L -H "Authorization: Bearer $SECRET_TOKEN" \
     -o results.json \
     "http://127.0.0.1:8011/v1/batch/jobs/<job_id>/results"
```

The file is an array of OpenAI‑shaped chat completion objects (token counts are null in batch):

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
- Uploaded files are stored in BATCH_JOB_TEMP_DIR (defaults to OS temp) and removed after processing.
- The output file remains available for download after completion (path is tracked in memory).
- BATCH_JOB_PIPELINE_SIZE controls how many prompts the pipeline feeds per forward pass.
