# Wordlist-Constrained Generation Server

FastAPI server for constrained-vocabulary generation using Transformers and lm-format-enforcer. 
Modularized for readability and maintainability.

## Features
- OpenAI-compatible chat completions: `POST /v1/chat/completions`
- Model listing: `GET /v1/models`
- Batch job processing: upload a JSON list of requests and retrieve results
- Regex-based constrained vocabulary (trie + lm-format-enforcer)
- Environment-driven configuration

## Quickstart

1) Create a virtual environment (Python 3.12)
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

2) Install
   pip install -e .

3) Copy .env.example to .env and edit values
   cp .env.example .env

4) Place any wordlists (e.g., es.txt) in the `wordlists/` directory (default). The example includes `wordlists/es.txt`.

5) Run the server
   uvicorn wordlist_generation.main:app --reload --host 0.0.0.0 --port 8010

6) Test authentication
   - The server requires a Bearer token matching SECRET_TOKEN in your .env.
   - Example:
     curl -H "Authorization: Bearer changeme" http://0.0.0.0:8010/v1/models

## Endpoints

- GET /v1/models
- POST /v1/chat/completions
- POST /v1/batch/jobs
- GET /v1/batch/jobs/{job_id}
- GET /v1/batch/jobs/{job_id}/results

## Input format (chat completions)
{
  "model": "google/gemma-3-27b-it",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 128,
  "vocab_lang": "es",
  "vocab_n_words": 3000,
  "num_beams": 10,
  "length_penalty": 1.0
}

Use `vocab_lang` + `vocab_n_words` to enable constrained vocabulary generation.

## Notes
- SECRET_TOKEN should be strong and never committed to Git.
- If you change `PREBUILD_LANGS` or wordlists, restart the server to rebuild prefix constraints.


## Batching
This job will use num_beams=4 AND the Spanish vocab with 1000 words
curl -X POST "http://127.0.0.1:8000/v1/batch/jobs?token=my-secret-token-structured-generation&num_beams=4&vocab_lang=es&vocab_n_words=1000" \
     -F "file=@/path/to/your/requests.json"

POST /v1/batch/jobs: You upload your JSON file (which should contain a list of ChatCompletionRequest objects). The server accepts it, assigns a job_id, and immediately returns, while starting the processing in the background.

GET /v1/batch/jobs/{job_id}: You use this to check the status ("pending", "processing", "completed", or "failed").

GET /v1/batch/jobs/{job_id}/results: Once the status is "completed", you call this to download the final JSON response file.

Example File (my_requests.json):
JSON

[
  {
    "model": "your-model-name",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  },
  {
    "model": "your-model-name",
    "messages": [
      {
        "role": "user",
        "content": "Translate 'hello world' to Spanish."
      }
    ]
  },
  {
    "model": "your-model-name",
    "messages": [
      {
        "role": "user",
        "content": "What is 2+2?"
      }
    ]
  }
]
