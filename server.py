import os
import time
import uuid
import json  # For reading/writing job files
import tempfile  # For handling temp file locations
from functools import cache
from typing import Literal, Optional, Dict, Any
import torch
from fastapi import (  # Added background tasks and file handling
    Depends,
    FastAPI,
    HTTPException,
    Header,
    Query,
    UploadFile,
    File,
    BackgroundTasks,
)
from fastapi.responses import FileResponse  # To send the JSON file back
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# CHANGE 1: Load .env as early as possible, BEFORE other imports
load_dotenv()

from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,  # Import the pipeline
)

# Assuming 'common.py' contains GenerateRequest. If not, define a placeholder.
try:
    from common import GenerateRequest
except ImportError:
    # Define placeholder based on usage in the script if common.py is missing
    class GenerateRequest(BaseModel):
        prompt: str
        vocab_lang: Literal["en", "es"] = "en"
        vocab_n_words: int = 1000
        max_new_tokens: int = 100
        num_beams: int = 1
        repetition_penalty: float = 1.0
        length_penalty: float = 1.0


app = FastAPI()

# -----------------------
# Config via environment
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")
DEFAULT_SYSTEM_PROMPT_EN = os.getenv("DEFAULT_SYSTEM_PROMPT_EN", "")
DEFAULT_SYSTEM_PROMPT_ES = os.getenv("DEFAULT_SYSTEM_PROMPT_ES", "")
PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(
    int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "500,1000,5000").split(",")
)

# Config for Batch Jobs
BATCH_JOB_TEMP_DIR = os.getenv("BATCH_JOB_TEMP_DIR", tempfile.gettempdir())
BATCH_JOB_PIPELINE_SIZE = int(os.getenv("BATCH_JOB_PIPELINE_SIZE", "8"))
print(f"Batch jobs will be stored in: {BATCH_JOB_TEMP_DIR}")
print(f"Batch job pipeline size: {BATCH_JOB_PIPELINE_SIZE}")

# -----------------------
# Model / tokenizer load
# -----------------------
model_init_kwargs = {
    "device_map": DEVICE_MAP,
    "trust_remote_code": TRUST_REMOTE_CODE,
}

print(f"Loading model '{MODEL_NAME}' (Trust Remote Code: {TRUST_REMOTE_CODE})...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    **model_init_kwargs,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)

# -----------------------
# Create the generation pipeline
# -----------------------
# We create the pipeline using the already-loaded model and tokenizer
print("Creating text-generation pipeline...")
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=model.device,
    trust_remote_code=TRUST_REMOTE_CODE,
)
print("Pipeline created.")


# -----------------------
# Helpers for stop tokens
# -----------------------
@cache  # Cache this function call
def get_stop_ids(tok: AutoTokenizer) -> list[int]:
    stop_ids: list[int] = []
    
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, int):
            stop_ids.append(tok.eos_token_id)
        elif isinstance(tok.eos_token_id, list):
            stop_ids.extend(tok.eos_token_id)

    common_end_markers = (
        "<end_of_turn>",
        "<|eot_id|>",
        "<|im_end|>",
    )

    for special in common_end_markers:
        try:
            eid = tok.convert_tokens_to_ids(special)
            if eid is not None and eid != tok.unk_token_id:
                stop_ids.append(eid)
        except Exception:
            pass
            
    return list(set(stop_ids))

# -----------------------
# Regex-constrained vocab
# -----------------------
@cache
def build_regexp_prefix_fn(lang: Literal["en", "es"], n_words: int):
    print(f"building prefix function for {lang} ({n_words} words)")
    
    filename = lang + ".txt"
    if not os.path.exists(filename):
        print(f"Warning: Language file not found: {filename}. Skipping build.")
        return None

    try:
        with open(filename) as fin:
            words = [word.strip().lower() for word in fin]
    except Exception as e:
        print(f"Error reading language file {filename}: {e}. Skipping build.")
        return None

    words = words[:n_words]
    
    if not words:
        print(f"Warning: Vocabulary file {filename} is empty. Skipping build.")
        return None

    word_regexp = "|".join(
        "[" + w[0].lower() + w[0].upper() + "]" + w[1:] for w in words if w
    )
    word_regexp = "(" + word_regexp + ")"
    punct_regexp = "[-.,!?():;¿!¡\\s]+"
    flexible_grammar = f"({word_regexp}{punct_regexp})+"

    parser = RegexParser(flexible_grammar)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    stop_ids = set(get_stop_ids(tokenizer))

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        return list(allowed | stop_ids)

    return wrapped_prefix_fn


if PREBUILD_PREFIX:
    print("prebuilding prefix functions...")
    for lang in ("es",): # Only checking 'es' as in original
        for n_words in PREBUILD_WORD_COUNTS:
            build_regexp_prefix_fn(lang, n_words)
    print("done")

# -----------------------
# Auth helper
# -----------------------
def verify_token(
    token: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
):
    supplied = token
    if not supplied and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            supplied = parts[1]
    if SECRET_TOKEN and supplied != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

# -----------------------
# Helper for generation kwargs
# -----------------------
def _get_gen_kwargs(max_new_tokens, stop_ids, temperature=1.0, top_p=1.0, num_beams=None, repetition_penalty=None, length_penalty=None, prefix_fn=None):
    """Helper to consolidate generation arguments."""
    do_sample = (temperature is None) or (temperature > 0)
    
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if temperature is not None else 1.0,
        top_p=top_p if top_p is not None else 1.0,
    )

    if prefix_fn:
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if num_beams:
        gen_kwargs["num_beams"] = num_beams
    if repetition_penalty:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if length_penalty:
        gen_kwargs["length_penalty"] = length_penalty

    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        if tokenizer.pad_token_id is not None:
             gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        else:
             gen_kwargs["pad_token_id"] = stop_ids[0] if isinstance(stop_ids, list) and stop_ids else None
            
    return gen_kwargs

# -----------------------
# OpenAI-compatible API
# -----------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[list[str] | str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    vocab_lang: Optional[Literal["en", "es"]] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None

@app.get("/v1/models")
def list_models(auth_ok: bool = Depends(verify_token)):
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "owner",
            }
        ],
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)):
    system_prompt = DEFAULT_SYSTEM_PROMPT_EN
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
            break

    messages = []
    if system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    for msg in req.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True
    }
    if MODEL_NAME == "zai-org/GLM-4.6-FP8":
        print("Applying 'enable_thinking: False' for GLM-4.6-FP8.")
        template_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    texts = tokenizer.apply_chat_template(
        messages, **template_kwargs
    )
    
    inputs = tokenizer(texts, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    prefix_fn = None
    if req.vocab_lang and req.vocab_n_words:
        prefix_fn = build_regexp_prefix_fn(req.vocab_lang, req.vocab_n_words)
        if prefix_fn is None:
             raise HTTPException(status_code=500, detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.")

    max_new_from_request = req.max_tokens if req.max_tokens is not None else 100
    allowed_new_tokens = 1024 - input_len
    max_new_tokens = max(0, min(max_new_from_request, allowed_new_tokens))
    
    stop_ids = get_stop_ids(tokenizer)

    gen_kwargs = _get_gen_kwargs(
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        temperature=req.temperature,
        top_p=req.top_p,
        num_beams=req.num_beams,
        repetition_penalty=req.repetition_penalty,
        length_penalty=req.length_penalty,
        prefix_fn=prefix_fn
    )

    outputs = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(outputs[0].shape[0] - input_len)
    created = int(time.time())

    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": req.model or MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp


# ---------------------------------------------------
# BATCH JOB API
# ---------------------------------------------------

# In-memory store for job statuses.
# For production, you would replace this with Redis, a database, or similar.
JOB_STATUS: Dict[str, Dict[str, Any]] = {}

# The background worker function
def process_batch_job(
    job_id: str,
    input_path: str,
    output_path: str,
    job_config: Dict[str, Any],
):
    """
    This function runs in the background.
    It loads the input file, processes all requests using the pipeline,
    and saves the results to the output file.
    """
    print(f"[Job {job_id}] Starting processing...")
    try:
        JOB_STATUS[job_id]["status"] = "processing"

        # 1. Read and parse the input file
        prompts = []
        raw_requests = []
        with open(input_path, "r") as f:
            try:
                raw_requests = json.load(f)
                if not isinstance(raw_requests, list):
                    raise ValueError("Input file must contain a JSON list.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}")

        # 2. Prepare all prompts
        print(f"[Job {job_id}] Preparing {len(raw_requests)} prompts...")
        for i, req_data in enumerate(raw_requests):
            try:
                # Validate that it's a valid request object
                req = ChatCompletionRequest(**req_data)

                # Use the *exact same* templating logic as the single endpoint
                system_prompt = DEFAULT_SYSTEM_PROMPT_EN
                for msg in req.messages:
                    if msg.role == "system":
                        system_prompt = msg.content
                        break

                messages = []
                if system_prompt != "":
                    messages.append({"role": "system", "content": system_prompt})
                for msg in req.messages:
                    if msg.role != "system":
                        messages.append({"role": msg.role, "content": msg.content})

                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if MODEL_NAME == "zai-org/GLM-4.6-FP8":
                    template_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

                text = tokenizer.apply_chat_template(messages, **template_kwargs)
                prompts.append(text)

            except ValidationError as e:
                print(f"[Job {job_id}] Skipping request {i}: Invalid format. {e}")
            except Exception as e:
                print(f"[Job {job_id}] Skipping request {i}: Error processing. {e}")
        
        # 3. Prepare shared generation kwargs for the *entire* batch job
        stop_ids = get_stop_ids(tokenizer)
        
        # We need to set eos_token_id and pad_token_id for the pipeline
        if tokenizer.pad_token_id is None:
             tokenizer.pad_token_id = stop_ids[0] if stop_ids else tokenizer.eos_token_id

        generation_kwargs = dict(
            max_new_tokens=job_config.get("max_tokens", 100),
            do_sample=True,
            temperature=job_config.get("temperature", 0.7),
            top_p=job_config.get("top_p", 1.0),
            num_beams=job_config.get("num_beams", 1),
            repetition_penalty=job_config.get("repetition_penalty", 1.0),
            length_penalty=job_config.get("length_penalty", 1.0),
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        # --- UPDATED BLOCK ---
        # Build prefix function if vocab is constrained
        vocab_lang = job_config.get("vocab_lang")
        vocab_n_words = job_config.get("vocab_n_words")
        if vocab_lang and vocab_n_words:
            print(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
            prefix_fn = build_regexp_prefix_fn(vocab_lang, vocab_n_words)
            if prefix_fn:
                generation_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
                print(f"[Job {job_id}] Successfully added prefix function.")
            else:
                # Vocab file was missing or empty, fail the job
                raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'. Check server logs.")
        # --- END OF UPDATED BLOCK ---


        # 4. Run the pipeline!
        print(f"[Job {job_id}] Running pipeline (Batch Size: {BATCH_JOB_PIPELINE_SIZE})...")
        results = []
        for output in text_gen_pipeline(
            prompts,
            batch_size=BATCH_JOB_PIPELINE_SIZE,
            return_full_text=False,  # <-- IMPORTANT: Only get the *new* text
            **generation_kwargs, # This will now include prefix_fn
        ):
            # output is like [{'generated_text': '...'}]
            results.append(output[0]["generated_text"])

        # 5. Format and save the output file
        print(f"[Job {job_id}] Formatting and saving results...")
        created = int(time.time())
        final_output = []
        for i, text in enumerate(results):
            resp = {
                "id": f"chatcmpl-batch-{job_id}-{i}",
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            }
            final_output.append(resp)

        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2)

        # 6. Mark job as completed
        JOB_STATUS[job_id]["status"] = "completed"
        print(f"[Job {job_id}] Processing complete.")

    except Exception as e:
        print(f"[Job {job_id}] Processing FAILED: {e}")
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["error"] = str(e)


# Endpoint 1: Submit a new batch job
@app.post("/v1/batch/jobs")
def create_batch_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth_ok: bool = Depends(verify_token),
    # You can add query params to configure the *entire* job
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    # --- ADDED THESE PARAMETERS ---
    vocab_lang: Optional[Literal["en", "es"]] = None,
    vocab_n_words: Optional[int] = None,
):
    job_id = str(uuid.uuid4())
    input_path = os.path.join(BATCH_JOB_TEMP_DIR, f"{job_id}_input.json")
    output_path = os.path.join(BATCH_JOB_TEMP_DIR, f"{job_id}_output.json")

    # 1. Save the uploaded file
    try:
        with open(input_path, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {e}")

    # 2. Store job config
    job_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        # --- ADDED THESE PARAMETERS ---
        "vocab_lang": vocab_lang,
        "vocab_n_words": vocab_n_words,
    }

    # 3. Register the job
    JOB_STATUS[job_id] = {
        "status": "pending",
        "input_path": input_path,
        "output_path": output_path,
        "submitted_at": int(time.time()),
        "config": job_config,
    }

    # 4. Schedule the background task
    background_tasks.add_task(
        process_batch_job, job_id, input_path, output_path, job_config
    )

    # 5. Return 202 Accepted immediately
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Batch job accepted and queued for processing.",
    }


# Endpoint 2: Check job status
@app.get("/v1/batch/jobs/{job_id}")
def get_batch_job_status(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Return a subset of the info
    return {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job["submitted_at"],
        "error": job.get("error"), # Will be None if no error
    }


# Endpoint 3: Get job results
@app.get("/v1/batch/jobs/{job_id}/results")
def get_batch_job_results(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "completed":
        output_path = job["output_path"]
        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=500, detail="Job completed but output file is missing."
            )
        
        # Return the JSON file as a downloadable response
        return FileResponse(
            path=output_path,
            media_type="application/json",
            filename=f"{job_id}_output.json",
        )
    elif job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job.get('error', 'Unknown error')}",
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job['status']}",
        )
