import json
import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional

import torch
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from ..models.chat import ChatCompletionRequest
from ..core.generation import (
    build_template_kwargs_for_model,
    truncate_inputs_to_max_length,
    strip_unused_model_inputs,
    move_inputs_to_correct_device,
    normalize_max_new_tokens,
)
from ..core.tokens import get_stop_ids
from ..core.prefix import build_regexp_prefix_fn

logger = logging.getLogger("batch-jobs")

JOB_STATUS: Dict[str, Dict[str, Any]] = {}


def process_batch_job(
    job_id: str,
    input_path: str,
    output_path: str,
    job_config: Dict[str, Any],
    settings,
    model_service,
):
    logger.info(f"[Job {job_id}] Starting processing...")
    try:
        JOB_STATUS[job_id]["status"] = "processing"

        # 1. Read and parse the input file
        with open(input_path, "r", encoding="utf-8") as f:
            try:
                raw_requests = json.load(f)
                if not isinstance(raw_requests, list):
                    raise ValueError("Input file must contain a JSON list.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}")

        # 2. Prepare prompts/messages
        logger.info(f"[Job {job_id}] Preparing {len(raw_requests)} prompts...")
        messages_list: List[List[Dict[str, str]]] = []
        for i, req_data in enumerate(raw_requests):
            try:
                req = ChatCompletionRequest(**req_data)
                system_prompt = ""
                for msg in req.messages:
                    if msg.role == "system":
                        system_prompt = msg.content
                        break
                messages: List[Dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                for msg in req.messages:
                    if msg.role != "system":
                        messages.append({"role": msg.role, "content": msg.content})
                messages_list.append(messages)
            except ValidationError as e:
                logger.warning(f"[Job {job_id}] Skipping request {i}: Invalid format. {e}")
            except Exception as e:
                logger.warning(f"[Job {job_id}] Skipping request {i}: Error processing. {e}")

        # 3. Shared generation kwargs
        tokenizer = model_service.tokenizer
        model = model_service.model
        stop_ids = get_stop_ids(tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = stop_ids[0] if stop_ids else tokenizer.eos_token_id

        max_new_tokens = normalize_max_new_tokens(job_config.get("max_tokens", 512), settings.ALLOWED_MAX_NEW_TOKENS)
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=job_config.get("num_beams", 10),
            length_penalty=job_config.get("length_penalty", 1.0),
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Constrained vocab prefix
        vocab_lang = job_config.get("vocab_lang")
        vocab_n_words = job_config.get("vocab_n_words")
        pf = None
        if vocab_lang and vocab_n_words:
            logger.info(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
            pf = build_regexp_prefix_fn(
                tokenizer=tokenizer,
                lang=vocab_lang,
                n_words=vocab_n_words,
                wordlist_dir=settings.WORDLIST_DIR,
                allow_cohere_controls=model_service.is_cohere_reasoning_model,
            )
            if pf:
                generation_kwargs["prefix_allowed_tokens_fn"] = pf
                logger.info(f"[Job {job_id}] Successfully added prefix function.")
            else:
                raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'.")

        # 4. Run generation
        logger.info(f"[Job {job_id}] Running generation...")
        results: List[str] = []
        if model_service.is_cohere_reasoning_model:
            # Cohere path: build tokenized inputs via chat template (batched)
            encoded_list = []
            input_lengths = []
            for messages in messages_list:
                tkwargs = build_template_kwargs_for_model(settings.MODEL_NAME)
                enc = tokenizer.apply_chat_template(messages, **tkwargs)
                if enc["input_ids"].size(1) > settings.MAX_INPUT_TOKENS:
                    enc = truncate_inputs_to_max_length(enc, settings.MAX_INPUT_TOKENS)
                encoded_list.append({
                    "input_ids": enc["input_ids"][0].tolist(),
                    "attention_mask": enc["attention_mask"][0].tolist() if "attention_mask" in enc else [1] * enc["input_ids"].size(1),
                })
                input_lengths.append(len(encoded_list[-1]["input_ids"]))
            # Pad to batch
            batch = tokenizer.pad(
                encoded_list,
                padding=True,
                pad_to_multiple_of=settings.PAD_TO_MULTIPLE_OF,
                return_tensors="pt",
            )
            batch = strip_unused_model_inputs(batch)
            batch = move_inputs_to_correct_device(batch, model)

            with torch.inference_mode():
                outputs = model.generate(**batch, **generation_kwargs)

            bsz = outputs.size(0)
            for i in range(bsz):
                in_len = input_lengths[i]
                out_ids = outputs[i][in_len:]
                text = tokenizer.decode(out_ids, skip_special_tokens=True)
                results.append(text)
        else:
            # Other models: use pipeline with string prompts
            prompts: List[str] = []
            for messages in messages_list:
                tkwargs = build_template_kwargs_for_model(settings.MODEL_NAME)
                text = tokenizer.apply_chat_template(messages, **tkwargs)
                if "glm-4.6" in settings.MODEL_NAME.lower():
                    text = text + "<think></think>"
                prompts.append(text)

            for output in model_service.text_pipeline(
                prompts,
                batch_size=settings.BATCH_JOB_PIPELINE_SIZE,
                return_full_text=False,
                padding=True,
                truncation=True,
                pad_to_multiple_of=settings.PAD_TO_MULTIPLE_OF,
                max_length=settings.MAX_INPUT_TOKENS,
                **generation_kwargs,
            ):
                # Each output is a list of sequences per prompt (usually 1)
                results.append(output[0]["generated_text"])

        # 5. Save output file
        logger.info(f"[Job {job_id}] Formatting and saving results...")
        created = int(time.time())
        final_output = []
        for i, text in enumerate(results):
            resp = {
                "id": f"chatcmpl-batch-{job_id}-{i}",
                "object": "chat.completion",
                "created": created,
                "model": settings.MODEL_NAME,
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

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)

        JOB_STATUS[job_id]["status"] = "completed"
        logger.info(f"[Job {job_id}] Processing complete.")
    except Exception as e:
        logger.error(f"[Job {job_id}] Processing FAILED: {e}")
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["error"] = str(e)
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"[Job {job_id}] Cleaned up input file.")
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to clean up input file: {e}")


def enqueue_batch_job(
    background_tasks: BackgroundTasks,
    file,
    settings,
    model_service,
    max_tokens: int,
    num_beams: int,
    length_penalty: float,
    vocab_lang: str | None,
    vocab_n_words: int | None,
):
    job_id = str(uuid.uuid4())
    input_path = os.path.join(settings.BATCH_JOB_TEMP_DIR, f"{job_id}_input.json")
    output_path = os.path.join(settings.BATCH_JOB_TEMP_DIR, f"{job_id}_output.json")

    try:
        with open(input_path, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {e}")

    job_config = {
        "max_tokens": max_tokens,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "vocab_lang": vocab_lang,
        "vocab_n_words": vocab_n_words,
    }

    JOB_STATUS[job_id] = {
        "status": "pending",
        "input_path": input_path,
        "output_path": output_path,
        "submitted_at": int(time.time()),
        "config": job_config,
    }

    background_tasks.add_task(
        process_batch_job, job_id, input_path, output_path, job_config, settings, model_service
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Batch job accepted and queued for processing.",
    }
