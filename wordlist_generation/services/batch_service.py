import json
import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from wordlist_generation.schemas.chat import ChatCompletionRequest
from wordlist_generation.services.generation_service import GenerationService

logger = logging.getLogger("batch-jobs")


class BatchService:
    def __init__(self, settings, generation_service: GenerationService):
        self.settings = settings
        self.gen_service = generation_service
        self.job_status: Dict[str, Dict[str, Any]] = {}

    def _process_job(
        self,
        job_id: str,
        input_path: str,
        output_path: str,
        job_config: Dict[str, Any],
    ):
        logger.info(f"[Job {job_id}] Starting processing...")
        try:
            self.job_status[job_id]["status"] = "processing"

            # 1. Read input file (Using simple read for now, can be upgraded to streaming for massive files)
            with open(input_path, "r", encoding="utf-8") as f:
                try:
                    raw_requests = json.load(f)
                    if not isinstance(raw_requests, list):
                        raise ValueError("Input file must contain a JSON list.")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON: {e}")

            logger.info(f"[Job {job_id}] Processing {len(raw_requests)} requests...")
            
            final_output = []
            created = int(time.time())
            
            # Mini-batching isn't strictly needed here because we are reusing the GenerationService
            # which handles one request at a time. If we wanted true batched inference we would need
            # a different interface on GenerationService. 
            # For now, we iterate to prevent OOM and ensure stability.
            
            for i, req_data in enumerate(raw_requests):
                try:
                    # Validate request
                    req = ChatCompletionRequest(**req_data)
                    
                    # Override config from job if not present in request or force overrides if desired.
                    # Here we assume job_config acts as defaults if request params are missing, 
                    # but the pydantic model has defaults. 
                    # Let's respect individual request params but fallback to job global params if passed
                    # (This logic depends on specific requirements, here we'll trust the request object mostly
                    # but could inject job_config values if needed).
                    
                    # Run generation
                    result = self.gen_service.generate(req)
                    
                    resp = {
                        "id": f"chatcmpl-batch-{job_id}-{i}",
                        "object": "chat.completion",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": result["text"]},
                                "finish_reason": result["finish_reason"],
                            }
                        ],
                        "usage": result["usage"],
                    }
                    final_output.append(resp)

                except ValidationError as e:
                    logger.warning(f"[Job {job_id}] Request {i} validation failed: {e}")
                    final_output.append({
                        "id": f"chatcmpl-batch-{job_id}-{i}",
                        "error": {"message": str(e), "type": "validation_error"}
                    })
                except Exception as e:
                    logger.warning(f"[Job {job_id}] Request {i} failed: {e}")
                    final_output.append({
                        "id": f"chatcmpl-batch-{job_id}-{i}",
                        "error": {"message": str(e), "type": "generation_error"}
                    })

            # Save output
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2)

            self.job_status[job_id]["status"] = "completed"
            logger.info(f"[Job {job_id}] Processing complete.")

        except Exception as e:
            logger.error(f"[Job {job_id}] Processing FAILED: {e}")
            self.job_status[job_id]["status"] = "failed"
            self.job_status[job_id]["error"] = str(e)
        finally:
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                    logger.info(f"[Job {job_id}] Cleaned up input file.")
            except Exception as e:
                logger.warning(f"[Job {job_id}] Failed to clean up input file: {e}")

    def enqueue(
        self,
        background_tasks: BackgroundTasks,
        file,
        max_tokens: int,
        num_beams: int,
        length_penalty: float,
        vocab_lang: str | None,
        vocab_n_words: int | None,
        # Sampling params
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ):
        job_id = str(uuid.uuid4())
        input_path = os.path.join(self.settings.BATCH_JOB_TEMP_DIR, f"{job_id}_input.json")
        output_path = os.path.join(self.settings.BATCH_JOB_TEMP_DIR, f"{job_id}_output.json")

        # Ensure directory exists
        os.makedirs(os.path.dirname(input_path), exist_ok=True)

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
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        self.job_status[job_id] = {
            "status": "pending",
            "input_path": input_path,
            "output_path": output_path,
            "submitted_at": int(time.time()),
            "config": job_config,
        }

        background_tasks.add_task(
            self._process_job, job_id, input_path, output_path, job_config
        )

        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Batch job accepted and queued for processing.",
        }

