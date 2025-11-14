from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse

from wordlist_generation.api.dependencies import verify_token

router = APIRouter(prefix="/v1/batch", tags=["batch"])


@router.post("/jobs")
def create_batch_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth_ok: bool = Depends(verify_token),
    max_tokens: int = 512,
    num_beams: int = 5,
    length_penalty: float = 1.0,
    vocab_lang: str | None = None,
    vocab_n_words: int | None = None,
    # Sampling parameters
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
):
    bp = request.app.state.batch_processor
    return bp.enqueue(
        background_tasks=background_tasks,
        file=file,
        max_tokens=max_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        vocab_lang=vocab_lang,
        vocab_n_words=vocab_n_words,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )


@router.get("/jobs/{job_id}")
def get_batch_job_status(job_id: str, request: Request, auth_ok: bool = Depends(verify_token)):
    bp = request.app.state.batch_processor
    job = bp.job_status.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job["submitted_at"],
        "error": job.get("error"),
    }


@router.get("/jobs/{job_id}/results")
def get_batch_job_results(job_id: str, request: Request, auth_ok: bool = Depends(verify_token)):
    bp = request.app.state.batch_processor
    job = bp.job_status.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "completed":
        output_path = job["output_path"]
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
