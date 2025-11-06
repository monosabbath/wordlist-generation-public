from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException, Request
from ..core.auth import verify_token
from ..tasks.batch_job import enqueue_batch_job, JOB_STATUS
from fastapi.responses import FileResponse

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
):
    return enqueue_batch_job(
        background_tasks=background_tasks,
        file=file,
        settings=request.app.state.settings,
        model_service=request.app.state.model_service,
        max_tokens=max_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        vocab_lang=vocab_lang,
        vocab_n_words=vocab_n_words,
    )


@router.get("/jobs/{job_id}")
def get_batch_job_status(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job["submitted_at"],
        "error": job.get("error"),
    }


@router.get("/jobs/{job_id}/results")
def get_batch_job_results(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
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
