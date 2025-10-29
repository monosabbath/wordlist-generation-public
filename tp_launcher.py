import os
import torch.distributed as dist
import uvicorn

# Importing server loads the model and (if env is ready) initializes distributed and prebuilds
import server  # noqa: F401

def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    if rank == 0:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8010"))
        uvicorn.run("server:app", host=host, port=port, workers=1, log_level="info")
    else:
        # Participate in TP inference (single worker loop per non-rank0)
        server.tp_worker_loop()

if __name__ == "__main__":
    main()
