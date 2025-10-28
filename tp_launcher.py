import os
import time
import datetime
import torch.distributed as dist
import uvicorn

# Importing server loads the model with PARALLEL_MODE=tp and joins/initializes the process group
import server  # noqa: F401

def main():
    # Ensure the PG is initialized (server will have initialized it if env was ready)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

    rank = dist.get_rank()
    if rank == 0:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8010"))
        # Single worker; other ranks participate in TP compute but don't run an HTTP server.
        uvicorn.run("server:app", host=host, port=port, workers=1, log_level="info")
    else:
        # Keep the worker alive to participate in collectives.
        while True:
            time.sleep(3600)

if __name__ == "__main__":
    main()
