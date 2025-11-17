"""
Example FSDP launch script for the FastAPI app using fused experts.
Run with torchrun:
torchrun --nproc_per_node=4 wordlist_generation/scripts/launch_fsdp_server.py \
    --model fused-cydonia-24b \
    --port 8000

This assumes you already produced a fused checkpoint (use fuse_save_checkpoint.py).
"""
import argparse
import os
import torch
import torch.distributed as dist
from fastapi import FastAPI
import uvicorn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def init_dist(backend: str = "nccl"):
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

def build_fsdp_model(model):
    # Simple auto wrap policy: adjust min_num_params for your scale
    auto_wrap = transformer_auto_wrap_policy(
        {type(model)},  # Replace with block classes if available (e.g., model.model.layers[0].__class__)
        min_num_params=5_000_000,
    )
    fsdp_model = FSDP(model, auto_wrap_policy=auto_wrap, device_id=torch.device(f"cuda:{dist.get_rank()}"))
    return fsdp_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to fused checkpoint (with use_grouped_gemm=True)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", default="nccl")
    args = ap.parse_args()

    init_dist(args.backend)
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "use_grouped_gemm"):
        if not getattr(config, "use_grouped_gemm", False):
            print("[FSDP] WARNING: Fused checkpoint should have use_grouped_gemm=True")

    print(f"[Rank {local_rank}] Loading fused model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(torch.cuda.current_device())

    model = build_fsdp_model(model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Minimal FastAPI app example (not full integration with existing services)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "rank": local_rank}

    # Avoid starting server on non-zero ranks for simplicity
    if local_rank == 0:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Keep process alive
        import time
        while True:
            time.sleep(60)

if __name__ == "__main__":
    main()
