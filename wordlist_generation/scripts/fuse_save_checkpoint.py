"""
Offline fusion script:
- Loads an unfused checkpoint entirely on CPU.
- Calls fuse_experts() (requires grouped_gemm + feature branch of Transformers).
- Saves a new fused checkpoint directory with config.use_grouped_gemm=True.
Usage:
    python wordlist_generation/scripts/fuse_save_checkpoint.py \
        --model TheDrummer/Cydonia-24B-v4.2.0 \
        --out_dir fused-cydonia-24b
Then set in .env:
    MODEL_NAME=fused-cydonia-24b
    USE_GROUPED_GEMM=true
    LOAD_FUSED_EXPERTS=true
    FUSE_ON_CPU_BEFORE_SHARD=false
"""
import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Source (unfused) model repo or local path")
    ap.add_argument("--out_dir", required=True, help="Destination directory for fused checkpoint")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","float32","auto"])
    args = ap.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, None)

    print(f"[Fuse] Loading config for {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "use_grouped_gemm"):
        config.use_grouped_gemm = False  # will set True after fusion
    print("[Fuse] Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype if torch_dtype else None,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if not hasattr(model, "fuse_experts"):
        raise RuntimeError("Model does not provide fuse_experts(); ensure feature branch is installed.")

    print("[Fuse] Fusing experts...")
    model.fuse_experts()
    if hasattr(model.config, "use_grouped_gemm"):
        model.config.use_grouped_gemm = True
        print("[Fuse] Set config.use_grouped_gemm=True")

    print("[Fuse] Saving tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tok.save_pretrained(args.out_dir)

    print("[Fuse] Saving fused model...")
    model.save_pretrained(args.out_dir)

    # Save minimal metadata
    meta = {
        "source_model": args.model,
        "fused": True,
        "dtype": args.dtype,
    }
    with open(os.path.join(args.out_dir, "fusion_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Fuse] Done. Fused checkpoint at: {args.out_dir}")

if __name__ == "__main__":
    main()
