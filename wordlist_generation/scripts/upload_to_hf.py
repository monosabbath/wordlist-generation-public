#!/usr/bin/env python3
"""
Upload large model folders to Hugging Face Hub.

Usage:
  python upload_to_hf.py \
      --repo_id your-username/model-name \
      --folder_path ./your-model-folder \
      --private
"""
import argparse
import os
import sys
import math
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import HfApi

REQUIRED_FILES = ["config.json", "tokenizer_config.json"]
WEIGHT_PATTERNS = ["model.safetensors", "model-00001-of-", "pytorch_model.bin"]

def human_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    p = int(math.floor(math.log(num_bytes, 1024)))
    return f"{num_bytes / (1024**p):.2f}{units[p]}"

def scan_folder(folder: Path):
    total = 0
    count = 0
    for p in folder.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            count += 1
    return total, count

def validate_folder(folder: Path):
    """Basic validation that this looks like a model folder"""
    missing = [f for f in REQUIRED_FILES if not (folder / f).exists()]
    
    # Check for at least one weight file
    has_weights = any(
        (folder / pattern).exists() or 
        any(f.name.startswith(pattern) for f in folder.iterdir() if f.is_file())
        for pattern in WEIGHT_PATTERNS
    )
    
    if missing:
        print(f"[warn] Missing recommended files: {', '.join(missing)}")
    if not has_weights:
        print("[warn] No weight files found (safetensors/bin)")
    
    print("[validate] Folder structure looks okay")

def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="HF repo id: username/model-name")
    ap.add_argument("--folder_path", required=True, help="Local model folder")
    ap.add_argument("--private", action="store_true", help="Create private repo")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    ap.add_argument("--num-workers", type=int, default=12, help="Upload workers")
    ap.add_argument("--use-large-if-over-gb", type=float, default=5.0)
    ap.add_argument("--force-large", action="store_true", help="Force large upload method")
    ap.add_argument("--dry-run", action="store_true", help="Don't upload, just validate")
    args = ap.parse_args()
    
    # Get folder
    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.is_dir():
        print(f"[error] Not a folder: {folder}")
        sys.exit(1)
    
    # Get token
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("[error] No HF token found. Set HF_TOKEN env var or use --token")
        sys.exit(1)
    
    # Scan folder
    total_bytes, file_count = scan_folder(folder)
    size_gb = total_bytes / (1024**3)
    print(f"[scan] {file_count} files, {human_size(total_bytes)} ({size_gb:.2f} GB)")
    
    # Validate
    validate_folder(folder)
    
    if args.dry_run:
        print("[dry-run] Stopping before upload")
        return
    
    # Determine upload method
    use_large = args.force_large or (size_gb > args.use_large_if_over_gb)
    
    api = HfApi(token=token)
    
    # Create repo
    print(f"[repo] Creating/verifying: {args.repo_id}")
    try:
        api.create_repo(
            repo_id=args.repo_id,
            private=args.private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"[repo] ✓ https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"[error] Failed to create repo: {e}")
        sys.exit(1)
    
    # Upload
    if use_large:
        print(f"[upload] Starting LARGE upload ({args.num_workers} workers)...")
        print("[upload] This is resumable - you can Ctrl+C and restart")
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",  # REQUIRED for upload_large_folder
            folder_path=str(folder),
            num_workers=args.num_workers,
        )
    else:
        print("[upload] Starting standard upload...")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(folder),
            commit_message="Upload model",
        )
    
    print("[done] ✓ Upload complete!")
    print(f"[done] View at: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
