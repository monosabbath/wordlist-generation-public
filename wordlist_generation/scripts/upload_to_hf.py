#!/usr/bin/env python3
"""
Optimized upload script for large fused model folders to Hugging Face Hub.

Features:
- Loads .env (HUGGINGFACE_HUB_TOKEN / HF_TOKEN)
- Validates minimal model file set
- Auto-selects upload_large_folder() vs upload_folder()
- Resumable when using upload_large_folder()
- Configurable workers & size threshold

Usage:
  python wordlist_generation/scripts/upload_fused_model_large.py \
      --repo_id monosabbath/fused-cydonia-24b \
      --folder_path ./fused-cydonia-24b \
      --private \
      --use-large-if-over-gb 5 \
      --num-workers 12

Force large folder mode even if small:
  python ... --force-large

Environment recommendations:
  export HF_XET_HIGH_PERFORMANCE=1
  export HF_HOME=/fast_local_cache
  export HUGGINGFACE_HUB_TOKEN=hf_...
"""
import argparse
import os
import sys
import math
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    upload_large_folder,
    HfHubHTTPError,
)


REQUIRED_ANY_ONE_OF = [
    # At least one weight file pattern must exist
    ("model.safetensors", "model-00001-of-"),
]

REQUIRED_EXACT = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
]

OPTIONAL_FILES = [
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.model",
    "fusion_metadata.json",
    "README.md",
    "LICENSE",
]


def human_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    p = int(math.floor(math.log(num_bytes, 1024)))
    return f"{num_bytes / (1024**p):.2f}{units[p]}"


def scan_folder(folder: Path) -> Tuple[int, int, List[Path]]:
    total = 0
    count = 0
    files = []
    for p in folder.rglob("*"):
        if p.is_file():
            size = p.stat().st_size
            total += size
            count += 1
            files.append(p)
    return total, count, files


def validate_structure(folder: Path) -> None:
    missing = []
    # Check exact matches
    for name in REQUIRED_EXACT:
        if not (folder / name).exists():
            missing.append(name)

    # Check weight existence (any accepted pattern)
    weight_found = False
    for wset in REQUIRED_ANY_ONE_OF:
        for candidate in wset:
            # either exact file or prefix of sharded
            if candidate.endswith("-of-"):
                # prefix match
                if any(f.name.startswith(candidate) for f in folder.iterdir()):
                    weight_found = True
                    break
            else:
                if (folder / candidate).exists():
                    weight_found = True
                    break
        if weight_found:
            break

    if not weight_found:
        missing.append("weight file (safetensors)")

    if missing:
        print("[validate] ERROR missing required files:", ", ".join(missing))
        sys.exit(1)
    else:
        print("[validate] Folder structure OK.")


def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="HF repo id, e.g. user/fused-model")
    ap.add_argument("--folder_path", required=True, help="Local folder with fused model")
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    ap.add_argument("--commit_message", default="Upload fused model checkpoint")
    ap.add_argument("--repo_type", choices=["model"], default="model")
    ap.add_argument("--token", default=None, help="Override token from env")
    ap.add_argument("--use-large-if-over-gb", type=float, default=4.0,
                    help="Threshold (GB): if folder size > this, switch to upload_large_folder()")
    ap.add_argument("--force-large", action="store_true", help="Force use of upload_large_folder()")
    ap.add_argument("--num-workers", type=int, default=12, help="Workers for large upload")
    ap.add_argument("--dry-run", action="store_true", help="Only scan & validate; do not upload")
    args = ap.parse_args()

    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"[error] Folder does not exist: {folder}")
        sys.exit(1)

    token = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("[warn] No token found; private repo or rate-limited operations may fail.")

    total_bytes, file_count, files = scan_folder(folder)
    size_gb = total_bytes / (1024**3)
    print(f"[scan] Found {file_count} files totaling {human_size(total_bytes)} ({size_gb:.2f} GB)")

    validate_structure(folder)

    if args.dry_run:
        print("[dry-run] Exiting before upload.")
        return

    use_large = args.force_large or (size_gb > args.use_large_if_over_gb)
    method = "upload_large_folder()" if use_large else "upload_folder()"
    print(f"[mode] Using {method}")

    # Show top 10 largest files (helpful for pruning)
    largest = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:10]
    print("[largest] Top 10 largest files:")
    for lf in largest:
        print(f"  {lf.name:50} {human_size(lf.stat().st_size)}")

    api = HfApi(token=token)

    # Create or ensure repo
    try:
        create_repo(
            repo_id=args.repo_id,
            private=args.private,
            repo_type=args.repo_type,
            exist_ok=True,
            token=token,
        )
        print(f"[repo] Ready at: https://huggingface.co/{args.repo_id}")
    except HfHubHTTPError as e:
        print(f"[repo] Failed to create/verify repo: {e}")
        sys.exit(1)

    if use_large:
        print(f"[upload] Starting resumable large upload with {args.num_workers} workers...")
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(folder),
            num_workers=args.num_workers,
            token=token,
        )
    else:
        print("[upload] Starting single-commit upload...")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(folder),
            commit_message=args.commit_message,
            token=token,
        )

    print("[done] Upload complete.")


if __name__ == "__main__":
    main()
