#!/usr/bin/env python3
"""
Upload a (fused) checkpoint folder to the Hugging Face Hub.

Usage:
  # Option 1: Provide token via .env (preferred keys: HUGGINGFACE_HUB_TOKEN or HF_TOKEN)
  python wordlist_generation/scripts/upload_to_hf.py \
      --repo_id monosabbath/fused-cydonia-24b \
      --folder_path ./fused-cydonia-24b \
      --private

  # Option 2: Pass token explicitly
  python wordlist_generation/scripts/upload_to_hf.py \
      --repo_id monosabbath/fused-cydonia-24b \
      --folder_path ./fused-cydonia-24b \
      --private \
      --token hf_xxx...

Notes:
- Ensure you have git-lfs installed if you also push via git elsewhere.
- This script uses the Hugging Face Hub API directly (no port usage).
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    # Load .env from the nearest location (repo root or above)
    load_dotenv(find_dotenv(usecwd=True), override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="e.g. monosabbath/fused-cydonia-24b")
    ap.add_argument("--folder_path", required=True, help="Local folder to upload (checkpoint dir)")
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    ap.add_argument("--commit_message", default="Upload fused checkpoint")
    ap.add_argument("--repo_type", choices=["model", "dataset", "space"], default="model")
    ap.add_argument("--token", default=None, help="HF token (overrides env). If omitted, uses env.")
    ap.add_argument("--allow_patterns", nargs="*", default=None, help="Optional allow patterns")
    ap.add_argument("--ignore_patterns", nargs="*", default=None, help="Optional ignore patterns")
    args = ap.parse_args()

    # Resolve token precedence: CLI > HUGGINGFACE_HUB_TOKEN > HF_TOKEN
    env_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    token = args.token or env_token

    if token:
        print("[upload] Using token from CLI/env.")
    else:
        print("[upload] WARNING: No token provided. Public repos may work, private will fail.")

    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"[upload] Folder does not exist or is not a directory: {folder}")

    print(f"[upload] Preparing to upload: {folder}")
    print(f"[upload] Target repo: {args.repo_id} (type={args.repo_type}, private={args.private})")

    api = HfApi(token=token)

    # Create repo if needed
    create_repo(
        repo_id=args.repo_id,
        private=args.private,
        repo_type=args.repo_type,
        exist_ok=True,
        token=token,
    )
    print(f"[upload] Repo is ready: https://huggingface.co/{args.repo_id}")

    # Upload folder (LFS is handled automatically for large files)
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(folder),
        commit_message=args.commit_message,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        token=token,
    )
    print(f"[upload] Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
