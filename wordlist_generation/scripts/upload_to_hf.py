#!/usr/bin/env python3
"""
Upload a (fused) checkpoint folder to the Hugging Face Hub.

- Loads .env (HUGGINGFACE_HUB_TOKEN / HF_TOKEN)
- Uses upload_large_folder() when available and selected, else falls back to upload_folder()
- Version-agnostic import for HfHubHTTPError
- Resumable uploads if your huggingface_hub is recent enough

Usage:
  python wordlist_generation/scripts/upload_to_hf.py \
      --repo_id <user>/<repo> \
      --folder_path /path/to/fused-model \
      --private \
      --force-large \
      --num-workers 12
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# Robust imports across huggingface_hub versions
try:
    from huggingface_hub import HfApi, create_repo, upload_folder, upload_large_folder  # type: ignore
    HAS_TOPLEVEL_LARGE = True
except Exception:
    from huggingface_hub import HfApi, create_repo, upload_folder  # type: ignore
    HAS_TOPLEVEL_LARGE = False

# HfHubHTTPError import compatibility
try:
    from huggingface_hub import HfHubHTTPError  # new-style
except Exception:
    try:
        from huggingface_hub.utils._errors import HfHubHTTPError  # older versions
    except Exception:
        class HfHubHTTPError(Exception):  # fallback
            pass


def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="e.g. username/fused-model")
    ap.add_argument("--folder_path", required=True, help="Local folder to upload")
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    ap.add_argument("--commit_message", default="Upload fused checkpoint")
    ap.add_argument("--repo_type", choices=["model", "dataset", "space"], default="model")
    ap.add_argument("--token", default=None, help="HF token (overrides env). If omitted, uses env.")
    ap.add_argument("--force-large", action="store_true", help="Force resumable upload_large_folder() when available")
    ap.add_argument("--num-workers", type=int, default=12, help="Workers for large upload")
    args = ap.parse_args()

    # Resolve token precedence: CLI > HUGGINGFACE_HUB_TOKEN > HF_TOKEN
    env_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    token = args.token or env_token

    if token:
        print("[upload] Using token from CLI/env.")
    else:
        print("[upload] WARNING: No token provided. Private repos will fail.")

    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"[upload] Folder does not exist or is not a directory: {folder}")

    print(f"[upload] Folder: {folder}")
    print(f"[upload] Target: {args.repo_id} (type={args.repo_type}, private={args.private})")

    api = HfApi(token=token)

    # Create repo if needed
    try:
        create_repo(
            repo_id=args.repo_id,
            private=args.private,
            repo_type=args.repo_type,
            exist_ok=True,
            token=token,
        )
        print(f"[upload] Repo is ready: https://huggingface.co/{args.repo_id}")
    except HfHubHTTPError as e:
        print(f"[upload] Failed to create/verify repo: {e}")
        raise

    # Decide upload method
    can_large = HAS_TOPLEVEL_LARGE or hasattr(api, "upload_large_folder")
    if args.force_large and not can_large:
        print("[upload] WARNING: upload_large_folder() not available in your huggingface_hub version.")
        print("         Run: pip install -U 'huggingface_hub>=0.32.0'")
    use_large = args.force_large and can_large

    if use_large:
        print(f"[upload] Starting resumable large upload with {args.num_workers} workers...")
        # Call either top-level or instance method depending on availability
        if HAS_TOPLEVEL_LARGE:
            upload_large_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(folder),
                num_workers=args.num_workers,
                token=token,
            )
        else:
            api.upload_large_folder(  # type: ignore[attr-defined]
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(folder),
                num_workers=args.num_workers,
                token=token,
            )
    else:
        print("[upload] Starting single-commit upload...")
        upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(folder),
            commit_message=args.commit_message,
            token=token,
        )

    print("[upload] Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
