import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_folder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="e.g. monosabbath/fused-cydonia-24b")
    ap.add_argument("--folder_path", required=True, help="Local fused checkpoint folder")
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    ap.add_argument("--commit_message", default="Upload fused checkpoint")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", None)
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    create_repo(repo_id=args.repo_id, private=args.private, repo_type="model", exist_ok=True, token=token)

    # Upload entire folder (will use LFS for large files)
    upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.folder_path,
        commit_message=args.commit_message,
        allow_patterns=None,  # or specify a list to include only certain files
    )
    print(f"Uploaded {args.folder_path} to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
