#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load env vars for HF_TOKEN if available
load_dotenv(find_dotenv(usecwd=True))

# Try to import qwen3_moe_fused
# Assuming it's in a sibling directory or installed
POSSIBLE_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../transformers-qwen3-moe-fused")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../transformers-qwen3-moe-fused")),
]

for p in POSSIBLE_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    from qwen3_moe_fused.convert import convert_model_to_fused
except ImportError:
    print("[ERROR] Could not import 'qwen3_moe_fused'. Please make sure the 'transformers-qwen3-moe-fused' repository is available.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Fuse Qwen3 MoE experts for faster inference.")
    parser.add_argument("--input_dir", required=True, help="Path to the input model directory (HF format).")
    parser.add_argument("--output_dir", required=True, help="Path to save the fused model.")
    parser.add_argument("--cache_dir", help="Optional path to cache downloaded models (e.g. /tmp for container disk).")
    args = parser.parse_args()

    input_path = args.input_dir
    # Check if input is a local directory or a repo ID
    if os.path.isdir(input_path):
        input_dir = Path(input_path).resolve()
        print(f"Using local input directory: {input_dir}")
    else:
        # Try to download from HF
        print(f"Input '{input_path}' is not a local directory. Attempting to download from HF Hub...")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        try:
            from huggingface_hub import snapshot_download
            
            download_kwargs = {
                "repo_id": input_path,
                "token": token,
                "cache_dir": args.cache_dir,
            }
            
            print(f"Downloading model '{input_path}' from HF Hub...")
            
            input_dir = Path(snapshot_download(**download_kwargs)).resolve()
            print(f"Model downloaded to: {input_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to download model '{input_path}': {e}")
            sys.exit(1)

    output_dir = Path(args.output_dir).resolve()

    print(f"Fusing model from {input_dir} to {output_dir}...")
    convert_model_to_fused(input_dir, output_dir)
    
    # Copy tokenizer and generation config
    print(f"Copying tokenizer and generation config to {output_dir}...")
    try:
        from transformers import AutoTokenizer, GenerationConfig
        
        # Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            print("Tokenizer saved.")
        except Exception as e:
            print(f"[WARN] Failed to save tokenizer: {e}")

        # Generation Config
        try:
            gen_config = GenerationConfig.from_pretrained(input_dir, trust_remote_code=True)
            gen_config.save_pretrained(output_dir)
            print("Generation config saved.")
        except Exception as e:
            print(f"[WARN] Failed to save generation config: {e}")
            
    except ImportError:
         print("[WARN] Transformers not installed or could not import AutoTokenizer/GenerationConfig. Skipping auxiliary file copy.")

    print("Done! You can now upload the output directory using upload_to_hf.py.")

if __name__ == "__main__":
    main()
