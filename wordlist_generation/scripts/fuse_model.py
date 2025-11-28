#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

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
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    print(f"Fusing model from {input_dir} to {output_dir}...")
    convert_model_to_fused(input_dir, output_dir)
    print("Done! You can now upload the output directory using upload_to_hf.py.")

if __name__ == "__main__":
    main()

