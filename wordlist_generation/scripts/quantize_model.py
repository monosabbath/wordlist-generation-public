#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig

# Try to import qwen3_moe_fused
POSSIBLE_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../transformers-qwen3-moe-fused")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../transformers-qwen3-moe-fused")),
]

for p in POSSIBLE_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
    from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer
except ImportError:
    print("[ERROR] Could not import 'qwen3_moe_fused'. Please make sure the 'transformers-qwen3-moe-fused' repository is available.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Quantize a Fused Qwen3 MoE model to 4-bit and save it.")
    parser.add_argument("--input_dir", required=True, help="Path to the FUSED model directory.")
    parser.add_argument("--output_dir", required=True, help="Path to save the quantized model.")
    args = parser.parse_args()

    input_path = args.input_dir
    # Check if input is a local directory or a repo ID
    if os.path.isdir(input_path):
        input_dir = Path(input_path).resolve()
        print(f"Using local input directory: {input_dir}")
    else:
        # Assuming it's a repo ID, let transformers handle the download/cache path resolution naturally
        # unless we need the explicit path for some reason. Qwen3MoeFusedForCausalLM.from_pretrained works with repo IDs.
        input_dir = input_path
        print(f"Using HF Repo ID: {input_dir}")

    output_dir = Path(args.output_dir).resolve()

    print(f"Loading fused model from {input_dir} with 4-bit quantization...")
    print("NOTE: This requires a GPU with enough VRAM to load the model!")
    
    # Patch BNB for fused modules
    patch_bnb_quantizer()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = Qwen3MoeFusedForCausalLM.from_pretrained(
            input_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print(f"Saving quantized model to {output_dir}...")
    model.save_pretrained(output_dir)
    
    # Also copy tokenizer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"[WARN] Could not save tokenizer: {e}")

    print("Done! You can now upload the output directory.")

if __name__ == "__main__":
    main()

