# Qwen3 MoE Fused - Inference-only module
# Minimal subset for loading and running fused Qwen3 MoE models with Triton grouped GEMM kernels

from .modular_qwen3_moe_fused import (
    MoeFusedLinear,
    Qwen3MoeFusedDecoderLayer,
    Qwen3MoeFusedForCausalLM,
    Qwen3MoeFusedModel,
    Qwen3MoeFusedSparseMoeBlock,
)
from .quantize.quantizer import patch_bnb_quantizer

__all__ = [
    "MoeFusedLinear",
    "Qwen3MoeFusedDecoderLayer",
    "Qwen3MoeFusedForCausalLM",
    "Qwen3MoeFusedModel",
    "Qwen3MoeFusedSparseMoeBlock",
    "patch_bnb_quantizer",
]

