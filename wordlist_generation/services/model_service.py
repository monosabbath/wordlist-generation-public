import json
import logging
from typing import Any, Dict

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    pipeline,
)

logger = logging.getLogger("model-service")


def _build_tokenizer_from_repo_files(model_id: str, padding_side: str):
    """
    Fallback tokenizer builder for models whose tokenizer class is not exposed by Transformers.
    It constructs a PreTrainedTokenizerFast from tokenizer.json and wires in chat_template and specials
    from tokenizer_config.json so apply_chat_template works as expected.
    """
    # Load tokenizer config for chat template and special tokens
    chat_template = None
    bos_token = None
    eos_token = None
    pad_token = None
    unk_token = None
    additional_special_tokens = None

    try:
        tok_cfg_path = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        chat_template = tok_cfg.get("chat_template")
        bos_token = tok_cfg.get("bos_token")
        eos_token = tok_cfg.get("eos_token")
        pad_token = tok_cfg.get("pad_token")
        unk_token = tok_cfg.get("unk_token")
        additional_special_tokens = tok_cfg.get("additional_special_tokens")
    except Exception as e:
        logger.warning(f"Could not load tokenizer_config.json for {model_id}: {e}. Proceeding without it.")

    # Load the fast tokenizer JSON (required)
    try:
        tok_json_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
    except Exception as e:
        raise RuntimeError(
            f"Could not download tokenizer.json from {model_id}. "
            f"If this model requires a custom tokenizer, install mistral-common and use remote code. "
            f"Original error: {e}"
        )

    # Build a fast tokenizer directly from the json file
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tok_json_path,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
    )
    if additional_special_tokens:
        try:
            tok.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        except Exception:
            # Ignore if already present or incompatible
            pass

    # Wire in chat template if provided so apply_chat_template works
    if chat_template:
        try:
            tok.chat_template = chat_template
        except Exception as e:
            logger.warning(f"Could not set chat_template on tokenizer: {e}")

    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Niceties for downstream code
    tok.name_or_path = model_id
    return tok


class ModelService:
    def __init__(self, model, tokenizer, text_pipeline, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.text_pipeline = text_pipeline
        self.settings = settings

    @property
    def is_cohere_reasoning_model(self) -> bool:
        # Keep original behavior (used only to optionally allow Cohere control tokens in prefix fn).
        return self.settings.MODEL_NAME.strip().lower() == "coherelabs/command-a-reasoning-08-2025"

    @classmethod
    def from_settings(cls, s):
        # Torch backend knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # Dtype selection
        dtype: Any = "auto"
        ts = s.TORCH_DTYPE.lower()
        if ts in ("bf16", "bfloat16", "torch.bfloat16"):
            dtype = torch.bfloat16
            logger.info("Using dtype: bfloat16")
        elif ts in ("fp16", "float16", "torch.float16"):
            dtype = torch.float16
            logger.info("Using dtype: float16")
        else:
            logger.info("Using dtype: auto")

        init_kwargs: Dict[str, Any] = {
            "trust_remote_code": s.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "local_files_only": False,
            "device_map": s.DEVICE_MAP,
        }
        if dtype != "auto":
            init_kwargs["torch_dtype"] = dtype

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (trust_remote_code={s.TRUST_REMOTE_CODE}, "
            f"attn='{s.ATTN_IMPLEMENTATION}', device_map='{s.DEVICE_MAP}')"
        )

        # Inspect config to route Mistral3 correctly
        cfg = AutoConfig.from_pretrained(s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False)
        model_type = getattr(cfg, "model_type", "").lower()
        cfg_name = cfg.__class__.__name__

        # Load model
        if model_type == "mistral3" or "Mistral3Config" in cfg_name:
            try:
                from transformers import Mistral3ForConditionalGeneration  # type: ignore

                try:
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        s.MODEL_NAME,
                        attn_implementation=s.ATTN_IMPLEMENTATION,
                        **init_kwargs,
                    )
                except TypeError:
                    logger.warning("Model.from_pretrained() did not accept attn_implementation; retrying without it.")
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        s.MODEL_NAME,
                        **init_kwargs,
                    )
                    try:
                        model.set_attention_implementation(s.ATTN_IMPLEMENTATION)
                    except Exception as e2:
                        logger.warning(f"Could not set attention implementation: {e2}. Trying SDPA.")
                        try:
                            model.set_attention_implementation("sdpa")
                        except Exception as e3:
                            logger.warning(f"Could not set SDPA either: {e3}")
            except ImportError as e:
                raise RuntimeError(
                    "Your Transformers version does not expose Mistral3ForConditionalGeneration. "
                    "Please upgrade: pip install -U 'transformers>=4.46.2' 'mistral-common>=1.6.2'"
                ) from e
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    attn_implementation=s.ATTN_IMPLEMENTATION,
                    **init_kwargs,
                )
            except TypeError:
                logger.warning("Model.from_pretrained() did not accept attn_implementation; retrying without it.")
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    **init_kwargs,
                )
                try:
                    model.set_attention_implementation(s.ATTN_IMPLEMENTATION)
                except Exception as e:
                    logger.warning(f"Could not set attention implementation: {e}. Trying SDPA.")
                    try:
                        model.set_attention_implementation("sdpa")
                    except Exception as e2:
                        logger.warning(f"Could not set SDPA either: {e2}")

        # Load tokenizer with robust fallbacks
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                s.MODEL_NAME,
                trust_remote_code=s.TRUST_REMOTE_CODE,
                local_files_only=False,
            )
        except KeyError as e:
            # Typical: KeyError on Mistral3Config when TOKENIZER_MAPPING lacks an entry
            logger.info(f"AutoTokenizer failed with {e}; falling back to repo files for tokenizer.")
            tokenizer = _build_tokenizer_from_repo_files(s.MODEL_NAME, s.TOKENIZER_PADDING_SIDE)
        except Exception as e:
            logger.warning(f"AutoTokenizer raised {e}; attempting repo-file fallback.")
            tokenizer = _build_tokenizer_from_repo_files(s.MODEL_NAME, s.TOKENIZER_PADDING_SIDE)

        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=s.TRUST_REMOTE_CODE,
        )

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
