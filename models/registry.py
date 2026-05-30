"""
Architecture registry for LLM-VAE block-wise weight experiments.

Supports four small transformer-decoder models of similar scale:
  gpt2_medium    — openai-community/gpt2-medium    (~355M)
  smollm2_360m   — HuggingFaceTB/SmolLM2-360M     (~360M)
  gemma3_270m    — google/gemma-3-270m             (~270M)
  opt_350m       — facebook/opt-350m               (~350M)

Each entry records the HuggingFace model identifier, how to navigate to the
transformer layer list, which integer family_idx to assign for the conditioning
embedding, and a tiny_config for random-init local smoke-testing (no download).

Block extraction does NOT use architecture-specific parameter name lists —
it calls block.named_parameters() directly (see weight_extractor.py).  The
registry only needs to know how to reach the layer list.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------

ARCH_CONFIGS: dict[str, dict] = {
    "gpt2_medium": {
        # ---- HuggingFace identifiers ----
        "default_model_id": "openai-community/gpt2-medium",
        "hf_model_type":    "gpt2",
        # ---- Model tree navigation ----
        # GPT2LMHeadModel: blocks live at model.transformer.h
        "layers_attr":      "transformer.h",
        # ---- Family index for conditioning embedding ----
        "family_idx":       0,
        # ---- Tiny mock config (random-init, no download) ----
        # GPT-2 uses n_embd / n_head / n_layer naming (not hidden_size)
        "tiny_config": dict(
            n_embd=128,
            n_layer=2,
            n_head=4,
            n_inner=512,        # 4 × n_embd; None uses 4× default
            vocab_size=1000,
            n_positions=512,
            n_ctx=512,
        ),
    },

    "smollm2_360m": {
        # LlamaForCausalLM variant (SmolLM2 uses LLaMA-2 architecture)
        "default_model_id": "HuggingFaceTB/SmolLM2-360M",
        "hf_model_type":    "llama",
        "layers_attr":      "model.layers",
        "family_idx":       1,
        "tiny_config": dict(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,     # GQA: 2 KV heads
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        ),
    },

    "qwen3_0_6b": {
        # Qwen3-0.6B — open weights, already cached on Explorer HPC.
        # LLaMA-style decoder; identical layer layout to smollm2_360m.
        # Replaces gemma3_270m at family_idx=2 (Gemma 3 is gated and requires
        # a HF_TOKEN + access approval; kept below as "gemma3_270m" for future use).
        "default_model_id": "Qwen/Qwen3-0.6B",
        "hf_model_type":    "qwen3",
        "layers_attr":      "model.layers",
        "family_idx":       2,
        # Real config: 28 layers, hidden=1024, heads=16, kv_heads=8, ffn=3072, head_dim=128
        "tiny_config": dict(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,               # Qwen3 requires explicit head_dim
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        ),
    },

    # ── Gated models (require HF_TOKEN + HuggingFace access approval) ────────
    "gemma3_270m": {
        # Gemma 3 (Google) — multimodal; decoder blocks at model.language_model.layers
        # Access: https://huggingface.co/google/gemma-3-270m
        # To use: set HF_TOKEN in slurm_train.sh and add to arch_list.
        "default_model_id": "google/gemma-3-270m",
        "hf_model_type":    "gemma3",
        "layers_attr":      "model.language_model.layers",
        "family_idx":       4,         # set to 4 when used alongside the default 4
        "tiny_config": dict(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
        ),
    },

    "opt_350m": {
        # OPTForCausalLM — pre-norm, biases on all projections
        "default_model_id": "facebook/opt-350m",
        "hf_model_type":    "opt",
        # OPT layers live at model.model.decoder.layers
        "layers_attr":      "model.decoder.layers",
        "family_idx":       3,
        "tiny_config": dict(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            ffn_dim=512,
            word_embed_proj_dim=128,
            vocab_size=1000,
            max_position_embeddings=512,
        ),
    },
}

N_FAMILIES = len(ARCH_CONFIGS)

# Maximum block index across all architectures (for embedding table sizing)
MAX_BLOCKS = max(
    cfg.get("n_layers_hint", 40)        # conservative upper bound
    for cfg in ARCH_CONFIGS.values()
)
# Actual upper bound: SmolLM2-360M has 32 layers, so 40 is a safe ceiling.
MAX_BLOCKS = 40


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def list_archs() -> list[str]:
    return list(ARCH_CONFIGS.keys())


def get_arch_config(arch: str) -> dict:
    if arch not in ARCH_CONFIGS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCH_CONFIGS)}")
    return ARCH_CONFIGS[arch]


def load_model(
    arch: str,
    model_id: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> nn.Module:
    """
    Load a full pretrained model from HuggingFace.

    Uses float32 by default so block extractions are lossless.
    Models are loaded with low_cpu_mem_usage=True to avoid OOM during loading.
    """
    from transformers import AutoModelForCausalLM

    cfg = get_arch_config(arch)
    mid = model_id or cfg["default_model_id"]
    hf_token = token or os.environ.get("HF_TOKEN")
    hf_cache = cache_dir or os.environ.get("HF_HOME", "/scratch/biggs.s/hf_cache")

    print(f"Loading {mid} (arch={arch}, dtype={dtype}) …")
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=dtype,
        token=hf_token,
        low_cpu_mem_usage=True,
        cache_dir=hf_cache,
        trust_remote_code=True,   # needed for Gemma3 and some SmolLM variants
    )
    model.eval()
    return model


def build_tiny_model(arch: str) -> nn.Module:
    """
    Construct a randomly-initialised model with the correct architecture class
    but at a reduced scale (hidden=128, 2 layers).  No download needed.
    Useful for local pipeline testing without any internet access.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = get_arch_config(arch)
    tiny = cfg["tiny_config"].copy()
    tiny["model_type"] = cfg["hf_model_type"]

    hf_cfg = AutoConfig.for_model(**tiny)
    if hasattr(hf_cfg, "tie_word_embeddings"):
        hf_cfg.tie_word_embeddings = False

    model = AutoModelForCausalLM.from_config(hf_cfg)
    model.eval()
    return model


def get_layers(model: nn.Module, arch: str):
    """Return the ModuleList of transformer decoder blocks."""
    cfg = get_arch_config(arch)
    obj = model
    for attr in cfg["layers_attr"].split("."):
        obj = getattr(obj, attr)
    return obj
