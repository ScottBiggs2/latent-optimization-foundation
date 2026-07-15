"""
Language model perplexity evaluation for reconstructed models.

For each model family:
  1. Load the pretrained model
  2. Reconstruct each transformer block through PCA + VAE
  3. Load reconstructed weights back into the model
  4. Measure cross-entropy / perplexity on WikiText-2
  5. Compare to original model PPL

Uses WikiText-2 test split (full mode) or a synthetic token sequence (tiny mode).
"""

from __future__ import annotations

import gc
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from data.block_dataset import BlockDataset
from dual_pca import BatchedCovariancePCA
from models.registry import get_arch_config, get_layers, list_archs, load_model, build_tiny_model
from models.weight_extractor import reconstruct_block, extract_block_flat, pad_block
from vae import ConditionedBlockVAE


# ---------------------------------------------------------------------------
# Perplexity measurement
# ---------------------------------------------------------------------------

# Need to expand with more metrics in the future. Check how bedio did it (screenshot)
# Add: MMLUU, GPQA, HellaSwag
# Don't bother with vLLM fuckery - just use torch/transformers for simplicity. 
# Make sure that the chat templates are correct!

def compute_perplexity(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Measure cross-entropy and perplexity over a DataLoader.

    Returns
    -------
    dict with keys: perplexity, ce_loss, n_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            # Shift labels: standard causal-LM cross-entropy
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            # Number of predicted tokens: seq_len - 1 per sequence
            n_tok = (input_ids.shape[0] * (input_ids.shape[1] - 1))
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok

    if total_tokens == 0:
        return {"perplexity": float("nan"), "ce_loss": float("nan"), "n_tokens": 0}

    ce = total_loss / total_tokens
    # Cap at e^20 to avoid overflow on degenerate reconstructions
    ppl = math.exp(min(ce, 20.0))
    return {"perplexity": ppl, "ce_loss": ce, "n_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Reconstruct one model's blocks through PCA + VAE
# ---------------------------------------------------------------------------

def reconstruct_model_blocks(
    model: nn.Module,
    arch: str,
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    max_block_size: int,
    device: torch.device,
) -> None:
    """
    Replace every transformer block's weights with their VAE reconstruction.
    Modifies model in-place.
    """
    cfg     = get_arch_config(arch)
    layers  = get_layers(model, arch)
    family_idx = int(cfg["family_idx"])
    vae.eval()

    for i, layer in enumerate(layers):
        flat, schema = extract_block_flat(layer)
        n_real = len(flat)

        # Pad to max_block_size
        padded, _mask = pad_block(flat, max_block_size)

        # PCA encode: (1, max_block_size) → (1, k)
        padded_t = padded.reshape(1, -1)
        mean_centered = padded_t - pca.mean_
        code = (mean_centered @ pca.components_.T)  # (1, k)

        # VAE round-trip
        with torch.no_grad():
            code_t  = torch.from_numpy(code).float().to(device)
            bidx_t  = torch.tensor([i],          dtype=torch.long, device=device)
            fidx_t  = torch.tensor([family_idx], dtype=torch.long, device=device)
            recon_code, _, _ = vae(code_t, bidx_t, fidx_t)

        recon_code_np = recon_code.cpu().numpy()

        # Inverse PCA: (1, k) → (1, max_block_size)
        recon_padded = pca.inverse_transform(recon_code_np).squeeze(0)

        # Strip padding
        recon_flat = recon_padded[:n_real]

        # Load back into layer
        reconstruct_block(recon_flat, layer, schema)


# ---------------------------------------------------------------------------
# Per-family evaluation
# ---------------------------------------------------------------------------

def evaluate_family(
    arch: str,
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    max_block_size: int,
    mode: str = "full",
    seq_len: int = 512,
    n_sequences: int = 16,
    hf_cache: Optional[str] = None,
) -> dict:
    """
    Full before/after PPL evaluation for one model family.

    Returns dict with original_ppl, reconstructed_ppl, ppl_delta, etc.
    """
    device = torch.device(
        next(vae.parameters()).device if hasattr(vae, "block_idx_emb") else "cpu"
    )

    # Load model
    if mode == "tiny":
        model = build_tiny_model(arch)
    else:
        model = load_model(arch, cache_dir=hf_cache)
    model = model.to(device)
    model.eval()

    # Build dataloader
    if mode == "tiny":
        from data.val_loader import get_synthetic_loader
        cfg = get_arch_config(arch)
        vocab_size = cfg["tiny_config"].get(
            "vocab_size",
            cfg["tiny_config"].get("n_positions", 1000)
        )
        dataloader = get_synthetic_loader(
            vocab_size=vocab_size, seq_len=seq_len, n_sequences=n_sequences
        )
    else:
        from data.val_loader import get_wikitext2_loader
        from transformers import AutoTokenizer
        cfg = get_arch_config(arch)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["default_model_id"],
            cache_dir=hf_cache,
            trust_remote_code=True,
        )
        dataloader = get_wikitext2_loader(
            tokenizer, seq_len=seq_len, n_sequences=n_sequences, cache_dir=hf_cache
        )

    # Measure original PPL
    print(f"  [{arch}] Measuring original PPL …")
    orig_result = compute_perplexity(model, dataloader, device)
    original_ppl = orig_result["perplexity"]
    print(f"  [{arch}] Original PPL = {original_ppl:.3f}")

    # Reconstruct all blocks
    print(f"  [{arch}] Reconstructing blocks …")
    reconstruct_model_blocks(model, arch, pca, vae, max_block_size, device)

    # Measure reconstructed PPL
    print(f"  [{arch}] Measuring reconstructed PPL …")
    recon_result = compute_perplexity(model, dataloader, device)
    reconstructed_ppl = recon_result["perplexity"]
    ppl_delta = reconstructed_ppl - original_ppl
    print(f"  [{arch}] Reconstructed PPL = {reconstructed_ppl:.3f}  "
          f"(Δ = {ppl_delta:+.3f})")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "arch":             arch,
        "original_ppl":     original_ppl,
        "reconstructed_ppl": reconstructed_ppl,
        "ppl_delta":        ppl_delta,
        "ppl_delta_pct":    100.0 * ppl_delta / max(original_ppl, 1e-9),
        "ce_original":      orig_result["ce_loss"],
        "ce_reconstructed": recon_result["ce_loss"],
        "n_tokens":         orig_result["n_tokens"],
    }


# ---------------------------------------------------------------------------
# All families
# ---------------------------------------------------------------------------

def evaluate_all_families(
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    dataset: BlockDataset,
    seq_len: int = 512,
    n_sequences: int = 16,
    mode: str = "full",
    artifact_dir: str = "/scratch/biggs.s/llm_vae",
) -> dict:
    """Run LM eval for every arch in dataset.arch_list."""
    hf_cache = None if mode == "tiny" else artifact_dir + "/hf_cache"
    results = {}
    for arch in dataset.arch_list:
        results[arch] = evaluate_family(
            arch, pca, vae,
            max_block_size=dataset.max_block_size,
            mode=mode,
            seq_len=seq_len,
            n_sequences=n_sequences,
            hf_cache=hf_cache,
        )
    return results
