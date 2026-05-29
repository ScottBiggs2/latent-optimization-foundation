"""
Block reconstruction quality metrics.

evaluate_block()    — single block: PCA code MSE, cosine sim on raw weights
evaluate_all()      — all blocks in dataset: global + per-family breakdown
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from data.block_dataset import BlockDataset
from dual_pca import BatchedCovariancePCA
from vae import ConditionedBlockVAE


# ---------------------------------------------------------------------------
# Single-block evaluation
# ---------------------------------------------------------------------------

def evaluate_block(
    original_flat: np.ndarray,    # (max_block_size,) padded
    mask: np.ndarray,             # (max_block_size,) bool
    original_code: np.ndarray,    # (k,) PCA code of the original
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    block_idx: int,
    family_idx: int,
    device: str = "cpu",
) -> dict:
    """
    Round-trip evaluation for one block.

    original_flat → PCA code → VAE encode → VAE decode → inverse PCA → recon_flat
    Metrics computed on the real (non-padded) portion only.
    """
    vae.eval()
    dev = torch.device(device)

    # VAE round-trip on PCA codes
    with torch.no_grad():
        code_t     = torch.from_numpy(original_code).float().unsqueeze(0).to(dev)
        bidx_t     = torch.tensor([block_idx],  dtype=torch.long, device=dev)
        fidx_t     = torch.tensor([family_idx], dtype=torch.long, device=dev)
        recon_code, mu, logvar = vae(code_t, bidx_t, fidx_t)

    recon_code_np = recon_code.squeeze(0).cpu().numpy()

    # Inverse PCA: (1, k) → (1, max_block_size)
    recon_padded = pca.inverse_transform(recon_code_np.reshape(1, -1)).squeeze(0)

    # Extract real (unpadded) portions
    real = original_flat[mask.astype(bool)]
    recon_real = recon_padded[mask.astype(bool)]

    # Metrics on real weights
    cos_sim = float(
        np.dot(real, recon_real)
        / (np.linalg.norm(real) * np.linalg.norm(recon_real) + 1e-30)
    )
    mse = float(np.mean((real - recon_real) ** 2))

    # KL divergence on weight histograms
    def _kl(a, b, bins=50):
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
        pa = pa + 1e-12;  pb = pb + 1e-12
        pa /= pa.sum();   pb /= pb.sum()
        return float(np.sum(pa * np.log(pa / pb)))

    kl = _kl(real, recon_real)

    # Code-space MSE
    code_mse = float(np.mean((original_code - recon_code_np) ** 2))

    return {
        "cosine_sim": cos_sim,
        "mse": mse,
        "kl_divergence": kl,
        "code_mse": code_mse,
        "n_real_params": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Full-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    dataset: BlockDataset,
    codes: torch.Tensor,          # (N, k) PCA codes
    block_idxs: torch.Tensor,     # (N,)
    family_idxs: torch.Tensor,    # (N,)
    device: str = "cpu",
) -> dict:
    """
    Evaluate reconstruction quality for every block in dataset.

    Returns a dict with:
      global        : aggregated metrics (mean cosine_sim, mse, kl)
      per_family    : per-arch mean metrics
      per_block     : full list of per-block dicts
    """
    vae.eval()
    codes_np   = codes.numpy()
    bidxs_np   = block_idxs.numpy()
    fidxs_np   = family_idxs.numpy()

    per_block: List[dict] = []

    for i in range(len(dataset)):
        flat   = dataset._blocks[i].copy()
        mask   = dataset._masks[i].copy()
        code   = codes_np[i]
        bidx   = int(bidxs_np[i])
        fidx   = int(fidxs_np[i])
        arch   = dataset.get_arch(i)

        result = evaluate_block(flat, mask, code, pca, vae, bidx, fidx, device)
        result["arch"]       = arch
        result["block_idx"]  = bidx
        result["family_idx"] = fidx
        per_block.append(result)

    # Global aggregates
    def mean_key(key):
        return float(np.mean([r[key] for r in per_block]))

    global_metrics = {
        "cosine_sim":   mean_key("cosine_sim"),
        "mse":          mean_key("mse"),
        "kl_divergence": mean_key("kl_divergence"),
        "code_mse":     mean_key("code_mse"),
        "n_blocks":     len(per_block),
    }

    # Per-family aggregates
    from models.registry import list_archs
    per_family: Dict[str, dict] = {}
    for arch in dataset.arch_list:
        arch_results = [r for r in per_block if r["arch"] == arch]
        if arch_results:
            per_family[arch] = {
                "cosine_sim":    float(np.mean([r["cosine_sim"]    for r in arch_results])),
                "mse":           float(np.mean([r["mse"]           for r in arch_results])),
                "kl_divergence": float(np.mean([r["kl_divergence"] for r in arch_results])),
                "n_blocks":      len(arch_results),
            }

    return {
        "global":     global_metrics,
        "per_family": per_family,
        "per_block":  per_block,
    }
