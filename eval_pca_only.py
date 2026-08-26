"""
PCA-only reconstruction evaluation — the VAE-free control.

Why this exists
---------------
Every other number in this repo measures PCA and VAE *jointly*, so when a family
reconstructs badly there is no way to tell which stage is responsible. The
2026-08-21 six-family run produced a clean bimodal split that nothing in the
pipeline explains:

    gpt2_medium    cos 0.99899   PPL     26.69 ->      6372
    smollm2_360m   cos 0.99978   PPL     14.67 ->        15.01
    smollm2_135m   cos 0.99935   PPL     19.35 ->        25.62
    qwen3_0_6b     cos 0.58642   PPL     26.25 ->  10539438
    pythia_160m    cos 0.61731   PPL     38.81 -> 485165195   (= e^20, clamped)
    pythia_410m    cos 0.37483   PPL     22.30 ->    126673

The split follows the architecture family and matches neither padding fraction
nor block count nor block size.

This script runs the pipeline with the VAE removed:

    block -> pad -> PCA project -> PCA inverse -> strip padding -> write back

so it isolates the shared PCA basis. Read the result as:

  * bad here too       -> the shared PCA basis is the problem. Try --exclude_1d
                          (norm gains + biases pollute the basis), or a
                          per-family PCA.
  * fine here, bad with the VAE -> the VAE is the problem. Look at capacity, the
                          code normalization, or posterior collapse.

Usage
-----
    python eval_pca_only.py --artifact_dir /scratch/biggs.s/llm_vae
    python eval_pca_only.py --n_components 40 20 10     # rank sweep
    python eval_pca_only.py --arch_list pythia_410m --no_wandb

Reuses BatchedCovariancePCA.components_/mean_ and weight_extractor.reconstruct_block
unchanged; nothing here re-fits a PCA.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from dual_pca import BatchedCovariancePCA
from models.registry import get_arch_config, get_layers, load_model
from models.weight_extractor import extract_block_flat, pad_block, reconstruct_block
import wandb_utils as wb


# ---------------------------------------------------------------------------
# The ablation itself
# ---------------------------------------------------------------------------

def pca_roundtrip_model_blocks(
    model: nn.Module,
    arch: str,
    pca: BatchedCovariancePCA,
    max_block_size: int,
    device: torch.device,
    exclude_1d: bool = False,
    n_components: Optional[int] = None,
) -> dict:
    """
    Replace every transformer block with its PCA projection/inverse round trip.
    No VAE. Modifies model in-place.

    n_components truncates the basis further than it was fitted, which is what
    makes the rank sweep possible without re-fitting: the components are
    orthogonal and ordered by decreasing variance, so keeping a prefix is exactly
    a lower-rank PCA.

    Returns per-block cosine similarity / MSE on the real (non-padded) weights,
    so this is directly comparable to evaluate.evaluate_all's numbers.
    """
    layers = get_layers(model, arch)

    comps = pca.components_
    if n_components is not None:
        comps = comps[:n_components]
    mean = pca.mean_

    cos_sims, mses = [], []
    for layer in layers:
        flat, schema = extract_block_flat(layer, exclude_1d=exclude_1d)
        n_real = len(flat)

        padded, _mask = pad_block(flat, max_block_size)

        # Project then invert, in float32. Same arithmetic as
        # BatchedCovariancePCA.transform + inverse_transform, done for a single
        # row so we never materialise an (N, n_params) intermediate.
        centered = padded - mean
        code = centered @ comps.T                 # (k,)
        recon_padded = (code @ comps) + mean      # (max_block_size,)
        recon_flat = recon_padded[:n_real]

        denom = (np.linalg.norm(flat) * np.linalg.norm(recon_flat)) + 1e-30
        cos_sims.append(float(np.dot(flat, recon_flat) / denom))
        mses.append(float(np.mean((flat - recon_flat) ** 2)))

        reconstruct_block(recon_flat.astype(np.float32), layer, schema)

    return {
        "cosine_sim": float(np.mean(cos_sims)),
        "cosine_sim_min": float(np.min(cos_sims)),
        "mse": float(np.mean(mses)),
        "per_block_cosine_sim": cos_sims,
        "n_blocks": len(cos_sims),
    }


def evaluate_family_pca_only(
    arch: str,
    pca: BatchedCovariancePCA,
    max_block_size: int,
    seq_len: int = 1024,
    n_sequences: int = 64,
    hf_cache: Optional[str] = None,
    exclude_1d: bool = False,
    n_components: Optional[int] = None,
) -> dict:
    """Before/after PPL for one family with the VAE removed from the loop."""
    from eval_lm import compute_perplexity
    from data.val_loader import get_wikitext2_loader
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_arch_config(arch)

    model = load_model(arch, cache_dir=hf_cache).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["default_model_id"], cache_dir=hf_cache, trust_remote_code=True)
    dataloader = get_wikitext2_loader(
        tokenizer, seq_len=seq_len, n_sequences=n_sequences, cache_dir=hf_cache)

    print(f"  [{arch}] Measuring original PPL …")
    orig = compute_perplexity(model, dataloader, device)
    print(f"  [{arch}] Original PPL = {orig['perplexity']:.3f}")

    k = n_components if n_components is not None else pca.components_.shape[0]
    print(f"  [{arch}] PCA round-trip (k={k}, exclude_1d={exclude_1d}) …")
    recon_metrics = pca_roundtrip_model_blocks(
        model, arch, pca, max_block_size, device,
        exclude_1d=exclude_1d, n_components=n_components)

    print(f"  [{arch}] Measuring PCA-only PPL …")
    recon = compute_perplexity(model, dataloader, device)
    delta = recon["perplexity"] - orig["perplexity"]
    print(f"  [{arch}] PCA-only PPL = {recon['perplexity']:.3f}  (Δ={delta:+.3f})  "
          f"cos={recon_metrics['cosine_sim']:.5f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "arch": arch,
        "n_components": k,
        "exclude_1d": exclude_1d,
        "original_ppl": orig["perplexity"],
        "pca_only_ppl": recon["perplexity"],
        "ppl_delta": delta,
        "ppl_delta_pct": 100.0 * delta / max(orig["perplexity"], 1e-9),
        "ce_original": orig["ce_loss"],
        "ce_pca_only": recon["ce_loss"],
        **recon_metrics,
    }
    wb.log({
        f"pca_only/{arch}/k{k}/ppl_delta": delta,
        f"pca_only/{arch}/k{k}/pca_only_ppl": recon["perplexity"],
        f"pca_only/{arch}/k{k}/cosine_sim": recon_metrics["cosine_sim"],
    })
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="PCA-only (VAE-free) reconstruction + PPL evaluation. "
                    "Separates 'the shared PCA basis is wrong' from 'the VAE is "
                    "wrong' for the families that reconstruct badly.")
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--pca_dir", default=None, help="Defaults to <artifact_dir>/pca")
    p.add_argument("--arch_list", nargs="+", default=None,
                   help="Defaults to the arch_list recorded in dataset_meta.json")
    p.add_argument("--n_components", nargs="+", type=int, default=None,
                   help="Rank sweep. Each value truncates the fitted basis to that "
                        "many components. Default: the full fitted basis.")
    p.add_argument("--exclude_1d", action="store_true", default=None,
                   help="Override the dataset's exclude_1d setting. Must match how "
                        "the PCA was fitted or the codes are meaningless.")
    p.add_argument("--eval_seq_len", type=int, default=1024)
    p.add_argument("--eval_n_sequences", type=int, default=64)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    hf_cache = os.environ.get("HF_HOME", os.path.join(args.artifact_dir, "hf_cache"))
    res_dir = os.path.join(args.artifact_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    with open(os.path.join(args.artifact_dir, "blocks", "dataset_meta.json")) as f:
        meta = json.load(f)

    arch_list = args.arch_list or meta["arch_list"]
    max_block_size = meta["max_block_size"]
    exclude_1d = (bool(meta.get("exclude_1d", False))
                  if args.exclude_1d is None else args.exclude_1d)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca = BatchedCovariancePCA.load(
        args.pca_dir or os.path.join(args.artifact_dir, "pca"), device=device)

    k_fitted = pca.components_.shape[0]
    k_list = args.n_components or [k_fitted]
    for k in k_list:
        if k > k_fitted:
            raise ValueError(f"--n_components {k} exceeds the fitted basis size "
                             f"({k_fitted}). Re-fit the PCA to go higher.")

    wb.init_run(
        job_type="eval_pca_only",
        config={**vars(args), "exclude_1d_effective": exclude_1d,
                "k_fitted": k_fitted, "max_block_size": max_block_size},
        tags=["pca_only"] + arch_list,
        enabled=not args.no_wandb,
        artifact_dir=args.artifact_dir,
    )

    print(f"\n{'='*64}")
    print("PCA-only reconstruction eval (no VAE)")
    print(f"  arch_list      : {arch_list}")
    print(f"  max_block_size : {max_block_size:,}")
    print(f"  basis fitted   : {k_fitted} components of {meta['total_blocks'] - 1} max")
    print(f"  rank sweep     : {k_list}")
    print(f"  exclude_1d     : {exclude_1d}")
    print(f"{'='*64}\n")

    results: dict = {}
    for k in k_list:
        for arch in arch_list:
            print(f"=== {arch}  (k={k}) ===")
            results[f"{arch}@k{k}"] = evaluate_family_pca_only(
                arch, pca, max_block_size,
                seq_len=args.eval_seq_len,
                n_sequences=args.eval_n_sequences,
                hf_cache=hf_cache,
                exclude_1d=exclude_1d,
                n_components=(None if k == k_fitted else k),
            )

    out_path = os.path.join(res_dir, "pca_only_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  PCA-only summary (VAE removed):")
    print(f"  {'arch@k':28s} {'cos_sim':>9s} {'PPL orig':>10s} {'PPL pca':>12s} {'Δ%':>12s}")
    for key, r in results.items():
        print(f"  {key:28s} {r['cosine_sim']:9.5f} {r['original_ppl']:10.2f} "
              f"{r['pca_only_ppl']:12.2f} {r['ppl_delta_pct']:+12.2f}")
    print(f"\n  Saved → {out_path}")
    wb.finish()


if __name__ == "__main__":
    main()
