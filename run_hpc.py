"""
HPC runner for Explorer (SLURM / A100/V100).

Sets HPC-specific defaults:
  - artifact_dir = /scratch/biggs.s/llm_vae
  - HF_HOME      = /scratch/biggs.s/hf_cache
  - LM eval enabled by default
  - Timestamped logging throughout

Invoked by slurm_train.sh.  Can also be run interactively on a compute node:
    python run_hpc.py
    python run_hpc.py --eval_lm --mode full
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Set HPC scratch paths BEFORE importing transformers
_HPC_SCRATCH     = "/scratch/biggs.s"
_HPC_HF_CACHE    = os.path.join(_HPC_SCRATCH, "hf_cache")
_HPC_ARTIFACT    = os.path.join(_HPC_SCRATCH, "llm_vae")

os.environ.setdefault("HF_HOME",        _HPC_HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", _HPC_HF_CACHE)
# Prevent triton / torch from writing to home
os.environ.setdefault("TRITON_CACHE_DIR",
                       os.path.join(_HPC_SCRATCH, "triton_cache"))

from models.registry import list_archs


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


def main():
    p = argparse.ArgumentParser(description="LLM-VAE HPC runner")

    # Default list excludes gated models (gemma3_270m requires HF_TOKEN + access).
    # To include Gemma 3: add --arch_list gpt2_medium smollm2_360m gemma3_270m opt_350m
    _DEFAULT_ARCHS = ["gpt2_medium", "smollm2_360m", "qwen3_0_6b", "opt_350m"]
    p.add_argument("--arch_list",  nargs="+", default=_DEFAULT_ARCHS)
    p.add_argument("--mode",       choices=["tiny", "full"], default="full")
    p.add_argument("--noise_scale", type=float, default=1e-7)

    # PCA
    p.add_argument("--n_components",  type=int, default=97)
    p.add_argument("--pca_batch_size", type=int, default=10,
                   help="Streaming batch size for PCA (10×12M×4B=480MB RAM)")

    # VAE
    p.add_argument("--latent_dim",    type=int,   default=32)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--cond_dim",      type=int,   default=64)
    p.add_argument("--epochs",        type=int,   default=500)
    p.add_argument("--patience",      type=int,   default=50)
    p.add_argument("--warmup_epochs", type=int,   default=50)
    p.add_argument("--beta",          type=float, default=1.0)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--batch_size",    type=int,   default=32)

    # Paths
    p.add_argument("--artifact_dir", type=str, default=_HPC_ARTIFACT)

    # Resume / force flags
    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--force_pca",     action="store_true")
    p.add_argument("--force_encode",  action="store_true")
    p.add_argument("--force_train",   action="store_true")

    # Evaluation
    p.add_argument("--eval_lm",          action="store_true", default=True)
    p.add_argument("--no_eval_lm",       dest="eval_lm", action="store_false")
    p.add_argument("--eval_seq_len",     type=int, default=512)
    p.add_argument("--eval_n_sequences", type=int, default=32)

    args = p.parse_args()

    # Ensure scratch dirs exist
    for d in [args.artifact_dir, _HPC_HF_CACHE]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{ts()} LLM-VAE HPC Pipeline")
    print(f"  artifact_dir : {args.artifact_dir}")
    print(f"  HF_HOME      : {os.environ.get('HF_HOME')}")
    print(f"  arch_list    : {args.arch_list}")
    print(f"  mode         : {args.mode}")
    print(f"{'='*60}\n")

    import json
    import torch
    from data.block_dataset import BlockDataset
    from dual_pca import BatchedCovariancePCA
    from evaluate import evaluate_all
    from vae import ConditionedBlockVAE
    import train as T

    pca_dir    = os.path.join(args.artifact_dir, "pca")
    vae_dir    = os.path.join(args.artifact_dir, "vae")
    res_dir    = os.path.join(args.artifact_dir, "results")
    blocks_dir = os.path.join(args.artifact_dir, "blocks")
    for d in [pca_dir, vae_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"{ts()} Stage 1: Extracting blocks …")
    dataset = T.stage_extract(args, args.arch_list, blocks_dir)
    print(f"{ts()} Stage 1 complete: {len(dataset)} blocks × {dataset.max_block_size:,} params")

    print(f"\n{ts()} Stage 2: Fitting PCA …")
    pca = T.stage_pca(args, dataset, pca_dir)
    print(f"{ts()} Stage 2 complete: {pca.n_components} components, "
          f"{np.sum(pca.explained_variance_ratio_):.4%} variance")

    print(f"\n{ts()} Stage 3: Encoding blocks …")
    codes, bidxs, fidxs = T.stage_encode(args, dataset, pca, vae_dir)

    print(f"\n{ts()} Stage 4: Training VAE …")
    vae = T.train_vae(args, codes, bidxs, fidxs, vae_dir)

    print(f"\n{ts()} Stage 5: Evaluating reconstruction …")
    device = str(next(vae.parameters()).device)
    results = evaluate_all(pca, vae, dataset, codes, bidxs, fidxs, device=device)
    res_path = os.path.join(res_dir, "reconstruction_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  global cosine_sim = {results['global']['cosine_sim']:.6f}")
    print(f"  global mse        = {results['global']['mse']:.3e}")
    for arch, fm in results["per_family"].items():
        print(f"  {arch}: cosine_sim={fm['cosine_sim']:.6f}  mse={fm['mse']:.3e}")

    if args.eval_lm:
        print(f"\n{ts()} Stage 6: LM perplexity evaluation …")
        from eval_lm import evaluate_all_families
        lm_results = evaluate_all_families(
            pca, vae, dataset,
            seq_len=args.eval_seq_len,
            n_sequences=args.eval_n_sequences,
            mode=args.mode,
            artifact_dir=args.artifact_dir,
        )
        lm_path = os.path.join(res_dir, "lm_eval_results.json")
        with open(lm_path, "w") as f:
            json.dump(lm_results, f, indent=2)
        print(f"\n  LM evaluation summary:")
        for arch, res in lm_results.items():
            print(f"  {arch}: PPL {res['original_ppl']:.2f} → "
                  f"{res['reconstructed_ppl']:.2f}  "
                  f"(Δ={res['ppl_delta']:+.3f}, {res['ppl_delta_pct']:+.2f}%)")
        print(f"  Saved → {lm_path}")

    print(f"\n{ts()} All done. Results in {res_dir}")


if __name__ == "__main__":
    main()
