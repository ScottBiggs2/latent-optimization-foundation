"""
Local demo / smoke-test runner.

Two modes:
  --mode tiny  (default) — random-init models, no downloads, runs in ~60s.
                           Tests full pipeline: extraction → PCA → VAE → eval.
  --mode full  —          Downloads all 4 pretrained models (~8-12 GB total).
                           Requires enough disk and RAM.

Usage
-----
    python run_demo.py                          # tiny smoke test
    python run_demo.py --mode tiny --eval_lm    # tiny + LM eval
    python run_demo.py --mode full              # full pipeline, local artifacts
    python run_demo.py --mode full --eval_lm    # full + LM eval
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from models.registry import list_archs
import train


def main():
    p = argparse.ArgumentParser(
        description="LLM-VAE local demo (tiny or full mode)"
    )
    p.add_argument("--mode", choices=["tiny", "full"], default="tiny")
    p.add_argument("--arch_list", nargs="+", default=list_archs())
    p.add_argument("--eval_lm", action="store_true")

    # Sensible demo defaults (override-able)
    p.add_argument("--n_components",  type=int,   default=10,
                   help="PCA components (tiny: capped by N_blocks)")
    p.add_argument("--latent_dim",    type=int,   default=8)
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--cond_dim",      type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--patience",      type=int,   default=30)
    p.add_argument("--warmup_epochs", type=int,   default=20)
    p.add_argument("--beta",          type=float, default=1.0)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--noise_scale",   type=float, default=1e-7)
    p.add_argument("--pca_batch_size",type=int,   default=4)
    p.add_argument("--eval_seq_len",  type=int,   default=64)
    p.add_argument("--eval_n_sequences", type=int, default=4)
    p.add_argument("--artifact_dir",  type=str,   default="")

    # Force flags
    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--force_pca",     action="store_true")
    p.add_argument("--force_encode",  action="store_true")
    p.add_argument("--force_train",   action="store_true")

    args = p.parse_args()

    # For tiny/demo mode, put artifacts in a local temp directory
    if not args.artifact_dir:
        if args.mode == "tiny":
            args.artifact_dir = os.path.join(tempfile.gettempdir(), "llm_vae_demo")
        else:
            args.artifact_dir = "./artifacts_local"
    os.makedirs(args.artifact_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LLM-VAE Demo — mode={args.mode}")
    print(f"artifact_dir: {args.artifact_dir}")
    print(f"{'='*60}\n")

    _run_pipeline(args)


def _run_pipeline(args):
    """Invoke train.py's pipeline stages directly."""
    import json
    import torch

    from data.block_dataset import BlockDataset
    from dual_pca import BatchedCovariancePCA
    from evaluate import evaluate_all
    from models.registry import MAX_BLOCKS, N_FAMILIES
    from vae import ConditionedBlockVAE

    pca_dir = os.path.join(args.artifact_dir, "pca")
    vae_dir = os.path.join(args.artifact_dir, "vae")
    res_dir = os.path.join(args.artifact_dir, "results")
    for d in [pca_dir, vae_dir, res_dir]:
        os.makedirs(d, exist_ok=True)
    blocks_dir = os.path.join(args.artifact_dir, "blocks")

    import train as T
    dataset      = T.stage_extract(args, args.arch_list, blocks_dir)
    pca          = T.stage_pca(args, dataset, pca_dir)
    codes, bidxs, fidxs = T.stage_encode(args, dataset, pca, vae_dir)
    vae          = T.train_vae(args, codes, bidxs, fidxs, vae_dir)

    results = evaluate_all(pca, vae, dataset, codes, bidxs, fidxs,
                           device=str(next(vae.parameters()).device))
    res_path = os.path.join(res_dir, "reconstruction_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] global cosine_sim = {results['global']['cosine_sim']:.6f}")
    print(f"[Results] global mse        = {results['global']['mse']:.3e}")
    print(f"[Results] Saved → {res_path}")

    if args.eval_lm:
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
        print(f"[LM eval] Saved → {lm_path}")
        for arch, res in lm_results.items():
            print(f"  {arch}: {res['original_ppl']:.2f} → {res['reconstructed_ppl']:.2f} "
                  f"(Δ={res['ppl_delta']:+.3f})")


if __name__ == "__main__":
    main()
