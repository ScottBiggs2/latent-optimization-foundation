"""
One-off diagnostic: re-fit the OLD shared six-family block PCA with the corrected
numerics (eigh + float64 Gram + fp32 storage) and write it to a separate directory.

Why: the 2026-08-21 basis was fit with randomized_svd at k=97 of n=150 and stored as
float16. Both are now known to be unreliable. pythia_410m reconstructed at cosine
0.38825 through that basis. If that number moves materially once the numerics are
fixed, part of the failure was arithmetic rather than a genuine statement about
whether a shared basis can represent a GPT-NeoX block.

Reads blocks/all_blocks.npy directly rather than going through BlockDataset, so it
does not trip the layout_version gate (the old artifacts predate it).

    python diag_refit_shared_pca.py --artifact_dir /scratch/biggs.s/llm_vae \
        --out_dir /scratch/biggs.s/llm_vae/runs/2026-08-21-shared-6family/pca_eigh_refit
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dual_pca import BatchedCovariancePCA


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_components", type=int, default=97)
    p.add_argument("--batch_size", type=int, default=25)
    args = p.parse_args()

    blocks_dir = os.path.join(args.artifact_dir, "blocks")
    with open(os.path.join(blocks_dir, "dataset_meta.json")) as f:
        meta = json.load(f)

    N = int(meta["total_blocks"])
    D = int(meta["max_block_size"])
    print(f"blocks: N={N}, D={D:,}  arch_list={meta['arch_list']}")
    print(f"exclude_1d={meta.get('exclude_1d')}  layout_version={meta.get('layout_version')}")

    blocks = np.memmap(os.path.join(blocks_dir, "all_blocks.npy"),
                       dtype=np.float32, mode="r", shape=(N, D))

    def loader(s: int, e: int) -> np.ndarray:
        return np.ascontiguousarray(blocks[s:e, :].T)   # (D, batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca = BatchedCovariancePCA(n_components=args.n_components, device=device)
    pca.fit(loader, n_models=N, batch_size=args.batch_size)

    os.makedirs(args.out_dir, exist_ok=True)
    pca.save(args.out_dir)

    # Side-by-side against the original basis, if it is still on disk.
    old_dir = os.path.join(args.artifact_dir, "pca")
    old_ev_path = os.path.join(old_dir, "explained_variance.npy")
    print("\n=== spectrum comparison ===")
    print(f"new (eigh)  : kept {pca.n_components} components")
    ev_new = pca.explained_variance_
    print(f"  ev[0]={ev_new[0]:.6g}  ev[-1]={ev_new[-1]:.6g}  "
          f"ratio={ev_new[0] / max(ev_new[-1], 1e-30):.4g}")
    print(f"  total variance captured: {np.sum(pca.explained_variance_ratio_):.6%}")
    if os.path.exists(old_ev_path):
        ev_old = np.load(old_ev_path)
        print(f"old (rsvd)  : {len(ev_old)} components")
        print(f"  ev[0]={ev_old[0]:.6g}  ev[-1]={ev_old[-1]:.6g}  "
              f"ratio={ev_old[0] / max(ev_old[-1], 1e-30):.4g}")
        k = min(len(ev_old), len(ev_new))
        rel = np.abs(ev_new[:k] - ev_old[:k]) / (ev_old[0] + 1e-30)
        print(f"  max relative eigenvalue difference (vs lead): {rel.max():.4g}")
        worst = int(np.argmax(rel))
        print(f"  worst at component {worst}: new={ev_new[worst]:.6g} old={ev_old[worst]:.6g}")

    print(f"\nSaved refit basis -> {args.out_dir}")
    print("Next: eval_pca_only.py --pca_dir <that dir> to get per-family cosine/PPL.")


if __name__ == "__main__":
    main()
