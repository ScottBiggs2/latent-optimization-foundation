#!/usr/bin/env python3
"""
Block-only vs whole-stack spectrum: the §4.5 embedding-share confound, measured.

    python scripts/diag_block_spectrum.py --zoo_dir $ARTIFACT_DIR/zoo_b030/gpt2_zoo_mini

WHY THIS EXISTS
---------------
RESEARCH_PLAN §4.7 asks whether the effective rank of the weight distribution
grows with D or stays flat, across Mini / Small / Medium. §4.5 then flags the
confound that makes a naive answer unpublishable: with the vocab fixed at 50257,
the EMBEDDING share of D falls 51.0% -> 31.6% -> 14.8% along the ladder. The
COMPOSITION of D changes, not only its size, so a trend in the whole-stack
spectrum could be a trend in how much of D is an embedding table.

§4.5's own remedy is to report the block-only spectrum beside the whole-stack
one. Nothing could do that: `spectrum_stats` was buried in scripts/train_stack.py
and every fit was whole-stack.

WHY IT NEEDS NO PCA, NO GPU AND NO TORCH
----------------------------------------
The bulk statistics depend only on the eigenvalues of the centred Gram, and a
Gram is a SUM OVER COORDINATES. So it splits exactly along the layout

    w = [ extra (embeddings + final LN) | block_0 | block_1 | ... ]

    C_whole = C_extra + C_block

and one streaming pass over each region gives all three spectra -- whole-stack,
block-only, and embeddings-only, which is the confound itself rather than just a
control for it. Costs one read of the ensemble and an 11x11 eigh.

Two implementation notes that are correctness, not tuning:

  * The Gram is taken of DISPLACEMENTS w_i - w_0, not of w_i. Members share a
    trunk, so ||w_i - w_j|| is a fraction of ||w_i||; centring a raw Gram means
    recovering a small number from large ones. Exactly equivalent, because
    w_i - mean(w) == d_i - mean(d) for any fixed reference.

  * Centring is applied to the Gram in closed form, C = P G P with
    P = I - 11'/N, rather than by a separate mean pass. DualGramPCA takes its
    mean pass in FLOAT32 (gram.py: `mean = np.empty(D, dtype=np.float32)`), so
    this route is very slightly MORE accurate than the fitted pipeline -- expect
    agreement with pipeline_summary to ~1e-6, not to 1e-14.

Chunking here is local to this script and does not touch
`EnsembleDataset.chunk_bounds`, which must stay a fixed grid (misstep 10). The
eigenvalues are chunking-invariant because the Gram is a sum.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from llmzoo.pca.gram import spectrum_stats            # noqa: E402


def region_gram(members, lo: int, hi: int, N: int, chunk: int,
                label: str) -> np.ndarray:
    """Uncentred Gram of (w_i - w_0) restricted to coordinates [lo, hi)."""
    G = np.zeros((N, N), dtype=np.float64)
    if hi <= lo:
        return G
    n_chunks = (hi - lo + chunk - 1) // chunk
    for ci, s in enumerate(range(lo, hi, chunk)):
        e = min(s + chunk, hi)
        Dm = np.empty((N, e - s), dtype=np.float64)
        for i, m in enumerate(members):
            Dm[i, :] = m[s:e]
        Dm -= Dm[0].copy()
        G += Dm @ Dm.T
        del Dm
        if n_chunks <= 8 or ci % max(1, n_chunks // 8) == 0:
            print(f"  {label}: chunk {ci + 1}/{n_chunks}", flush=True)
    return G


def centred(G: np.ndarray) -> np.ndarray:
    """C = P G P with P = I - 11'/N. Exact, and independent of the reference."""
    N = G.shape[0]
    r = G.mean(axis=1, keepdims=True)
    c = G.mean(axis=0, keepdims=True)
    return G - r - c + G.mean()


def evals_desc(C: np.ndarray) -> np.ndarray:
    ev = np.linalg.eigh(C)[0][::-1]
    return np.clip(ev, 0.0, None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zoo_dir", required=True, help="the ARCH-level dir")
    p.add_argument("--k", type=int, default=None, help="default N-1")
    p.add_argument("--chunk_bytes", type=float, default=1.5e9)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    with open(os.path.join(args.zoo_dir, "zoo_meta.json")) as f:
        z = json.load(f)
    paths = sorted(glob.glob(os.path.join(args.zoo_dir, "w_*.npy")),
                   key=lambda s: int(s.rsplit("_", 1)[1].split(".")[0]))
    N = len(paths)
    if N < 3:
        print("need >= 3 members for a usable rank"); return 2
    members = [np.load(q, mmap_mode="r") for q in paths]
    D = int(members[0].shape[0])
    extra = int(z.get("extra_size", 0))
    if D != int(z["n_params"]):
        print(f"member D={D:,} but zoo_meta n_params={int(z['n_params']):,}")
        return 2
    k = args.k if args.k is not None else N - 1
    chunk = max(1, int(args.chunk_bytes // (8 * N)))

    print("=" * 74)
    print(f"block-only vs whole-stack spectrum — {z['arch']}  beta={z.get('beta')}")
    print(f"  N={N}  D={D:,}  k={k}")
    print(f"  extra (embeddings + final LN) : {extra:,}  "
          f"({100.0 * extra / D:.1f}% of D)")
    print(f"  blocks ({z.get('n_layers')} x {int(z.get('block_size', 0)):,}) : "
          f"{D - extra:,}  ({100.0 * (D - extra) / D:.1f}% of D)")
    print("=" * 74)

    G_extra = region_gram(members, 0, extra, N, chunk, "extra ")
    G_block = region_gram(members, extra, D, N, chunk, "blocks")

    regions = {
        "whole_stack": G_extra + G_block,
        "block_only": G_block,
        "embeddings_only": G_extra,
    }
    out = {
        "zoo_dir": args.zoo_dir, "arch": z["arch"], "beta": z.get("beta"),
        "n_members": N, "k": k, "D": D, "extra_size": extra,
        "block_size": int(z.get("block_size", 0)),
        "n_layers": z.get("n_layers"),
        "embedding_fraction_of_D": extra / D,
        "regions": {},
    }
    tr_whole = float(np.trace(centred(regions["whole_stack"])))
    for name, G in regions.items():
        C = centred(G)
        ev = evals_desc(C)
        st = spectrum_stats(ev, k)
        tr = float(np.trace(C))
        out["regions"][name] = {
            **{f"spectrum_{a}": b for a, b in st.items()},
            "trace": tr,
            "variance_share_of_whole": tr / tr_whole if tr_whole else None,
            "evals": ev[:k].tolist(),
        }

    print(f"\n  {'region':<18} {'ev0/median':>11} {'eff_rank_ratio':>15} "
          f"{'eff_rank':>9} {'var share':>10}")
    for name in ("whole_stack", "block_only", "embeddings_only"):
        r = out["regions"][name]
        print(f"  {name:<18} {r['spectrum_ev0_over_median']:>11.3f} "
              f"{r['spectrum_effective_rank_ratio']:>15.4f} "
              f"{r['spectrum_effective_rank']:>9.3f} "
              f"{100 * (r['variance_share_of_whole'] or 0):>9.1f}%")
    print(f"\n  'var share' is each region's share of the total centred variance,")
    print(f"  which is the confound in one number: the embedding block is "
          f"{100.0 * extra / D:.1f}% of D")
    print(f"  and carries "
          f"{100 * (out['regions']['embeddings_only']['variance_share_of_whole'] or 0):.1f}% "
          f"of the variance.")
    print(f"\n  Gate on ev0/median and eff_rank_ratio. NOT on ev0/ev[k-1], which")
    print(f"  reads 12.09 at k=N-1 on data whose true ratio is 1.01 (misstep 15b).")

    dest = args.json_out or os.path.join(args.zoo_dir, "block_spectrum.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
