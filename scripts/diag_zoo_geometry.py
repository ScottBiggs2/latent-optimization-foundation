#!/usr/bin/env python3
"""
Weight-space geometry of a zoo: is it a real manifold or a ball around the trunk?

    python scripts/diag_zoo_geometry.py --zoo_dir $ARTIFACT_DIR/zoo_b030/gpt2_zoo_mini

The §4.3 gate answers "is mixture identity DETECTABLE" and nothing else. It is
maximised by a tight within-anchor spread -- which is precisely the regime where
the ensemble mean is already a good model, `gauss_codes` becomes a strong null, and
a flow can memorise the code distribution. That is misstep 19's mechanism and
RESEARCH_PLAN §11's warning, and no number in domain_separation.json sees it.

So measure the geometry instead. The quantities that matter:

  disp        ||w_i - trunk|| / ||trunk||     how far members actually moved.
              If this is ~1e-4 the zoo is the trunk with rounding on top, and
              "we generated a working model" is unfalsifiable.

  spread/disp mean ||w_i - w_j|| / mean ||w_i - trunk||
              THE diagnostic. Members that all drifted the same direction give
              ~0; members that moved independently give ~sqrt(2)=1.41. Low means
              the ensemble has one direction and the flow has nothing to learn.

  mean_frac   ||w_i - mean|| / mean ||w_i - w_j||
              sqrt((N-1)/N / 2) for an i.i.d. spread -- 0.677 at N=12, ->0.707
              as N grows. Much lower means every member sits near the centroid,
              so decoding the mean scores like a member and dPPL cannot police
              the generative arm.

  between/within   inter-group distance / intra-group distance, over the groups
              that HAVE a within (i.e. the anchors). The weight-space analogue
              of the gate's SNR, on the quantity a PCA basis is actually built
              from.

--------------------------------------------------------------------------------
HOW THIS SCALES, AND WHY IT HAD TO BE REWRITTEN FOR N=100
--------------------------------------------------------------------------------
The first version materialised `blk[i] - blk[j]` for every pair in every chunk.
At N=12 that is 66 pairs and it ran in 35 s. At N=100 it is 4950 pairs over
D=51.5M, i.e. ~255 G float64 element-ops in pure numpy -- hours, and it also
held N chunks of float64 at once (3.2 GB at chunk=4M, N=100).

Every quantity above comes out of an N x N Gram instead, in ONE streaming pass
and one BLAS call per chunk.

The Gram is taken of the DISPLACEMENTS d_i = w_i - ref, not of w_i, and that is
a numerical choice rather than a convenience. Every member descends from one
trunk, so ||w_i - w_j|| is 16-56% of ||w_i|| (measured, Phase 1). Recovering a
small difference from `G_ii + G_jj - 2 G_ij` on RAW vectors is catastrophic
cancellation; on displacements the Gram entries are already the scale of the
answer, so nothing cancels. Identities used, all exact:

    ||w_i - w_j||^2 = G_ii + G_jj - 2 G_ij                    (ref cancels)
    ||w_i - ref||^2 = G_ii
    ||w_i - mean||^2 = G_ii - (2/N) sum_j G_ij + (1/N^2) sum_jk G_jk
    ||w_i||^2       = G_ii + 2 t_i + ||ref||^2 ,  t_i = d_i . ref

Reads .npy members with mmap and streams in chunks, so peak RSS is one (N, chunk)
float64 block rather than the whole ensemble.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np


def flat_trunk(trunk_path: str, arch: str, exclude_1d: bool, include_extra: bool):
    """Flatten trunk.pt through the SAME code path that wrote the members."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from llmzoo.models.registry import build_zoo_model
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tz", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_zoo.py"))
    tz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tz)

    ck = torch.load(trunk_path, map_location="cpu", weights_only=False)
    model = build_zoo_model(arch, seed=None)
    model.load_state_dict(ck["model"])
    model.eval()
    w, _frag = tz.flatten_model(model, arch, exclude_1d=exclude_1d,
                               include_extra=include_extra)
    return w


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zoo_dir", required=True, help="the ARCH-level dir")
    p.add_argument("--chunk_bytes", type=float, default=1.5e9,
                   help="peak size of the (N, chunk) float64 working block. The "
                        "chunk length is derived from this and N, so N=12 and "
                        "N=100 both stay inside one --mem=200G job.")
    p.add_argument("--chunk", type=int, default=None,
                   help="override the derived chunk length in ELEMENTS")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    with open(os.path.join(args.zoo_dir, "zoo_meta.json")) as f:
        zmeta = json.load(f)
    arch = zmeta["arch"]
    paths = sorted(glob.glob(os.path.join(args.zoo_dir, "w_*.npy")),
                   key=lambda s: int(s.rsplit("_", 1)[1].split(".")[0]))
    N = len(paths)
    if N < 2:
        print("need >= 2 members"); return 2
    members = [np.load(p, mmap_mode="r") for p in paths]
    D = members[0].shape[0]
    idx = [int(p.rsplit("_", 1)[1].split(".")[0]) for p in paths]
    plan = {m["idx"]: m for m in zmeta["members"]}
    group = {i: plan[i]["mixture_id"] for i in idx}
    kind = {i: plan[i]["kind"] for i in idx}

    trunk = None
    tp = os.path.join(args.zoo_dir, "trunk.pt")
    if os.path.exists(tp):
        print(f"[geom] flattening {tp} …", flush=True)
        trunk = flat_trunk(tp, arch, zmeta["exclude_1d"], zmeta["include_extra"])
        if trunk.shape[0] != D:
            print(f"  trunk D={trunk.shape[0]} != member D={D}; ignoring trunk")
            trunk = None

    # Reference for the displacement Gram. The trunk when we have it (it is also
    # the thing `disp` is measured against); member 0 otherwise, which still
    # gives exact pairwise distances because the reference cancels in
    # G_ii + G_jj - 2 G_ij -- it just cannot give `disp` or `spread/disp`.
    ref = trunk if trunk is not None else np.asarray(members[0], dtype=np.float64)
    ref_name = "trunk" if trunk is not None else "member_0"

    chunk = args.chunk or max(1, int(args.chunk_bytes // (8 * N)))
    n_chunks = (D + chunk - 1) // chunk
    print(f"[geom] N={N} D={D:,} ref={ref_name}  {n_chunks} chunks of "
          f"<= {chunk:,} elems ({8 * N * chunk / 1e9:.2f} GB working block)",
          flush=True)

    G = np.zeros((N, N), dtype=np.float64)   # Gram of the displacements
    t = np.zeros(N, dtype=np.float64)        # d_i . ref
    ref_sq = 0.0

    for ci, s in enumerate(range(0, D, chunk)):
        e = min(s + chunk, D)
        r = np.asarray(ref[s:e], dtype=np.float64)
        Dm = np.empty((N, e - s), dtype=np.float64)
        for i, m in enumerate(members):
            # copy out of the memmap, then subtract in place -- one temporary,
            # not one per member per pair as the old inner loop had.
            Dm[i, :] = m[s:e]
        Dm -= r
        G += Dm @ Dm.T
        t += Dm @ r
        ref_sq += float(r @ r)
        del Dm, r
        if n_chunks <= 20 or ci % max(1, n_chunks // 10) == 0:
            print(f"  chunk {ci + 1}/{n_chunks}", flush=True)

    diag = np.diag(G).copy()
    to_ref = np.sqrt(np.clip(diag, 0.0, None))
    # ||w_i||^2 = ||d_i + ref||^2
    nrm = np.sqrt(np.clip(diag + 2.0 * t + ref_sq, 0.0, None))
    # ||w_i - mean||^2, from the Gram alone
    rowsum = G.sum(axis=1)
    total = float(G.sum())
    to_mean = np.sqrt(np.clip(diag - 2.0 * rowsum / N + total / (N * N),
                              0.0, None))

    pairs = list(itertools.combinations(range(N), 2))
    pd = {(i, j): float(np.sqrt(max(diag[i] + diag[j] - 2.0 * G[i, j], 0.0)))
          for i, j in pairs}

    # between/within over the groups that HAVE a within. Singletons each carry a
    # unique mixture_id, so counting singleton-singleton pairs as "between"
    # would silently redefine the statistic between N=12 (all anchors) and
    # N=100 (20 anchors + 80 unique singletons) and destroy comparability with
    # the Phase 1 table. Restricting to multi-member groups reproduces Phase 1
    # exactly and stays meaningful at N=100.
    sizes = {}
    for i in idx:
        sizes[group[i]] = sizes.get(group[i], 0) + 1
    multi = {g for g, c in sizes.items() if c > 1}
    within = [pd[(i, j)] for i, j in pairs
              if group[idx[i]] == group[idx[j]] and group[idx[i]] in multi]
    between = [pd[(i, j)] for i, j in pairs
               if group[idx[i]] != group[idx[j]]
               and group[idx[i]] in multi and group[idx[j]] in multi]
    allpd = list(pd.values())

    out = {
        "zoo_dir": args.zoo_dir, "arch": arch, "beta": zmeta.get("beta"),
        "n_members": N, "D": D,
        "reference": ref_name, "n_chunks": n_chunks, "chunk": chunk,
        "member_norm_mean": float(np.mean(nrm)),
        "pairwise_mean": float(np.mean(allpd)),
        "pairwise_min": float(np.min(allpd)),
        "within_group_mean": float(np.mean(within)) if within else None,
        "between_group_mean": float(np.mean(between)) if between else None,
        "between_over_within": (float(np.mean(between) / np.mean(within))
                                if within and between else None),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
        "n_multi_member_groups": len(multi),
        "n_anchor_members": sum(1 for i in idx if kind[i] == "anchor"),
        "n_singleton_members": sum(1 for i in idx if kind[i] == "singleton"),
        "to_mean_mean": float(np.mean(to_mean)),
        "mean_frac": float(np.mean(to_mean) / np.mean(allpd)),
    }
    if trunk is not None:
        out.update({
            "trunk_norm": float(np.sqrt(ref_sq)),
            "to_trunk_mean": float(np.mean(to_ref)),
            "disp_rel": float(np.mean(to_ref) / np.sqrt(ref_sq)),
            "spread_over_disp": float(np.mean(allpd) / np.mean(to_ref)),
        })

    print("\n" + "=" * 70)
    print(f"zoo geometry — {arch}  beta={out['beta']}  N={N}  D={D:,}")
    print("=" * 70)
    if trunk is not None:
        print(f"  displacement from trunk   ||w_i-trunk||/||trunk||   "
              f"{out['disp_rel']:.5f}   ({100*out['disp_rel']:.3f}% of the weights moved)")
        print(f"  spread / displacement     mean||w_i-w_j|| / mean||w_i-trunk||   "
              f"{out['spread_over_disp']:.3f}")
        print(f"      sqrt(2)=1.414 means members moved INDEPENDENTLY.")
        print(f"      near 0 means they all moved the SAME direction -> one-dimensional zoo.")
    else:
        print(f"  no trunk.pt here, so displacement is undefined; pairwise "
              f"distances are still exact (the reference cancels).")
    print(f"  centroid fraction         ||w_i-mean||/mean||w_i-w_j||   "
          f"{out['mean_frac']:.3f}   (i.i.d. spread at N={N} would be "
          f"{((N-1)/N/2)**0.5:.3f})")
    if out["between_over_within"]:
        print(f"  between/within (weights)  {out['between_over_within']:.3f}"
              f"    within {out['within_group_mean']:.2f}  between {out['between_group_mean']:.2f}")
        print(f"      over {out['n_multi_member_groups']} multi-member groups "
              f"({out['n_within_pairs']} within / {out['n_between_pairs']} between "
              f"pairs). Unique-pi singletons are excluded by construction, so "
              f"this stays comparable across N.")
    else:
        print(f"  between/within (weights)  undefined -- "
              f"{out['n_multi_member_groups']} multi-member group(s). An "
              f"all-singleton probe has no within-group pair.")
    print("=" * 70)

    dest = args.json_out or os.path.join(args.zoo_dir, "zoo_geometry.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
