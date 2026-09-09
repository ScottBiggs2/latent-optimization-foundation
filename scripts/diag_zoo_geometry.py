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
              as N grows. Much lower means every member
              sits near the centroid, so decoding the mean scores like a member
              and dPPL cannot police the generative arm.

  between/within   inter-anchor-group distance / intra-group distance.
              The weight-space analogue of the gate's SNR, on the quantity a
              PCA basis is actually built from.

Reads .npy members with mmap and streams in chunks, so peak RSS stays near one
member rather than N.
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
    p.add_argument("--chunk", type=int, default=4_000_000)
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

    trunk = None
    tp = os.path.join(args.zoo_dir, "trunk.pt")
    if os.path.exists(tp):
        print(f"[geom] flattening {tp} …", flush=True)
        trunk = flat_trunk(tp, arch, zmeta["exclude_1d"], zmeta["include_extra"])
        if trunk.shape[0] != D:
            print(f"  trunk D={trunk.shape[0]} != member D={D}; ignoring trunk")
            trunk = None

    # Streamed accumulation: pairwise sq distances, norms, and the mean.
    pairs = list(itertools.combinations(range(N), 2))
    sq = {pr: 0.0 for pr in pairs}
    nrm = [0.0] * N
    to_trunk = [0.0] * N
    tnorm = 0.0
    mean = np.zeros(D, dtype=np.float64)

    for s in range(0, D, args.chunk):
        e = min(s + args.chunk, D)
        blk = [np.asarray(m[s:e], dtype=np.float64) for m in members]
        for i, b in enumerate(blk):
            nrm[i] += float(b @ b)
            mean[s:e] += b
        if trunk is not None:
            t = np.asarray(trunk[s:e], dtype=np.float64)
            tnorm += float(t @ t)
            for i, b in enumerate(blk):
                d = b - t
                to_trunk[i] += float(d @ d)
        for i, j in pairs:
            d = blk[i] - blk[j]
            sq[(i, j)] += float(d @ d)
    mean /= N

    to_mean = [0.0] * N
    for s in range(0, D, args.chunk):
        e = min(s + args.chunk, D)
        mu = mean[s:e]
        for i, m in enumerate(members):
            d = np.asarray(m[s:e], dtype=np.float64) - mu
            to_mean[i] += float(d @ d)

    nrm = np.sqrt(nrm); to_mean = np.sqrt(to_mean)
    pd = {pr: np.sqrt(v) for pr, v in sq.items()}
    tnorm = np.sqrt(tnorm) if trunk is not None else None
    to_trunk = np.sqrt(to_trunk) if trunk is not None else None

    within = [pd[(i, j)] for i, j in pairs if group[idx[i]] == group[idx[j]]]
    between = [pd[(i, j)] for i, j in pairs if group[idx[i]] != group[idx[j]]]
    allpd = list(pd.values())

    out = {
        "zoo_dir": args.zoo_dir, "arch": arch, "beta": zmeta.get("beta"),
        "n_members": N, "D": D,
        "member_norm_mean": float(np.mean(nrm)),
        "pairwise_mean": float(np.mean(allpd)),
        "pairwise_min": float(np.min(allpd)),
        "within_group_mean": float(np.mean(within)) if within else None,
        "between_group_mean": float(np.mean(between)) if between else None,
        "between_over_within": (float(np.mean(between) / np.mean(within))
                                if within and between else None),
        "to_mean_mean": float(np.mean(to_mean)),
        "mean_frac": float(np.mean(to_mean) / np.mean(allpd)),
    }
    if trunk is not None:
        out.update({
            "trunk_norm": float(tnorm),
            "to_trunk_mean": float(np.mean(to_trunk)),
            "disp_rel": float(np.mean(to_trunk) / tnorm),
            "spread_over_disp": float(np.mean(allpd) / np.mean(to_trunk)),
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
    print(f"  centroid fraction         ||w_i-mean||/mean||w_i-w_j||   "
          f"{out['mean_frac']:.3f}   (i.i.d. spread at N={N} would be "
          f"{((N-1)/N/2)**0.5:.3f})")
    if out["between_over_within"]:
        print(f"  between/within (weights)  {out['between_over_within']:.3f}"
              f"    within {out['within_group_mean']:.2f}  between {out['between_group_mean']:.2f}")
    print("=" * 70)

    dest = args.json_out or os.path.join(args.zoo_dir, "zoo_geometry.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
