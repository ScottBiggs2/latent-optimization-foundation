"""
Does a generative arm produce samples with the RIGHT SPREAD, or has it collapsed?

Why this exists
---------------
`eval_stack.py` gates every arm on dPPL, which is correct for reconstruction arms and
INSUFFICIENT for generative ones. On a manufactured ensemble `mean(w) = w_0 +
O(s*sigma/sqrt(N))`, so the ensemble mean IS essentially the real pretrained model. A
generator that has collapsed toward its per-family mean therefore scores a dPPL near
ZERO -- better than an honest sample -- while generating nothing at all.

That is exactly the trap misstep 13 describes in a different guise: a metric that
ranks a degenerate output above a faithful one. `gauss_codes` was added as the null
model for whether a flow learned STRUCTURE, but it cannot detect collapse, because a
collapsed flow beats it on dPPL by construction.

So measure dispersion in code space, where it is cheap and unambiguous. No ensemble,
no PCA, no inverse_transform -- this reads the 40 KB codes artifact and the sealed
flows.

    python scripts/diag_flow_dispersion.py --run_name emb3 --k 99
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from llmzoo.artifacts.bundle import load_run


def stats(x: torch.Tensor, name: str) -> dict:
    """
    x is (n, dim) in NORMALIZED code space, where the training target has RMS ~1.

    `rms` is the headline: the per-family normalization makes the real codes land at
    RMS ~1, so anything much below 1 is shrinkage toward the mean.

    `mean_pairwise` guards the case rms cannot see: a generator that emits ONE
    off-centre point repeatedly has healthy rms and zero diversity.
    """
    n = x.shape[0]
    rms = float(x.pow(2).mean().sqrt())
    per_dim = x.std(dim=0, unbiased=(n > 1))
    centre = float(x.mean(dim=0).pow(2).mean().sqrt())
    if n > 1:
        d = torch.cdist(x, x)
        pw = float(d[~torch.eye(n, dtype=torch.bool, device=x.device)].mean())
    else:
        pw = float("nan")
    return {"arm": name, "n": n, "rms": rms,
            "per_dim_std_mean": float(per_dim.mean()),
            "per_dim_std_min": float(per_dim.min()),
            "per_dim_std_max": float(per_dim.max()),
            "centre_offset_rms": centre,
            "mean_pairwise_dist": pw}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--run_name", default="emb3")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--flow_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    # No "dataset"/"pca": dispersion lives in code space, so none of the expensive
    # machinery is needed.
    bundle = load_run(run_root, args.k, device="cpu",
                      want=("codes", "vae", "flow"))
    cs = bundle.code_stats
    codes = torch.from_numpy(cs.codes)
    fidx = torch.from_numpy(cs.family_idxs)

    print("=" * 78)
    print(f"Flow dispersion diagnostic — run={args.run_name} k={args.k}")
    print("Everything below is in NORMALIZED code space, where the real codes")
    print("sit at RMS ~1 by construction. rms << 1 means collapse toward the mean.")
    print("=" * 78)

    out = {}
    for arch, f in sorted(cs.arch_to_family.items()):
        if not bool((fidx == f).any()):
            continue
        g = torch.Generator().manual_seed(args.seed)
        real = cs.normalize(codes[fidx == f], fidx[fidx == f])
        n = args.n_samples
        fv = torch.full((n,), int(f), dtype=torch.long)

        rows = [stats(real, "real_codes")]

        # gauss_codes, exactly as the eval arm draws it: N(mean_f, std_f) in RAW
        # code space, which is N(0, 1) once normalized.
        rows.append(stats(torch.randn(n, cs.k, generator=g), "gauss_codes"))

        for space in ("codes", "latent"):
            fm = bundle.flows.get(space)
            if fm is None:
                continue
            gg = torch.Generator().manual_seed(args.seed)
            raw = fm.sample_codes(fv, n_steps=args.flow_steps, generator=gg)
            rows.append(stats(cs.normalize(raw, fv), f"flow_{space}"))

        if bundle.vae is not None:
            gg = torch.Generator().manual_seed(args.seed)
            z = torch.randn(n, bundle.vae.latent_dim, generator=gg)
            with torch.no_grad():
                raw = bundle.vae.decode_cfg(z, fv, guidance_scale=1.0)
            rows.append(stats(cs.normalize(raw, fv), "generate"))

        print(f"\n=== {arch} (family {f}) ===")
        print(f"{'arm':14s} {'n':>5s} {'rms':>8s} {'perdim':>8s} "
              f"{'centre':>8s} {'pairwise':>9s}   {'verdict':s}")
        base = rows[0]["rms"]
        for r in rows:
            ratio = r["rms"] / max(base, 1e-12)
            if r["arm"] in ("real_codes", "gauss_codes"):
                verdict = "reference"
            elif ratio < 0.5:
                verdict = f"COLLAPSED ({ratio:.2f}x real)"
            elif ratio < 0.8:
                verdict = f"shrunken ({ratio:.2f}x real)"
            elif ratio > 1.25:
                verdict = f"over-dispersed ({ratio:.2f}x real)"
            else:
                verdict = f"ok ({ratio:.2f}x real)"
            print(f"{r['arm']:14s} {r['n']:5d} {r['rms']:8.4f} "
                  f"{r['per_dim_std_mean']:8.4f} {r['centre_offset_rms']:8.4f} "
                  f"{r['mean_pairwise_dist']:9.4f}   {verdict}")
        out[arch] = rows

    print("\n" + "=" * 78)
    print("HOW TO READ THIS. A collapsed generator scores a dPPL near zero on this")
    print("ensemble because the ensemble mean is essentially w_0, so a LOW dPPL from")
    print("a generative arm is only meaningful if rms is also ~1. Check this table")
    print("before reading anything into flow_codes vs gauss_codes.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"run": args.run_name, "k": args.k,
                       "n_samples": args.n_samples, "per_arch": out}, fh, indent=2)
        print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
