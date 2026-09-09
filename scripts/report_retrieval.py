#!/usr/bin/env python3
"""
RESEARCH_PLAN §6.4's retrieval baseline: the arm nobody in this literature reports.

    python scripts/report_retrieval.py                      # both beta arms
    python scripts/report_retrieval.py --arms 0.15
    python scripts/report_retrieval.py --data_dir reports/data/n100   # laptop mode
    python scripts/report_retrieval.py --gen flow=<...>/domain_separation.json

PURE STDLIB, exactly like scripts/report_stack.py and report_singleton_probe.py: no
numpy, no torch, no llmzoo import. This is the control that decides whether any
generative number in the paper means anything, so it must never be the reason a
report needs a GPU -- and it needs no new measurement at all, because every trained
model's per-domain perplexity is already sealed in domain_separation.json.

THE QUESTION
------------
For a held-out mixture pi, is the generated model better than "just use the trained
model whose mixture is nearest to pi"? If not, the flow retrieved rather than
generalised. It is `gauss_codes` one level up, it is the objection a reviewer raises
unprompted, and at RESEARCH_PLAN §11's memorisation regime -- 100 members in a
~4.5-dimensional manifold -- it is load-bearing rather than a nicety.

THE SCALAR
----------
    L(model, pi) = sum_d  pi_d * ln ppl[model][d]

The mixture-weighted mean log perplexity: the training objective's own weighting of
the domains, so a model is scored on what it was ASKED to be good at. Log space
because perplexity composes multiplicatively, which makes the advantage additive and
comparable across domains whose absolute PPL differs by an order of magnitude --
the same reasoning as report_singleton_probe.py's §6.5 statistic.

Lower is better. Reported as differences, never as absolutes.

THE RESOLUTION, AND WHY IT IS PRINTED NEXT TO EVERY NUMBER
----------------------------------------------------------
The four branches of each anchor share one pi and differ only by data order, so the
spread of L across them is this instrument's noise floor. Measured at Phase 2 Mini it
is sigma ~ 0.007 nats pooled, and up to 0.035 in the worst per-domain cell.

That matters more than usual here, because the sealed interior holdout turns out to
be SATURATED: mixtures.holdout_split picks the four singletons nearest the
barycentre, which is the densest region of the simplex, so the nearest trained
neighbour sits 0.17-0.34 in L1 and scores within +-0.02 nats of the model actually
trained there -- at or inside the noise floor, and negative (retrieval BEATS the
ceiling) in three of four cells at beta=0.15. Only the anchor_math vertex has real
headroom, at +0.20 nats.

So this script refuses to compute a recovery fraction where there is nothing to
recover. The rule is pre-registered here rather than applied afterwards:

    headroom  H = L(retrieval, pi) - L(ceiling, pi)
    resolvable iff  H > 3 * sigma_pooled

Below that it prints NOT RESOLVABLE and the raw difference with its error bar. Left
unguarded, rho = (L_retr - L_gen) / H explodes with an arbitrary sign on exactly the
cells where nothing can be concluded -- interior member 63 has H = -0.002 nats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Must match llmzoo.data.mixtures.DOMAINS. Hand-copied for the same reason
# report_stack.py hand-copies ARCH_DISPLAY_NAMES: importing llmzoo would pull numpy
# into a script whose whole point is that it does not need it.
DOMAINS = ("web", "code", "math", "books", "multilingual")

# Pre-registered: a cell is only interpretable when retrieval leaves at least this
# many noise-floor sigmas of room between itself and the ceiling.
RESOLVE_SIGMAS = 3.0


# ---------------------------------------------------------------------------
# The scalar and the geometry
# ---------------------------------------------------------------------------

def l1(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def mix_logppl(ppl: Dict[str, float], pi: Sequence[float],
               domains: Sequence[str] = DOMAINS) -> Optional[float]:
    """L(model, pi). None if any needed domain is missing, rather than a wrong sum."""
    tot = 0.0
    for w, d in zip(pi, domains):
        v = ppl.get(d)
        if v is None or v <= 0:
            return None
        tot += w * math.log(v)
    return tot


def nearest(pi: Sequence[float], pool: Sequence[dict]) -> Optional[dict]:
    """The pool member whose mixture is nearest to pi in L1 on the simplex."""
    if not pool:
        return None
    return min(pool, key=lambda m: l1(pi, m["pi"]))


def noise_floor(rows: List[dict],
                domains: Sequence[str] = DOMAINS) -> Tuple[Optional[float], dict]:
    """
    The within-anchor spread: the resolution of every difference in this report.

    An anchor group is the >=2 members that share one mixture and differ only in data
    order, which is exactly what RESEARCH_PLAN §6.3 provisioned the 20 anchors for.
    Returns (pooled sigma on L, per-domain sigma on ln ppl).
    """
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("kind") == "anchor":
            groups.setdefault(r["mixture_id"], []).append(r)

    pooled, per_dom = [], {d: [] for d in domains}
    for members in groups.values():
        if len(members) < 2:
            continue
        pi = members[0]["pi"]
        vals = [mix_logppl(m["ppl"], pi, domains) for m in members]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            pooled.append(statistics.stdev(vals))
        for d in domains:
            lp = [math.log(m["ppl"][d]) for m in members
                  if m["ppl"].get(d, 0) > 0]
            if len(lp) >= 2:
                per_dom[d].append(statistics.stdev(lp))
    sigma = statistics.fmean(pooled) if pooled else None
    per = {d: (statistics.fmean(v) if v else None) for d, v in per_dom.items()}
    return sigma, per


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_arm(ds_path: str, meta_path: str, label: str) -> Optional[dict]:
    ds, meta = _read(ds_path), _read(meta_path)
    if ds is None or meta is None:
        print(f"  [skip] {label}: need both {ds_path} and {meta_path}")
        return None
    domains = tuple(meta.get("domains") or DOMAINS)
    rows = {int(r["idx"]): r for r in ds["rows"]}
    members = {int(m["idx"]): m for m in meta["members"]}
    holdout = meta.get("holdout") or {}
    return {"label": label, "domains": domains, "rows": rows, "members": members,
            "holdout": holdout, "beta": meta.get("beta"),
            "n_members": meta.get("n_members"),
            "ds_rows": list(ds["rows"])}


def load_generated(spec: str) -> Optional[dict]:
    """--gen LABEL=path/to/domain_separation.json, as eval_generated.py writes it."""
    if "=" not in spec:
        print(f"  [skip] --gen {spec!r}: expected LABEL=PATH")
        return None
    label, path = spec.split("=", 1)
    ds = _read(path)
    if ds is None:
        print(f"  [skip] --gen {label}: no file at {path}")
        return None
    out: Dict[str, List[dict]] = {}
    for r in ds["rows"]:
        out.setdefault(str(r.get("arm", label)), []).append(r)
    return {"label": label, "by_arm": out}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def cells(arm: dict, gen: Optional[dict] = None) -> List[dict]:
    """One row per held-out mixture, with every baseline that needs no GPU."""
    domains = arm["domains"]
    rows, members, ho = arm["rows"], arm["members"], arm["holdout"]
    train_idx = [i for i in ho.get("train", []) if i in members and i in rows]
    pool = [{"idx": i, "pi": members[i]["pi"], "ppl": rows[i]["ppl"]}
            for i in train_idx]

    out = []
    for kind in ("interior", "vertex"):
        # The four vertex branches share ONE mixture, so they are one cell with four
        # ceiling replicates, not four cells. Treating them as four would quadruple
        # an n of 1.
        seen: Dict[Tuple[float, ...], List[int]] = {}
        for i in ho.get(kind, []):
            if i in members:
                seen.setdefault(tuple(members[i]["pi"]), []).append(i)
        for pi_t, idxs in seen.items():
            pi = list(pi_t)
            ceil_vals = [mix_logppl(rows[i]["ppl"], pi, domains)
                         for i in idxs if i in rows]
            ceil_vals = [v for v in ceil_vals if v is not None]
            if not ceil_vals:
                continue
            nn_ = nearest(pi, pool)
            l_retr = mix_logppl(nn_["ppl"], pi, domains) if nn_ else None
            l_ceil = statistics.fmean(ceil_vals)
            c = {
                "kind": kind,
                "mixture_id": members[idxs[0]]["mixture_id"],
                "idxs": sorted(idxs),
                "pi": pi,
                "L_ceiling": l_ceil,
                "ceiling_replicates": len(ceil_vals),
                "ceiling_sd": (statistics.stdev(ceil_vals)
                               if len(ceil_vals) > 1 else None),
                "nn_idx": nn_["idx"] if nn_ else None,
                "nn_l1": l1(pi, nn_["pi"]) if nn_ else None,
                "L_retrieval": l_retr,
                "headroom": (l_retr - l_ceil) if l_retr is not None else None,
                "n_pool": len(pool),
            }
            if gen:
                for aname, grows in gen["by_arm"].items():
                    match = [r for r in grows
                             if l1(r["pi"], pi) < 1e-6]
                    vals = [mix_logppl(r["ppl"], pi, domains) for r in match]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        c[f"L_{aname}"] = statistics.median(vals)
                        c[f"n_{aname}"] = len(vals)
            out.append(c)
    return out


def print_arm(arm: dict, gen: Optional[dict] = None) -> dict:
    print("\n" + "=" * 78)
    print(f"RETRIEVAL BASELINE — {arm['label']}  (beta={arm['beta']}, "
          f"N={arm['n_members']})")
    print("=" * 78)

    sigma, per_dom = noise_floor(arm["ds_rows"], arm["domains"])
    cs = cells(arm, gen)
    gen_arms = sorted({k[2:] for c in cs for k in c if k.startswith("L_")}
                      - {"ceiling", "retrieval"})

    print(f"\n  L(model, pi) = sum_d pi_d ln ppl_d   (nats, LOWER is better)")
    print(f"  noise floor  : sigma = "
          f"{'n/a' if sigma is None else f'{sigma:.4f}'} nats, pooled over "
          f"within-anchor spread")
    print(f"  resolvable   : headroom H > {RESOLVE_SIGMAS:.0f} sigma = "
          f"{'n/a' if sigma is None else f'{RESOLVE_SIGMAS * sigma:.4f}'} nats "
          f"(pre-registered)")

    hdr = (f"\n  {'cell':<22} {'nnL1':>6} {'nn':>4} {'ceiling':>9} "
           f"{'retrieval':>10} {'H':>8} {'H/sig':>7}")
    for a in gen_arms:
        hdr += f" {a[:9]:>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    for c in cs:
        H = c["headroom"]
        hs = (H / sigma) if (H is not None and sigma) else None
        line = (f"  {c['mixture_id'][:22]:<22} {c['nn_l1']:>6.3f} "
                f"{c['nn_idx']:>4} {c['L_ceiling']:>9.4f} "
                f"{c['L_retrieval']:>10.4f} {H:>+8.4f} "
                f"{'' if hs is None else f'{hs:>+7.1f}'}")
        for a in gen_arms:
            v = c.get(f"L_{a}")
            line += f" {v:>10.4f}" if v is not None else f" {'-':>10}"
        print(line)

    # The verdict, per cell, with the guard that keeps rho honest.
    print("\n  VERDICT")
    for c in cs:
        H, tag = c["headroom"], f"  {c['mixture_id'][:22]:<22}"
        if sigma is None or H is None:
            print(f"{tag} no noise floor — cannot judge"); continue
        if H <= RESOLVE_SIGMAS * sigma:
            print(f"{tag} NOT RESOLVABLE  (H = {H:+.4f} nats, "
                  f"{H / sigma:+.1f} sigma). Retrieval is already at the ceiling "
                  f"here, so no generative claim is available in this cell.")
            c["resolvable"] = False
            continue
        c["resolvable"] = True
        if not gen_arms:
            print(f"{tag} RESOLVABLE      (H = {H:+.4f} nats, "
                  f"{H / sigma:+.1f} sigma) — awaiting a generated arm.")
            continue
        for a in gen_arms:
            v = c.get(f"L_{a}")
            if v is None:
                continue
            adv = c["L_retrieval"] - v
            rho = adv / H
            c[f"rho_{a}"], c[f"adv_{a}"] = rho, adv
            beat = ("BEAT retrieval" if adv > sigma else
                    "TIED retrieval" if adv > -sigma else "LOST to retrieval")
            print(f"{tag} {a:<12} {beat}: {adv:+.4f} nats "
                  f"({adv / sigma:+.1f} sigma), recovery rho = {rho:+.2f}")

    print("\n  per-domain noise floor (sd of ln ppl within an anchor group):")
    print("    " + "  ".join(
        f"{d}={'n/a' if per_dom[d] is None else f'{per_dom[d]:.4f}'}"
        for d in arm["domains"]))
    print("    Quote the WORST cell, not the pooled figure, beside any per-domain "
          "table.")
    return {"label": arm["label"], "beta": arm["beta"], "sigma_pooled": sigma,
            "sigma_per_domain": per_dom, "resolve_sigmas": RESOLVE_SIGMAS,
            "cells": cs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir",
                   default=os.environ.get("ARTIFACT_DIR", "./artifacts"))
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--arms", type=float, nargs="*", default=[0.15, 0.30])
    p.add_argument("--data_dir", default=None,
                   help="Read committed reports/data/n100-style JSON instead of "
                        "the cluster: {ds,meta}_b<tag>.json. Laptop mode.")
    p.add_argument("--zoo_dir", default=None,
                   help="An explicit arch-level zoo dir, overriding --arms.")
    p.add_argument("--gen", action="append", default=[],
                   help="LABEL=path/to/domain_separation.json for a generated "
                        "cohort (eval_generated.py's output). Repeatable.")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    gen = None
    for spec in args.gen:
        g = load_generated(spec)
        if g:
            gen = g if gen is None else {
                "label": f"{gen['label']}+{g['label']}",
                "by_arm": {**gen["by_arm"], **g["by_arm"]}}

    loaded = []
    if args.zoo_dir:
        a = load_arm(os.path.join(args.zoo_dir, "domain_separation.json"),
                     os.path.join(args.zoo_dir, "zoo_meta.json"), "custom")
        if a:
            loaded.append(a)
    else:
        for beta in args.arms:
            tag = f"b{round(beta * 100):03d}"
            if args.data_dir:
                ds = os.path.join(args.data_dir, f"ds_{tag}.json")
                mt = os.path.join(args.data_dir, f"meta_{tag}.json")
            else:
                d = os.path.join(args.artifact_dir,
                                 f"zoo_{tag}_mini_n100", args.arch)
                ds = os.path.join(d, "domain_separation.json")
                mt = os.path.join(d, "zoo_meta.json")
            a = load_arm(ds, mt, f"beta={beta}")
            if a:
                loaded.append(a)

    if not loaded:
        print("No scorable arm found.")
        return 2

    out = [print_arm(a, gen) for a in loaded]
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
