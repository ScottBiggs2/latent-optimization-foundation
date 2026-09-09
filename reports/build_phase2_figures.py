#!/usr/bin/env python3
"""
Assemble reports/phase2_payload.json from the sealed Phase 2 artifacts.

PURE STDLIB, like scripts/report_stack.py and report_beta_calibration.py: it runs
on a laptop against the copied-down JSON under reports/data/, so a figure never
needs a GPU or a cluster round-trip to regenerate.

Inputs (all committed):
    reports/data/n100/{ds,geom,meta,sum,blockspec}_b0*.json   Phase 2, N=100
    reports/data/{ds,geom,sum}_b0*.json                       Phase 1, N=12
    reports/data/probe/{ds,geom}_b0*_singleton.json           the singleton probe
    reports/data/probe/probe_slopes.json                      the §6.5 slopes
"""
from __future__ import annotations

import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
DOMAINS = ("web", "code", "math", "books", "multilingual")
ARMS = (("b015", 0.15), ("b030", 0.30))
NTOP = 24          # components to plot; the elbow is well inside this


def rd(*parts):
    p = os.path.join(D, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def snr_of(ds):
    v = ((ds or {}).get("verdict") or {}).get("signal_over_noise") or {}
    return v.get("per_domain") or {}


def sep_of(ds):
    return ((ds or {}).get("verdict") or {}).get("separation") or {}


def cumvar(evals):
    tot = sum(evals) or 1.0
    out, acc = [], 0.0
    for e in evals[:NTOP]:
        acc += e
        out.append(acc / tot)
    return out


def build():
    p = {"domains": list(DOMAINS), "n_top": NTOP, "arms": []}

    for tag, beta in ARMS:
        n100_ds = rd("n100", f"ds_{tag}.json")
        n100_sum = rd("n100", f"sum_{tag}.json")
        n100_bs = rd("n100", f"blockspec_{tag}.json")
        n100_gm = rd("n100", f"geom_{tag}.json")
        n12_ds = rd(f"ds_{tag}.json")
        n12_sum = rd(f"sum_{tag}.json")
        n12_gm = rd(f"geom_{tag}.json")
        n12_bs = rd("probe", f"blockspec_{tag}.json")
        pr_gm = rd("probe", f"geom_{tag}_singleton.json")

        pa100 = ((n100_sum or {}).get("per_arch") or {}).get("gpt2_zoo_mini") or {}
        pa12 = ((n12_sum or {}).get("per_arch") or {}).get("gpt2_zoo_mini") or {}

        regions = {}
        for name, src in (("whole_stack", n100_bs), ("block_only", n100_bs),
                          ("embeddings_only", n100_bs)):
            r = ((src or {}).get("regions") or {}).get(name) or {}
            if not r:
                continue
            regions[name] = {
                "ev0_over_median": r.get("spectrum_ev0_over_median"),
                "effective_rank": r.get("spectrum_effective_rank"),
                "effective_rank_ratio": r.get("spectrum_effective_rank_ratio"),
                "variance_share": r.get("variance_share_of_whole"),
                "cumvar": cumvar(r.get("evals") or []),
            }
        # the same three regions at N=12, for the "design ceiling lifted" contrast
        regions12 = {}
        for name in ("whole_stack", "block_only", "embeddings_only"):
            r = ((n12_bs or {}).get("regions") or {}).get(name) or {}
            if r:
                regions12[name] = {
                    "effective_rank": r.get("spectrum_effective_rank"),
                    "effective_rank_ratio": r.get("spectrum_effective_rank_ratio"),
                    "variance_share": r.get("variance_share_of_whole"),
                }

        # per-domain PPL of the 20 anchors at N=100, normalised per domain to its
        # own best member -- the block-diagonal pattern IS the result (chart 1 of
        # the Phase 1 report, now with all five anchors present)
        rows = [r for r in (n100_ds or {}).get("rows", [])
                if r.get("kind") == "anchor"]
        rows.sort(key=lambda r: (r["mixture_id"], r["branch"]))
        best = {d: min(r["ppl"][d] for r in rows) for d in DOMAINS} if rows else {}
        heat = [{"idx": r["idx"], "mixture_id": r["mixture_id"],
                 "branch": r["branch"],
                 "rel": [r["ppl"][d] / best[d] for d in DOMAINS],
                 "ppl": [r["ppl"][d] for d in DOMAINS]} for r in rows]

        # anchor-group means per domain, for the separation panel
        groups = {}
        for r in rows:
            groups.setdefault(r["mixture_id"], []).append(r)
        gmeans = {g: [statistics.fmean([r["ppl"][d] for r in v]) for d in DOMAINS]
                  for g, v in sorted(groups.items())}

        p["arms"].append({
            "tag": tag, "beta": beta,
            "n100": {
                "n_members": (n100_ds or {}).get("n_evaluated"),
                "separation": sep_of(n100_ds),
                "snr": snr_of(n100_ds),
                "ev0_over_median": pa100.get("spectrum_ev0_over_median"),
                "effective_rank": pa100.get("spectrum_effective_rank"),
                "effective_rank_ratio": pa100.get("spectrum_effective_rank_ratio"),
                "fingerprint": (n100_sum or {}).get("ensemble_fingerprint"),
                "regions": regions,
                "geom": n100_gm,
                "heat": heat, "group_means": gmeans,
            },
            "n12": {
                "separation": sep_of(n12_ds),
                "snr": snr_of(n12_ds),
                "ev0_over_median": pa12.get("spectrum_ev0_over_median"),
                "effective_rank": pa12.get("spectrum_effective_rank"),
                "effective_rank_ratio": pa12.get("spectrum_effective_rank_ratio"),
                "regions": regions12,
                "geom": n12_gm,
            },
            "probe": {"geom": pr_gm},
        })

    # the §6.5 slopes, as report_singleton_probe.py computed them
    slopes = rd("probe", "probe_slopes.json") or []
    p["probe_slopes"] = [{
        "label": a.get("label"),
        "beta": (a.get("meta") or {}).get("beta"),
        "pooled_slope": a.get("pooled_slope"),
        "pooled_p": a.get("pooled_p_one_sided"),
        "n_positive": a.get("n_positive"),
        "n_perms": a.get("n_perms"),
        "per_domain": a.get("per_domain"),
        "points": [{"idx": r["idx"], "pi": r["pi"], "ppl": r["ppl"]}
                   for r in a.get("rows", [])],
    } for a in slopes if isinstance(a, dict)]

    # simplex dimension: 5 domains -> Delta^4, so between-mixture structure can
    # span at most 4 dimensions no matter how many distinct pi are sampled.
    p["simplex_dim"] = len(DOMAINS) - 1
    p["k"] = 99
    return p


if __name__ == "__main__":
    payload = build()
    dest = os.path.join(HERE, "phase2_payload.json")
    with open(dest, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {dest}  ({os.path.getsize(dest)/1024:.0f} KB)")
    for a in payload["arms"]:
        r = a["n100"]["regions"]
        print(f"  beta={a['beta']}: effrank whole={r['whole_stack']['effective_rank']:.2f} "
              f"block={r['block_only']['effective_rank']:.2f} "
              f"emb={r['embeddings_only']['effective_rank']:.2f}  "
              f"(simplex dim {payload['simplex_dim']})")
