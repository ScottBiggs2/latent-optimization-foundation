#!/usr/bin/env python3
"""
Build reports/beta_calibration.html from the sealed calibration artifacts.

    python reports/build_calibration_figures.py

Reads only JSON + the Gram eigenvalues, writes one self-contained HTML file with
no external assets, so it opens from disk and survives the /scratch purge.

Why a file rather than a W&B view: the calibration is 45 runs across 3 groups and
the numbers that matter are CROSS-run (per-domain PPL as a 12x5 block structure,
one spectrum per arm against a flat reference). W&B's run table shows one row per
run and its charts are per-run histories, so the comparisons that decide the
result cannot be seen at once there.

numpy is used only to load gram_evals.npy. Everything else is stdlib.
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOMAINS = ("web", "code", "math", "books", "multilingual")
ARMS = [("b015", 0.15), ("b030", 0.30), ("b060", 0.60)]

# Measured on a B200, 2026-09-08. Peak is the ACHIEVABLE bf16 dense GEMM peak
# (1662.4 TFLOPS at 8192^3), not the 2250 datasheet figure.
PEAK_TFLOPS = 1662.4
GEMM = [
    ("mlp down  (T,4d)x(4d,d)", 900.3),
    ("qkv proj  (T,d)x(d,3d)", 704.3),
    ("mlp up    (T,d)x(d,4d)", 697.3),
    ("attn out  (T,d)x(d,d)", 554.0),
    ("lm head   (T,d)x(d,V)", 83.8),
]
# Per-token fwd FLOPs, Mini: 8 blocks vs the LM head.
FLOP_SPLIT = {"blocks": 50.3, "lm_head": 51.5}
SCALES = [
    ("Mini", 51_475_968, 51.0, 325.6, 6.05),
    ("Small", 124_439_808, 31.6, 191.4, 8.60),
    ("Medium", 354_823_168, 14.8, 92.9, 11.90),
]


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def build():
    out = {"peak_tflops": PEAK_TFLOPS, "gemm": GEMM, "flop_split": FLOP_SPLIT,
           "scales": SCALES, "arms": []}

    for tag, beta in ARMS:
        ds = load(f"ds_{tag}.json")
        summ = load(f"sum_{tag}.json")
        spec = summ["per_arch"]["gpt2_zoo_mini"]

        ev = np.load(os.path.join(DATA, f"evals_{tag}.npy")).astype(np.float64)
        ev = np.sort(ev)[::-1]
        ev = ev[ev > 0]
        cum = list(np.cumsum(ev) / ev.sum())

        rows = sorted(ds["rows"], key=lambda r: r["idx"])
        verdict = ds["verdict"]

        # Per-domain: anchor-group means, and the within-group pooled std, so the
        # heatmap can be normalised per column (PPL scale differs ~90x across
        # domains: code sits near 6, web near 500).
        per_dom = {}
        for d in DOMAINS:
            vals = [r["ppl"][d] for r in rows]
            per_dom[d] = {"min": min(vals), "max": max(vals)}

        with open(os.path.join(DATA, f"zoo_{tag}", "gpt2_zoo_mini",
                               "trunk_throughput.json")) as f:
            thr = json.load(f)

        members = []
        mdir = os.path.join(DATA, f"zoo_{tag}", "gpt2_zoo_mini")
        for fn in sorted(os.listdir(mdir)):
            if fn.startswith("member_") and fn.endswith(".json"):
                members.append(json.load(open(os.path.join(mdir, fn))))
        stds = [m["weight_std"] for m in members]
        btps = [m["throughput"]["tokens_per_sec"] for m in members]
        bmfu = [m["throughput"]["mfu"] for m in members]

        out["arms"].append({
            "tag": tag, "beta": beta,
            "rows": [{"idx": r["idx"], "mixture_id": r["mixture_id"],
                      "kind": r["kind"],
                      "ppl": {d: r["ppl"][d] for d in DOMAINS}} for r in rows],
            "per_domain": per_dom,
            "snr": verdict["signal_over_noise"]["per_domain"],
            "snr_min": verdict["signal_over_noise"]["min_ratio"],
            "sep_pass": verdict["separation"]["pass"],
            "sep_wins": verdict["separation"]["wins"],
            "sep_checks": verdict["separation"]["checks"],
            "snr_pass": verdict["signal_over_noise"]["pass"],
            "ev": list(ev), "cum": cum,
            "ev0_over_median": spec["spectrum_ev0_over_median"],
            "effrank_ratio": spec["spectrum_effective_rank_ratio"],
            "effrank": spec["spectrum_effective_rank"],
            "trunk_min": thr["seconds"] / 60.0,
            "trunk_ktoks": thr["tokens_per_sec"] / 1e3,
            "trunk_mfu": 100 * thr["mfu"],
            "branch_min": sum(m["throughput"]["seconds"] for m in members)
                          / max(len(members), 1) / 60.0,
            "branch_ktoks": (sum(btps) / len(btps)) / 1e3,
            "branch_mfu": 100 * sum(bmfu) / len(bmfu),
            "wstd_spread": max(stds) - min(stds),
            "geom": json.load(open(os.path.join(DATA, f"geom_{tag}.json")))
                    if os.path.exists(os.path.join(DATA, f"geom_{tag}.json")) else None,
            "fingerprint": summ.get("ensemble_fingerprint"),
        })
    return out


if __name__ == "__main__":
    payload = build()
    with open(os.path.join(HERE, "calibration_payload.json"), "w") as f:
        json.dump(payload, f, indent=1)
    print("wrote reports/calibration_payload.json")
    for a in payload["arms"]:
        print(f"  beta={a['beta']:.2f}  ev0/med={a['ev0_over_median']:.1f} "
              f"effrank={a['effrank']:.2f}  snr_min={a['snr_min']:.2f} "
              f"branch={a['branch_min']:.1f}min  mfu={a['branch_mfu']:.2f}%")
