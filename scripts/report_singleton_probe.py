#!/usr/bin/env python3
"""
Score the diversity / singleton probe: does the REQUESTED mixture control the model?

    python scripts/report_singleton_probe.py                       # both beta arms
    python scripts/report_singleton_probe.py --arms 0.15 0.30
    python scripts/report_singleton_probe.py --zoo_dir DIR --label mine

PURE STDLIB, exactly like scripts/report_stack.py and
scripts/report_beta_calibration.py: no numpy, no torch, no llmzoo import. So it
runs on a login node, on the `cpu` partition, or on a laptop against
copied-down JSON, and it can never be the reason a reporting step needs a GPU.

WHY NOT THE GATE
----------------
`scripts/eval_domains.py` groups only `kind == "anchor"` (:234) and its whole
verdict block sits behind `if by_anchor:` (:240), so on an ALL-SINGLETON zoo it
exits 1 with `separation: None`. That is a MISSING VERDICT, not a gate failure.
It still writes every row's `pi` and per-domain `ppl`, which is all this needs.

THE STATISTIC (RESEARCH_PLAN §6.5, in miniature)
------------------------------------------------
Per domain d, regress *measured per-domain PPL advantage* on *requested weight*
pi_d across the members:

    adv[i][d] = mean_j( ln ppl[j][d] )  -  ln ppl[i][d]

Cohort-relative, and in LOG space because perplexity composes multiplicatively:
that makes the advantage additive and comparable across domains whose absolute
PPL differs by an order of magnitude. Positive means member i beats the cohort
on domain d.

If the requested mixture controls the output, the slope is positive. This is a
CONTROL test, not a ranking test, which is the whole point -- §6.5 notes it is
immune to misstep 19's collapse trap, where the most collapsed arm posted the
best dPPL.

THE NULL
--------
H0: the pairing between a member's requested pi and its measured perplexities
carries no information. The exchangeable unit is the MEMBER -- it has one pi
vector and one ppl vector -- so the null permutes that pairing jointly across
all five domains. With 6 members there are 6! = 720 permutations, so the null is
enumerated EXACTLY rather than sampled. No asymptotics, no t-distribution, and
no pretending 30 points support a parametric p-value.

Reported alongside the slope, because a slope without a scale is unreadable:
`effect` is slope x (observed pi range), i.e. the log-PPL advantage bought by
moving pi_d across the span the probe actually covers, and its %PPL equivalent.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import statistics
from typing import Dict, List, Optional, Sequence

DOMAINS = ("web", "code", "math", "books", "multilingual")


# ---------------------------------------------------------------------------
# small stats, written out rather than imported
# ---------------------------------------------------------------------------

def _ols_slope(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Least-squares slope of y on x. None when x has no spread."""
    n = len(x)
    if n < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0.0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / sxx


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def _centre(v: Sequence[float]) -> List[float]:
    m = statistics.fmean(v)
    return [x - m for x in v]


# ---------------------------------------------------------------------------

def load_arm(zoo_dir: str, label: str) -> Optional[dict]:
    path = os.path.join(zoo_dir, "domain_separation.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rep = json.load(f)
    rows = [r for r in rep.get("rows", []) if r.get("kind") == "singleton"]
    rows.sort(key=lambda r: r["idx"])
    if len(rows) < 3:
        return {"label": label, "zoo_dir": zoo_dir, "rows": rows,
                "too_few": True}

    meta = {}
    mp = os.path.join(zoo_dir, "zoo_meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            z = json.load(f)
        meta = {k: z.get(k) for k in
                ("beta", "alpha", "min_l1_gap", "n_members", "trunk_steps",
                 "total_steps")}
    geom = None
    gp = os.path.join(zoo_dir, "zoo_geometry.json")
    if os.path.exists(gp):
        with open(gp) as f:
            geom = json.load(f)

    doms = [d for d in DOMAINS if all(d in r.get("ppl", {}) for r in rows)]
    # x = requested weight, a = measured advantage, per domain
    X: Dict[str, List[float]] = {}
    A: Dict[str, List[float]] = {}
    for k, d in enumerate(doms):
        lp = [math.log(r["ppl"][d]) for r in rows]
        mlp = statistics.fmean(lp)
        X[d] = [float(r["pi"][DOMAINS.index(d)]) for r in rows]
        A[d] = [mlp - v for v in lp]

    per = {}
    for d in doms:
        per[d] = {
            "slope": _ols_slope(X[d], A[d]),
            "r": _pearson(X[d], A[d]),
            "pi_min": min(X[d]), "pi_max": max(X[d]),
        }

    # Pooled: centre BOTH variables per domain, so each domain contributes its
    # covariance and no domain-level intercept (a domain with systematically
    # higher PPL) can manufacture a slope.
    xc = {d: _centre(X[d]) for d in doms}
    ac = {d: _centre(A[d]) for d in doms}
    num = sum(sum(a * b for a, b in zip(xc[d], ac[d])) for d in doms)
    den = sum(sum(v * v for v in xc[d]) for d in doms)
    pooled = (num / den) if den > 0 else None

    # EXACT null: enumerate all n! permutations of the member <-> pi pairing,
    # applied jointly across domains (a member is one unit).
    n = len(rows)
    perms = list(itertools.permutations(range(n)))
    pooled_null, per_null = [], {d: [] for d in doms}
    for sg in perms:
        pnum = 0.0
        for d in doms:
            xs = [xc[d][i] for i in sg]
            pnum += sum(a * b for a, b in zip(xs, ac[d]))
            per_null[d].append(_ols_slope([X[d][i] for i in sg], A[d]))
        pooled_null.append(pnum / den if den > 0 else 0.0)

    def _p_ge(obs, null):
        if obs is None:
            return None
        vals = [v for v in null if v is not None]
        return sum(1 for v in vals if v >= obs - 1e-12) / len(vals)

    for d in doms:
        per[d]["p_one_sided"] = _p_ge(per[d]["slope"], per_null[d])

    return {
        "label": label, "zoo_dir": zoo_dir, "meta": meta, "geom": geom,
        "rows": rows, "domains": doms, "per_domain": per,
        "pooled_slope": pooled,
        "pooled_p_one_sided": _p_ge(pooled, pooled_null),
        "n_perms": len(perms), "n_members": n,
        "n_positive": sum(1 for d in doms
                          if (per[d]["slope"] or 0.0) > 0.0),
    }


# ---------------------------------------------------------------------------

def print_arm(a: dict) -> None:
    print("\n" + "=" * 78)
    print(f"ARM {a['label']}   {a['zoo_dir']}")
    m = a.get("meta") or {}
    if m:
        print(f"  beta={m.get('beta')}  alpha={m.get('alpha')}  "
              f"min_l1_gap={m.get('min_l1_gap')}  "
              f"branch_steps={(m.get('total_steps') or 0) - (m.get('trunk_steps') or 0)}")
    print("=" * 78)
    if a.get("too_few"):
        print(f"  only {len(a['rows'])} singleton rows -- not scorable yet")
        return

    print("\n  requested pi (rows) vs measured PPL")
    print(f"    {'idx':>4} " + " ".join(f"{d[:4]:>7}" for d in a["domains"])
          + "  |  " + " ".join(f"{d[:4]:>8}" for d in a["domains"]))
    for r in a["rows"]:
        pis = " ".join(f"{r['pi'][DOMAINS.index(d)]:7.3f}" for d in a["domains"])
        ppl = " ".join(f"{r['ppl'][d]:8.2f}" for d in a["domains"])
        print(f"    {r['idx']:>4} {pis}  |  {ppl}")

    print("\n  §6.5 slope: measured log-PPL advantage regressed on requested pi_d")
    print(f"    {'domain':<14} {'slope':>9} {'r':>7} {'p(exact)':>9} "
          f"{'pi range':>12} {'effect':>9} {'= %PPL':>8}")
    for d in a["domains"]:
        p = a["per_domain"][d]
        s, r_, pv = p["slope"], p["r"], p["p_one_sided"]
        span = p["pi_max"] - p["pi_min"]
        eff = None if s is None else s * span
        pct = None if eff is None else 100.0 * (1.0 - math.exp(-eff))
        f = lambda v, w, prec=3: ("%*.*f" % (w, prec, v)) if v is not None else " " * (w - 2) + "--"
        print(f"    {d:<14} {f(s,9)} {f(r_,7)} {f(pv,9)} "
              f"{p['pi_min']:5.3f}-{p['pi_max']:5.3f} {f(eff,9)} {f(pct,8,1)}%")
    print(f"    {'-'*72}")
    f = lambda v, w, prec=3: ("%*.*f" % (w, prec, v)) if v is not None else " " * (w - 2) + "--"
    print(f"    {'POOLED':<14} {f(a['pooled_slope'],9)} {'':>7} "
          f"{f(a['pooled_p_one_sided'],9)}   "
          f"(exact null over {a['n_perms']} permutations of "
          f"{a['n_members']} members)")
    print(f"    {'sign count':<14} {a['n_positive']}/{len(a['domains'])} "
          f"domains have a positive slope")
    print("\n  'effect' is slope x observed pi range: the log-PPL advantage bought by")
    print("  moving pi_d across the span this probe actually covers. '%PPL' is the")
    print("  same number as a perplexity reduction.")

    g = a.get("geom")
    if g:
        print(f"\n  weight-space geometry (misstep 19 read; no anchor groups here, so")
        print(f"  between/within is undefined by construction)")
        print(f"    displacement from trunk   {g.get('disp_rel')}")
        print(f"    spread / displacement     {g.get('spread_over_disp')}   "
              f"(sqrt(2)=1.414 = moved independently)")
        n = g.get("n_members") or 1
        iid = ((n - 1) / n / 2) ** 0.5 if n > 1 else float("nan")
        print(f"    centroid fraction         {g.get('mean_frac')}   "
              f"(i.i.d. at N={n} would be {iid:.3f})")


def verdict(arms: List[dict]) -> int:
    print("\n" + "=" * 78)
    print("VERDICT -- against the rule fixed in advance (docs/PHASE1_HANDOFF.md §5)")
    print("=" * 78)
    scorable = [a for a in arms if a and not a.get("too_few")]
    if not scorable:
        print("  No arm is scorable yet. That is INCOMPLETE, not a failure:")
        print("  eval_domains.py must have run and written domain_separation.json.")
        return 2

    any_clear = False
    for a in scorable:
        s, p = a["pooled_slope"], a["pooled_p_one_sided"]
        pos = a["n_positive"]
        nd = len(a["domains"])
        clear = (s is not None and s > 0 and p is not None and p <= 0.05
                 and pos >= nd - 1)
        any_clear |= clear
        tag = "CLEARLY POSITIVE" if clear else "NOT clearly positive"
        print(f"  beta={a['meta'].get('beta')}  pooled slope="
              f"{s:+.3f}  p={p:.4f}  {pos}/{nd} domains positive  ->  {tag}")

    print()
    if any_clear:
        print("  -> The requested mixture CONTROLS the model at Phase 2's hardest")
        print("     separation (min pair L1 0.164 vs Phase 2's 0.173). beta is")
        print("     confirmed; proceed to the N=100 zoo.")
        print("     If the SMALLER beta is also clearly positive, prefer it: §4.3's")
        print("     rule is the smallest beta at which mixture identity is")
        print("     measurable, and it halves the cost of Small and Medium.")
    else:
        print("  -> Slopes are flat or mixed. Per the pre-registered rule this")
        print("     implicates the 80-SINGLETON DESIGN, not beta. Do NOT simply")
        print("     raise beta: 0.60 was worst on the gate (4.54), worst on")
        print("     weight-space between/within (2.88 vs 4.08), and its 1.7x")
        print("     saving guts §4.1 motivation 1. Reconsider §6.3's composition")
        print("     -- fewer, better-separated pi, or more branches per mixture --")
        print("     before spending anything further.")
    print("=" * 78)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir",
                   default=os.environ.get("ARTIFACT_DIR", "./artifacts"))
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--arms", type=float, nargs="*", default=[0.15, 0.30],
                   help="betas whose <artifact_dir>/zoo_b<NNN>_singleton/<arch> "
                        "to read")
    p.add_argument("--zoo_dir", default=None,
                   help="score one explicit arch-level dir instead")
    p.add_argument("--label", default="custom")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    arms: List[dict] = []
    if args.zoo_dir:
        arms.append(load_arm(args.zoo_dir, args.label))
    else:
        for b in args.arms:
            tag = f"b{round(b * 100):03d}"
            d = os.path.join(args.artifact_dir, f"zoo_{tag}_singleton", args.arch)
            a = load_arm(d, f"beta={b:.2f}")
            if a is None:
                print(f"[probe] no domain_separation.json under {d} -- skipping")
                continue
            arms.append(a)

    arms = [a for a in arms if a]
    print("=" * 78)
    print("singleton / diversity probe -- RESEARCH_PLAN §6.5 slope")
    print(f"artifact_dir = {args.artifact_dir}   arch = {args.arch}")
    print("=" * 78)
    for a in arms:
        print_arm(a)
    rc = verdict(arms)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(arms, f, indent=2, default=str)
        print(f"\nwrote {args.json_out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
