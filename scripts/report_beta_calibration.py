#!/usr/bin/env python3
"""
Render the RESEARCH_PLAN §4.2 beta-calibration table from sealed artifacts.

    python scripts/report_beta_calibration.py                      # $ARTIFACT_DIR
    python scripts/report_beta_calibration.py --artifact_dir DIR
    python scripts/report_beta_calibration.py --betas 0.15 0.30 0.60

PURE STDLIB, on purpose, exactly like scripts/report_stack.py: no numpy, no
torch, no llmzoo import. That means it runs on a login node, on the `cpu`
partition, or on a laptop against copied-down JSON -- and it can never be the
reason a reporting step needs a GPU.

Reads, per beta:
    <zoo_root>/<arch>/domain_separation.json   the §4.3 gate verdict + per-domain PPL
    <zoo_root>/<arch>/trunk_throughput.json    trunk wall clock and MFU
    <zoo_root>/<arch>/member_*.json            per-branch throughput
    runs/zoo_b<NNN>_k<k>/pipeline_summary_k<k>.json   the spectrum

Gate reporting rule (§4.3, and the reason this is not just a mean): at N=12 the
plan is all one-hot anchors and `build_zoo_plan` emits only 3 whole groups, so
condition 1 (separation) is evaluated on 3 domains while condition 2
(signal>noise) is evaluated on all 5 -- including books and multilingual, which
no member specialised in. A failure there is a DIFFERENT diagnosis from a
separation failure, and only the latter is an argument for raising beta. So the
deciding sub-condition and domain are always named, never collapsed to PASS/FAIL.

Spectrum reporting rule (misstep 15b): report `ev0/median` and
`effective_rank_ratio` only. `ev[0]/ev[k-1]` reads 12.09 on data whose true ratio
is 1.01, so it is printed only under --show-tail-ratio and labelled as unusable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import Dict, List, Optional

DOMAINS = ("web", "code", "math", "books", "multilingual")


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def tag_for(beta: float) -> str:
    return f"b{round(beta * 100):03d}"


def fmt(x, spec="8.3f", none="    --"):
    return none if x is None else format(x, spec)


def collect(artifact_dir: str, arch: str, beta: float, k: int,
            zoo_name_tmpl: str = "zoo_{tag}",
            run_name_tmpl: str = "zoo_{tag}_k{k}") -> dict:
    """
    Gather one beta arm's artifacts.

    The two templates exist because Phase 2 does not use Phase 1's directory
    layout. Phase 1 wrote `zoo_b030/<arch>` and `runs/zoo_b030_k11`; Phase 2 keys
    both on (beta, arch, N) because every N=100 scale has k=99 and would
    otherwise collide -- see slurm/launch_beta_arm.sh. Defaults reproduce the
    Phase 1 paths exactly, so the calibration table still renders unchanged.

    Available substitutions: {tag} {arch} {slug} {k} {beta}, where slug is the
    arch with its `gpt2_zoo_` prefix removed.
    """
    tag = tag_for(beta)
    sub = {"tag": tag, "arch": arch, "slug": arch.replace("gpt2_zoo_", ""),
           "k": k, "beta": beta}
    zoo = os.path.join(artifact_dir, zoo_name_tmpl.format(**sub), arch)
    out = {"beta": beta, "tag": tag, "zoo_dir": zoo}

    out["n_members"] = len(glob.glob(os.path.join(zoo, "w_*.npy")))
    out["gate"] = read_json(os.path.join(zoo, "domain_separation.json"))
    out["trunk"] = read_json(os.path.join(zoo, "trunk_throughput.json"))

    members = []
    for p in sorted(glob.glob(os.path.join(zoo, "member_*.json"))):
        m = read_json(p)
        if m and isinstance(m.get("throughput"), dict):
            members.append(m)
    out["members"] = members

    run = run_name_tmpl.format(**sub)
    summary = read_json(os.path.join(
        artifact_dir, "runs", run, f"pipeline_summary_k{k}.json"))
    out["run_name"] = run
    out["spectrum"] = (summary or {}).get("per_arch", {}).get(arch)
    out["ensemble_fingerprint"] = (summary or {}).get("ensemble_fingerprint")
    return out


def ppl_spread(gate: Optional[dict]) -> Dict[str, dict]:
    """Per-domain best/worst ANCHOR-GROUP mean, and the pooled within-group std."""
    if not gate or not gate.get("rows"):
        return {}
    groups: Dict[str, List[dict]] = {}
    for r in gate["rows"]:
        if r.get("kind") == "anchor":
            groups.setdefault(r["mixture_id"], []).append(r)
    spread = {}
    for dom in DOMAINS:
        means, within = [], []
        for rows in groups.values():
            vals = [r["ppl"][dom] for r in rows if dom in r.get("ppl", {})]
            if not vals:
                continue
            means.append(statistics.fmean(vals))
            if len(vals) > 1:
                within.append(statistics.stdev(vals))
        if means:
            spread[dom] = {
                "best": min(means), "worst": max(means),
                "gap": max(means) - min(means),
                "within": statistics.fmean(within) if within else None,
            }
    return spread


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir",
                   default=os.environ.get("ARTIFACT_DIR", "./artifacts"))
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.15, 0.30, 0.60])
    p.add_argument("--k", type=int, default=11)
    p.add_argument("--n_expected", type=int, default=12,
                   help="members a complete arm should have; only affects the "
                        "'members' column's denominator")
    p.add_argument("--zoo_name_tmpl", default="zoo_{tag}",
                   help="Phase 2 uses zoo_{tag}_{slug}_n<N>. Substitutions: "
                        "{tag} {arch} {slug} {k} {beta}")
    p.add_argument("--run_name_tmpl", default="zoo_{tag}_k{k}",
                   help="Phase 2 uses zoo_{tag}_{slug}_k{k}")
    p.add_argument("--show-tail-ratio", action="store_true",
                   help="also print ev[0]/ev[k-1], which is NOT a flatness "
                        "measure at the rank bound (misstep 15b)")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    arms = [collect(args.artifact_dir, args.arch, b, args.k,
                    args.zoo_name_tmpl, args.run_name_tmpl)
            for b in args.betas]

    print("=" * 78)
    print(f"beta calibration -- RESEARCH_PLAN §4.2 / §4.3   arch={args.arch} "
          f"k={args.k}")
    print(f"artifact_dir = {args.artifact_dir}")
    print("=" * 78)

    # ---- completeness -----------------------------------------------------
    print("\n## members and artifacts")
    print(f"  {'beta':>5} {'members':>8} {'gate':>6} {'spectrum':>9} "
          f"{'ens fingerprint':>17}")
    for a in arms:
        print(f"  {a['beta']:5.2f} {a['n_members']:6d}/{args.n_expected} "
              f"{'yes' if a['gate'] else 'MISSING':>6} "
              f"{'yes' if a['spectrum'] else 'MISSING':>9} "
              f"{str(a['ensemble_fingerprint'] or '--'):>17}")

    # ---- throughput -------------------------------------------------------
    print("\n## wall clock and MFU  (steady state = median of recent windows;")
    print("##   the post-startup AVERAGE under-reports by ~10%, and the peak is a")
    print("##   MEASURED 1662.4 TFLOPS bf16 dense on B200, not the 2250 datasheet)")
    print(f"  {'beta':>5} {'trunk min':>10} {'trunk ktok/s':>13} {'trunk MFU':>10} "
          f"{'branch min':>11} {'branch ktok/s':>14} {'branch MFU':>11}")
    for a in arms:
        t = a["trunk"] or {}
        tmin = (t.get("seconds") or 0) / 60 or None
        bt = [m["throughput"] for m in a["members"]]
        bmin = statistics.fmean([x["seconds"] for x in bt]) / 60 if bt else None
        btps = statistics.fmean([x["tokens_per_sec"] for x in bt]) if bt else None
        bmfu = statistics.fmean([x["mfu"] for x in bt if x.get("mfu")]) if bt else None
        print(f"  {a['beta']:5.2f} {fmt(tmin,'10.1f')} "
              f"{fmt((t.get('tokens_per_sec') or 0)/1e3 or None,'13.1f')} "
              f"{fmt(100*t['mfu'] if t.get('mfu') else None,'9.2f')}% "
              f"{fmt(bmin,'11.1f')} {fmt(btps/1e3 if btps else None,'14.1f')} "
              f"{fmt(100*bmfu if bmfu else None,'10.2f')}%")

    # ---- spectrum ---------------------------------------------------------
    print("\n## spectrum at k=%d  (§4.4 predicted effective_rank_ratio 0.2-0.3, "
          "ev0/median >> 1)" % args.k)
    print(f"  {'beta':>5} {'ev0/median':>11} {'eff_rank_ratio':>15} "
          f"{'eff_rank':>9} {'var@k':>8}"
          + ("  ev0/ev[k-1] (UNUSABLE)" if args.show_tail_ratio else ""))
    for a in arms:
        s = a["spectrum"] or {}
        line = (f"  {a['beta']:5.2f} "
                f"{fmt(s.get('spectrum_ev0_over_median'),'11.3f')} "
                f"{fmt(s.get('spectrum_effective_rank_ratio'),'15.3f')} "
                f"{fmt(s.get('spectrum_effective_rank'),'9.2f')} "
                f"{fmt(s.get('variance_captured_at_k'),'8.4f')}")
        if args.show_tail_ratio:
            line += f"  {fmt(s.get('spectrum_ev0_over_evlast'),'10.2f')}"
        print(line)

    # ---- per-domain PPL spread -------------------------------------------
    print("\n## per-domain PPL: between-anchor gap vs within-anchor spread")
    for a in arms:
        sp = ppl_spread(a["gate"])
        if not sp:
            print(f"\n  beta={a['beta']:.2f}: no gate artifact")
            continue
        print(f"\n  beta={a['beta']:.2f}")
        print(f"    {'domain':<14} {'best':>9} {'worst':>9} {'gap':>9} "
              f"{'within':>9} {'gap/within':>11}")
        for dom, d in sp.items():
            ratio = (d["gap"] / d["within"]) if d["within"] else None
            print(f"    {dom:<14} {d['best']:9.2f} {d['worst']:9.2f} "
                  f"{d['gap']:9.2f} {fmt(d['within'],'9.3f')} "
                  f"{fmt(ratio,'11.2f')}")

    # ---- the gate ---------------------------------------------------------
    print("\n## §4.3 GATE")
    for a in arms:
        g = a["gate"]
        if not g:
            print(f"  beta={a['beta']:.2f}  no verdict")
            continue
        v = g.get("verdict", {})
        sep, snr = v.get("separation"), v.get("signal_over_noise")
        sep_ok = bool(sep and sep.get("pass"))
        snr_ok = bool(snr and snr.get("pass"))
        overall = sep_ok and snr_ok
        print(f"\n  beta={a['beta']:.2f}  ->  "
              f"{'PASS' if overall else 'FAIL'}   "
              f"(n_evaluated={g.get('n_evaluated')})")
        if sep:
            print(f"    separation    {'PASS' if sep_ok else 'FAIL'}  "
                  f"{sep.get('wins')}/{sep.get('checks')} anchors best on own domain")
        if snr:
            per = snr.get("per_domain") or {}
            worst = min(per, key=per.get) if per else None
            print(f"    signal>noise  {'PASS' if snr_ok else 'FAIL'}  "
                  f"min_ratio={fmt(snr.get('min_ratio'),'.3f')}"
                  + (f"  (worst domain: {worst})" if worst else ""))
            if per:
                print("      " + "  ".join(f"{d}={per[d]:.2f}" for d in per))
        if not overall:
            which = []
            if not sep_ok:
                which.append("separation")
            if not snr_ok:
                per = (snr or {}).get("per_domain") or {}
                worst = min(per, key=per.get) if per else "?"
                which.append(f"signal>noise (worst: {worst})")
            print(f"    decided by: {', '.join(which)}")
            if sep_ok and not snr_ok:
                per = (snr or {}).get("per_domain") or {}
                worst = min(per, key=per.get) if per else None
                if worst in ("books", "multilingual"):
                    print(f"    NOTE: separation PASSED and the failure is on "
                          f"'{worst}', which no N=12 anchor trained on. That is "
                          f"NOT the same finding as a separation failure and is "
                          f"a weaker argument for raising beta.")

    # ---- recommendation ---------------------------------------------------
    def judged(a) -> bool:
        """Did this arm actually produce a verdict? Missing != failed."""
        return bool(a["gate"] and (a["gate"].get("verdict") or {}).get("separation"))

    def passed(a) -> bool:
        v = (a["gate"] or {}).get("verdict") or {}
        return all((v.get(kk) or {}).get("pass") for kk in
                   ("separation", "signal_over_noise"))

    incomplete = [a for a in arms if not judged(a)]
    passing = [a for a in arms if judged(a) and passed(a)]
    failing = [a for a in arms if judged(a) and not passed(a)]

    print("\n" + "=" * 78)
    # "No artifact" and "gate fired" are DIFFERENT states and must not be
    # collapsed. Reporting an unbuilt zoo as a gate failure would invite exactly
    # the wrong response (raise beta) to a build problem, and RESEARCH_PLAN §4.3
    # is explicit that a measurement on a zoo that failed the gate says nothing.
    if incomplete:
        print("INCOMPLETE -- these arms produced no verdict, which is NOT a gate "
              "failure:")
        for a in incomplete:
            print(f"    beta={a['beta']:.2f}  members={a['n_members']}/"
                  f"{args.n_expected}  "
                  f"gate={'present' if a['gate'] else 'absent'}")
        print("  Finish or re-run these before drawing any conclusion. Resubmitting")
        print("  a branch array is safe: a member whose w_<i>.npy exists exits")
        print("  immediately, so only the missing members are retrained.")
    def betas_str(items) -> str:
        return ", ".join(format(a["beta"], ".2f") for a in items)

    if passing:
        best = min(passing, key=lambda a: a["beta"])
        others = [a for a in passing if a is not best]
        extra = f"   (also passing: {betas_str(others)})" if others else ""
        print(f"RECOMMENDED beta = {best['beta']:.2f}{extra}")
        print("  §4.3's rule is the SMALLEST beta at which mixture identity is")
        print("  measurable in the models themselves -- not the largest gap.")
    elif failing and not incomplete:
        print("NO beta PASSED the gate.")
        print(f"  Tried: {betas_str(failing)}")
        print("  Per §4.3 the response to a failure is a LARGER beta and a re-run,")
        print("  not a reframe. If the largest beta tried still fails, that is a")
        print("  real finding about the trunk-branch design and needs review")
        print("  before more compute.")
    elif failing:
        print("Of the arms that were judged, none passed -- but some are "
              "incomplete, so this is not yet a result.")
    print("=" * 78)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(arms, f, indent=2, default=str)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
