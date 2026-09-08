"""
Render a stack-run results JSON as markdown. Rows are arch x k, columns are arms.

Why this is a separate file from report.py
------------------------------------------
1. `report.py` is a BLOCK-PIPELINE file (README lists it under the legacy section)
   and legacy files are left alone.
2. The schemas differ in SHAPE, not just field names. `report.py`'s formatters all
   assume `{arch: {...}}` -- one flat row per arch. The stack schema is
   `{"<arch>@k<k>": {..., "arms": {arm: {...}}}}`, a three-axis cube that has to be
   pivoted before it can be tabulated. Sharing a formatter would mean branching on
   schema at every level.
3. `report.py` is stdlib-pure so it runs locally against files copied down from
   Explorer. This file keeps that property, which is why it reads the results JSON
   and the manifest directly and NEVER calls run_bundle.load_run -- constructing an
   EnsembleDataset creates directories, so load_run is not a pure read.

The one thing shared is the display-name table, imported read-only.

    python scripts/report_stack.py --run_dir ./runs/perfam
    python scripts/report_stack.py --results ./stack_eval_results_s5.json -o report.md
    python scripts/report_stack.py --run_dir ./runs/perfam --inspect
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Mirrors llmzoo/models/registry.py ARCH_CONFIGS default_model_id -- a plain dict
# here (rather than importing llmzoo.models.registry) so this script keeps ZERO heavy
# imports. That property is enforced by test_stdlib_purity in tests/test_report.py and
# is what lets this render on the `cpu` partition and on a laptop.
ARCH_DISPLAY_NAMES = {
    "gpt2_medium":   "openai-community/gpt2-medium",
    "smollm2_360m":  "HuggingFaceTB/SmolLM2-360M",
    "qwen3_0_6b":    "Qwen/Qwen3-0.6B",
    "gemma3_270m":   "google/gemma-3-270m",
    "opt_350m":      "facebook/opt-350m",
    "smollm2_135m":  "HuggingFaceTB/SmolLM2-135M",
    "pythia_160m":   "EleutherAI/pythia-160m",
    "pythia_410m":   "EleutherAI/pythia-410m",
}

# Display order. Reconstruction arms first, then generative, then diagnostics --
# reading left to right walks from "can the basis represent this" to "can we sample
# something that works".
ARM_ORDER = ["pca_only", "vae", "generate", "gauss_codes",
             "flow_codes", "flow_latent", "flow_rt_codes", "flow_rt_latent"]

# A negative dPPL means two DIFFERENT things depending on the arm, and conflating
# them mis-attributes a real failure. A reconstruction arm targets one specific
# member, so a negative delta is truncation denoising a noisy target (misstep 14).
# A generative arm targets the distribution, so a negative delta usually means it
# collapsed toward the family mean -- which on this ensemble IS w_0 (misstep 19).
RECON_ARMS = {"pca_only", "vae", "flow_rt_codes", "flow_rt_latent"}
GEN_ARMS = {"generate", "gauss_codes", "flow_codes", "flow_latent"}

KEY_RE = re.compile(r"^(?P<arch>.+)@k(?P<k>\d+)$")


def _name(arch: str) -> str:
    return ARCH_DISPLAY_NAMES.get(arch, arch)


def parse_key(key: str) -> Tuple[str, int]:
    """
    'pythia_410m@k99' -> ('pythia_410m', 99).

    Raises on anything else, deliberately. A legacy flat-schema file
    (`lm_eval_results.json`, keyed by bare arch) must fail loudly here rather than
    render as a table of blanks that looks like a real result.
    """
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(
            f"{key!r} is not a stack results key. Expected '<arch>@k<rank>'. If this "
            f"is a legacy block-pipeline file, use report.py instead.")
    return m.group("arch"), int(m.group("k"))


def load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    for key in data:
        parse_key(key)          # validate every key before rendering anything
    return data


def pivot(results: dict) -> Tuple[List[Tuple[str, int]], List[str], Dict]:
    """-> (row_keys sorted by (arch, -k), arm columns, {(arch,k,arm): arm_dict})."""
    rows, arms_seen, cell = set(), set(), {}
    for key, r in results.items():
        arch, k = parse_key(key)
        rows.add((arch, k))
        for arm, a in r.get("arms", {}).items():
            arms_seen.add(arm)
            cell[(arch, k, arm)] = a
    # Descending k, so the rank bound (the self-test) is the first row per arch.
    row_keys = sorted(rows, key=lambda t: (t[0], -t[1]))
    cols = [a for a in ARM_ORDER if a in arms_seen] + \
           sorted(a for a in arms_seen if a not in ARM_ORDER)
    return row_keys, cols, cell


def _table(results: dict, header: str, value, note: str = "") -> str:
    row_keys, cols, cell = pivot(results)
    if not row_keys:
        return ""
    out = [f"### {header}", ""]
    if note:
        out += [note, ""]
    out.append("| Model | k | " + " | ".join(cols) + " |")
    out.append("|" + "---|" * (len(cols) + 2))
    for arch, k in row_keys:
        cells = []
        for arm in cols:
            a = cell.get((arch, k, arm))
            cells.append("—" if a is None else value(a, results[f"{arch}@k{k}"]))
        out.append(f"| {_name(arch)} | {k} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def table_ppl_delta(results: dict) -> str:
    return _table(
        results, "Perplexity delta (%), WikiText-2",
        lambda a, r: f"{a['ppl_delta_pct']:+.3f}",
        note="Signed percentage change against the evaluation target's own "
             "perplexity. This is the gate — not cosine.")


def table_ppl_absolute(results: dict) -> str:
    row_keys, cols, cell = pivot(results)
    if not row_keys:
        return ""
    out = ["### Perplexity, absolute", "",
           "| Model | k | target PPL | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 3)]
    for arch, k in row_keys:
        r = results[f"{arch}@k{k}"]
        cells = [("—" if cell.get((arch, k, arm)) is None
                  else f"{cell[(arch, k, arm)]['ppl']:.3f}") for arm in cols]
        out.append(f"| {_name(arch)} | {k} | {r['original_ppl']:.3f} | "
                   + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def benchmarks_present(results: dict) -> List[str]:
    names = set()
    for r in results.values():
        for a in r.get("arms", {}).values():
            names.update((a.get("bench") or {}).keys())
    return sorted(names)


def _metric_duplicates_acc(results: dict, bench: str) -> bool:
    """True when acc_norm equals acc in every cell for this benchmark."""
    seen = False
    for r in results.values():
        for a in r.get("arms", {}).values():
            b = (a.get("bench") or {}).get(bench)
            if not b or "acc_delta" not in b or "acc_norm_delta" not in b:
                continue
            seen = True
            if abs(b["acc_delta"] - b["acc_norm_delta"]) > 1e-12:
                return False
    return seen


def table_bench_delta(results: dict, bench: str, metric: str = "acc") -> str:
    key = f"{metric}_delta"

    def val(a, r):
        b = (a.get("bench") or {}).get(bench)
        return "—" if b is None or key not in b else f"{b[key]:+.4f}"

    row_keys, cols, cell = pivot(results)
    baseline = {}
    for arch, k in row_keys:
        ob = (results[f"{arch}@k{k}"].get("original_bench") or {}).get(bench)
        if ob and metric in ob:
            baseline[(arch, k)] = ob[metric]

    out = [f"### {bench.upper()} — {metric} delta", ""]
    out.append("Delta against the target model's own score. These are small base "
               "models, so absolute accuracy sits near chance on MMLU and GPQA; the "
               "delta is the signal.")
    out.append("")
    out.append("| Model | k | baseline | " + " | ".join(cols) + " |")
    out.append("|" + "---|" * (len(cols) + 3))
    for arch, k in row_keys:
        b = baseline.get((arch, k))
        cells = [("—" if cell.get((arch, k, arm)) is None
                  else val(cell[(arch, k, arm)], None)) for arm in cols]
        out.append(f"| {_name(arch)} | {k} | "
                   f"{'—' if b is None else f'{b:.4f}'} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def table_dispersion(results: dict) -> str:
    """
    Generative arms only: did they produce codes with the right SPREAD?

    Separate from table_geometry because it answers a different question. Geometry
    asks "how far from the target", which is the wrong question for a generative
    arm -- the target is a distribution, not a point. This asks "is the spread
    right", which is the only thing that makes a generative ΔPPL readable at all.
    """
    rows = []
    for key, r in sorted(results.items()):
        arch, k = parse_key(key)
        for arm, a in r.get("arms", {}).items():
            if a.get("code_rms_ratio") is None:
                continue
            rows.append((_name(arch), k, arm, a["code_rms_ratio"],
                         a.get("code_rms"), a.get("ppl_delta_pct")))
    if not rows:
        return ""
    out = ["### Generative sample dispersion", "",
           "`rms ratio` is the RMS of the codes an arm produced, over the RMS of the "
           "real codes for that family, in normalized code space. **1.0 is correct; "
           "below 0.8 means the arm is contracting toward the family mean.** Because "
           "the ensemble mean is essentially `w_0`, a collapsed arm scores a ΔPPL "
           "near zero while generating nothing — so read this column BEFORE the ΔPPL "
           "one (misstep 19).", "",
           "| Model | k | arm | rms ratio | verdict | ΔPPL % |",
           "|---|---|---|---|---|---|"]
    for nm, k, arm, ratio, _rms, pct in sorted(rows, key=lambda t: (t[0], -t[1], t[3])):
        verdict = ("**COLLAPSED**" if ratio < 0.5 else
                   "shrunken" if ratio < 0.8 else
                   "over-dispersed" if ratio > 1.25 else "ok")
        out.append(f"| {nm} | {k} | {arm} | {ratio:.3f} | {verdict} | "
                   f"{'—' if pct is None else f'{pct:+.3f}'} |")
    out.append("")
    return "\n".join(out)


def table_geometry(results: dict) -> str:
    return _table(
        results, "Weight-space geometry",
        lambda a, r: f"{a['cosine_sim']:.6f} / {a['rel_l2']:.3g}",
        note="`cosine / relL2`. **Do not gate on these.** A measured cosine of "
             "0.99939 came with +72,468% perplexity, so cosine only becomes "
             "informative above about 0.9999, and error magnitude does not predict "
             "functional damage: the VAE's 2e-3 error cost pythia_410m +0.378% PPL "
             "while an isotropic 2e-3 perturbation costs about +0.03%.")


def table_roundtrip(results: dict) -> str:
    """flow_rt_* only: ODE self-consistency against step count."""
    rows = []
    for key, r in sorted(results.items()):
        arch, k = parse_key(key)
        for arm, a in r.get("arms", {}).items():
            if not arm.startswith("flow_rt_"):
                continue
            for ns, d in sorted((a.get("flow_rt_sweep") or {}).items(),
                                key=lambda t: int(t[0])):
                rows.append((_name(arch), k, arm, int(ns),
                             d.get("rel_l2_space"), d.get("x0_hat_rms"),
                             d.get("source_std")))
    if not rows:
        return ""
    out = ["### ODE round trip (flow diagnostics)", "",
           "Integrate x1 -> x0 -> x1. `rel_l2` folds together Euler discretisation "
           "error and the model failing to be a consistent vector field, and cannot "
           "separate them — but discretisation error falls like 1/n_steps and model "
           "inconsistency does not, so the step sweep tells you which dominates. "
           "`x0 rms` against `source std` says whether the reverse pass lands in the "
           "source distribution at all.",
           "",
           "| Model | k | arm | steps | rel_l2 | x0 rms | source std |",
           "|---|---|---|---|---|---|---|"]
    for nm, k, arm, ns, rl, rms, ss in rows:
        out.append(f"| {nm} | {k} | {arm} | {ns} | "
                   f"{'—' if rl is None else f'{rl:.4g}'} | "
                   f"{'—' if rms is None else f'{rms:.4g}'} | "
                   f"{'—' if ss is None else f'{ss:g}'} |")
    out.append("")
    return "\n".join(out)


def caveats(results: dict, manifest: Optional[dict] = None) -> List[str]:
    """
    Every warning here is a recorded misstep that already cost time once.

    A table without them invites the same misreading a third time, which is why this
    block is emitted ABOVE the tables rather than as a footnote.
    """
    out: List[str] = []
    man = manifest or {}
    n_samples = (man.get("ensemble") or {}).get("n_samples")

    sample_idxs = {r.get("sample_idx") for r in results.values()}
    if 0 in sample_idxs:
        out.append(
            "**`sample_idx = 0` is the real pretrained model, and it sits about "
            "sqrt(N) nearer the ensemble mean than a typical member.** Its truncation "
            "residual is therefore smaller by that factor, so a rank sweep measured "
            "on sample 0 understates truncation loss badly. Read the k axis on a "
            "nonzero `--sample_idx` run instead (misstep 9).")

    if n_samples:
        at_bound = sorted({k for _a, k in pivot(results)[0] if k == n_samples - 1})
        if at_bound:
            out.append(
                f"**`pca_only` at k = {n_samples - 1} = N-1 is a SELF-TEST, not a "
                f"result.** Centering removes one degree of freedom, so the rank "
                f"bound makes projection exact in-sample. Cosine 1.000000 there "
                f"proves the arithmetic works and nothing else; anything other than "
                f"1.000000 is a bug (misstep 12).")

    cells = pivot(results)[2]
    negatives = [(f"{_name(a)} k={k}", arm, c["ppl_delta_pct"])
                 for (a, k, arm), c in cells.items()
                 if c.get("ppl_delta_pct", 0) < -0.05]
    recon_neg = [n for n in negatives if n[1] in RECON_ARMS]
    if recon_neg:
        worst = min(recon_neg, key=lambda t: t[2])
        out.append(
            f"**A negative perplexity delta on a RECONSTRUCTION arm is not success** "
            f"({worst[0]}, `{worst[1]}`: {worst[2]:+.3f}%). On a manufactured noise "
            f"ensemble the evaluation target is `w_0 + noise`, i.e. a DEGRADED model, "
            f"and discarding variance discards some of that noise — so truncation "
            f"acts as a denoiser and reconstruction fidelity and model quality become "
            f"anti-correlated at low rank (misstep 14).")

    # Misstep 19. Checked BEFORE the dPPL is read, because on this ensemble the
    # collapsed arm wins the dPPL table and the honest one loses.
    collapsed = [(f"{_name(a)} k={k}", arm, c["code_rms_ratio"],
                  c.get("ppl_delta_pct"))
                 for (a, k, arm), c in cells.items()
                 if c.get("code_rms_ratio") is not None
                 and c["code_rms_ratio"] < 0.8]
    if collapsed:
        worst = min(collapsed, key=lambda t: t[2])
        out.append(
            f"**A generative arm here has COLLAPSED toward the family mean** "
            f"({worst[0]}, `{worst[1]}`: codes at {worst[2]:.2f}x the correct RMS, "
            f"ΔPPL {worst[3]:+.3f}%). `mean(w) = w_0 + O(s·σ/√N)`, so the ensemble "
            f"mean essentially IS the real pretrained model and a collapsed "
            f"generator scores ΔPPL near zero — *better* than an honest sample — "
            f"while generating nothing. **The ΔPPL ranking of generative arms is "
            f"anti-correlated with sample fidelity when this fires.** `gauss_codes` "
            f"cannot detect it, because a collapsed flow beats the Gaussian null by "
            f"construction. Read `code_rms_ratio` first, then ΔPPL (misstep 19).")
    elif any(c.get("code_rms_ratio") is not None for c in cells.values()):
        out.append(
            "Generative arms report `code_rms_ratio` and none is below 0.8, so no "
            "arm has collapsed toward the family mean and the ΔPPL column can be "
            "read at face value (misstep 19).")
    elif any(arm in GEN_ARMS for _a, _k, arm in cells):
        out.append(
            "**No `code_rms_ratio` was recorded for the generative arms, so their "
            "ΔPPL cannot be read.** A generator collapsed toward the family mean "
            "scores ΔPPL near zero while generating nothing. Re-run `eval_stack.py` "
            "(which records it per arm) or `diag_flow_dispersion.py` (misstep 19).")

    _rows, cols, _cell = pivot(results)
    if any(c.startswith("flow_") for c in cols) and "gauss_codes" not in cols:
        out.append(
            "**No `gauss_codes` arm was run, so the flow numbers cannot be read.** "
            "`gauss_codes` samples the per-family Gaussian fitted to the codes and "
            "costs one decode with zero training. Without it there is no way to tell "
            "a flow that learned structure from one that learned the prior.")

    # ev0/median, not ev0/ev[k-1]: at k = N-1 the latter is set by the smallest
    # direction surviving the rank floor and reads ~12 on a pure-noise ensemble,
    # which would suppress exactly the warning this block exists to give.
    flat = []
    for kk, spaces in (man.get("flow") or {}).items():
        for space, d in (spaces or {}).items():
            for arch, sp in (d.get("spectrum") or {}).items():
                ev = sp.get("ev0_over_median")
                if ev is not None and ev < 2.0:
                    flat.append((arch, kk, space, ev,
                                 sp.get("effective_rank_ratio")))
    if flat:
        worst = min(flat, key=lambda t: t[3])
        eff = "" if worst[4] is None else \
            f", effective-rank ratio {worst[4]:.3f} of 1.0"
        out.append(
            f"**The spectrum these flows were trained on is FLAT** "
            f"(`{worst[0]}` at k={worst[1]}, {worst[2]} space: ev0/median = "
            f"{worst[3]:.3g}{eff}). The ensemble is manufactured as `w_0 + s*sigma*eps_i`, "
            f"so the per-family code distribution is near-Gaussian BY CONSTRUCTION "
            f"and a flow fit to it may have learned nothing beyond the prior. "
            f"**Compare `flow_codes` against `gauss_codes`, not against `pca_only`.** "
            f"A real answer needs an ensemble of genuinely different complete models "
            f"(RESEARCH_NOTES Experiment 3).")

    # Benchmark resolution. At n questions the quantum of an accuracy delta is 1/n,
    # so a table full of "-0.0100" at n=200 is TWO questions and invites
    # over-reading. This is a PAIRED comparison (same questions, same model,
    # perturbed weights), so the relevant scale is the flip count, not the naive
    # binomial SE of ~0.031 at p=0.25, n=200.
    for bench in benchmarks_present(results):
        nq, deltas = None, []
        for r in results.values():
            for a in r.get("arms", {}).values():
                b = (a.get("bench") or {}).get(bench)
                if b and "acc_delta" in b:
                    deltas.append(abs(b["acc_delta"]))
                    nq = nq or b.get("n_examples")
        if not deltas or not nq:
            continue
        q = 1.0 / nq
        if max(deltas) <= 4 * q:
            out.append(
                f"**No resolvable change on {bench.upper()}.** Every accuracy delta "
                f"is within {max(deltas) / q:.0f} question(s) of zero at "
                f"n={nq} (quantum 1/{nq} = {q:.4f}). Do not read a sign or a "
                f"ranking off that column — it is consistent with no change at all. "
                f"Raise `--bench_n_questions` if a real effect is expected.")

    unverified = set()
    for section in ("vae", "flow"):
        for kk, entry in (man.get(section) or {}).items():
            entries = entry.values() if section == "flow" else [entry]
            for e in entries:
                if isinstance(e, dict) and e.get("trust") == "unverified-legacy":
                    unverified.add(f"{section} k={kk}")
    for _key, r in results.items():
        for arm, a in r.get("arms", {}).items():
            if a.get("flow_trust") == "unverified-legacy":
                unverified.add(f"arm {arm}")
    if unverified:
        out.append(
            f"**† Some artifacts are `trust = unverified-legacy`** "
            f"({', '.join(sorted(unverified))}). Those were adopted from a "
            f"pre-provenance directory, so nothing records which ensemble or which "
            f"code statistics produced them. Every number derived from them is "
            f"provisional. A full re-train is about 10 minutes on one V100.")

    return out


def build_report(results: dict, manifest: Optional[dict] = None,
                 source: Optional[str] = None) -> str:
    man = manifest or {}
    ens = man.get("ensemble") or {}
    out = ["# Whole-stack evaluation report", ""]

    if source:
        out += [f"Source: `{source}`", ""]
    if ens:
        out += ["| Setting | Value |", "|---|---|",
                f"| run | `{man.get('run_name', '?')}` |",
                f"| architectures | {', '.join(ens.get('arch_list', []))} |",
                f"| ensemble N | {ens.get('n_samples')} "
                f"(rank bound {(ens.get('n_samples') or 1) - 1}) |",
                f"| ensemble source | `{ens.get('source', 'noise')}` |",
                f"| noise scale | {ens.get('noise_scale')} |",
                f"| embeddings + LM head in D | {ens.get('include_extra')} |",
                f"| 1-D params excluded | {ens.get('exclude_1d')} |",
                f"| git commit | `{man.get('git_commit')}` |", ""]

    idxs = sorted({r.get("sample_idx") for r in results.values()})
    out += [f"Evaluation target: ensemble member {idxs}"
            + (" (member 0 is the real pretrained model)" if 0 in idxs else ""), ""]

    notes = caveats(results, manifest)
    if notes:
        out += ["## Read this before the tables", ""]
        out += [f"{i}. {n}" for i, n in enumerate(notes, start=1)]
        out.append("")

    out += ["## Tables", ""]
    for t in (table_ppl_delta(results), table_ppl_absolute(results)):
        if t:
            out.append(t)
    for bench in benchmarks_present(results):
        for metric in ("acc", "acc_norm"):
            # MMLU and GPQA are scored on single-letter choices, so length
            # normalization is a no-op there and acc_norm duplicates acc exactly.
            # Emitting both doubles the report with zero information.
            if metric == "acc_norm" and _metric_duplicates_acc(results, bench):
                continue
            t = table_bench_delta(results, bench, metric)
            if t:
                out.append(t)
    for t in (table_dispersion(results), table_geometry(results),
              table_roundtrip(results)):
        if t:
            out.append(t)
    return "\n".join(out)


def inspect(manifest: dict) -> str:
    out = ["# Run manifest", "", "```json",
           json.dumps(manifest, indent=2), "```"]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default=None,
                   help="A runs/<name>/ directory. Renders every "
                        "results/stack_eval_results*.json it contains.")
    p.add_argument("--results", nargs="+", default=None,
                   help="Explicit results JSON paths, instead of --run_dir.")
    p.add_argument("--output", "-o", default=None, help="Write here instead of stdout.")
    p.add_argument("--inspect", action="store_true",
                   help="Print the run manifest and provenance only, no tables.")
    args = p.parse_args()

    manifest = None
    paths: List[str] = []
    if args.run_dir:
        mp = os.path.join(args.run_dir, "run_manifest.json")
        if os.path.exists(mp):
            with open(mp) as f:
                manifest = json.load(f)
        paths = sorted(glob.glob(os.path.join(args.run_dir, "results",
                                              "stack_eval_results*.json")))
    if args.results:
        paths = list(args.results)
    if args.inspect:
        if manifest is None:
            raise SystemExit("--inspect needs --run_dir pointing at a run with a "
                             "run_manifest.json.")
        text = inspect(manifest)
    else:
        if not paths:
            raise SystemExit(
                "No results found. Pass --run_dir <runs/name> or --results <file>. "
                "Run eval_stack.py first.")
        chunks = []
        for path in paths:
            chunks.append(build_report(load_results(path), manifest,
                                       source=os.path.basename(path)))
        text = "\n\n---\n\n".join(chunks)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
