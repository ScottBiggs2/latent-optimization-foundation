"""
report_stack.py renders the real schema, and refuses the wrong one.

Pure stdlib -- no numpy, no torch, no downloads -- so this runs on the `cpu`
partition, and also locally against files copied down from the cluster. That purity
is a property of report_stack.py worth protecting, so this file imports nothing heavy
either. It is the only test in this suite that can run on a laptop.

The subject is misreading, not formatting. A reporting script that renders a legacy
flat-schema file as a table of blanks, or that omits the `sample_idx = 0` warning,
produces something that LOOKS like a result and is not one. Every `caveats()` check
below corresponds to a recorded misstep (9, 12, 14, 15b) that already cost time.

    python tests/test_report.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import List

# report_stack.py is a CLI in scripts/, deliberately NOT part of the llmzoo package:
# importing it must not pull the package in, or the stdlib-purity property below
# would depend on llmzoo/__init__.py staying empty forever.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))
import report_stack as rs                                        # noqa: E402

FAILS: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _raises(fn, needle: str) -> bool:
    try:
        fn()
        return False
    except Exception as exc:
        return needle in str(exc)


# ---------------------------------------------------------------------------
# Fixtures: literal copies of the schemas eval_stack.py actually writes
# ---------------------------------------------------------------------------

def _arm(pct: float, cos: float = 0.999999, rel: float = 1e-4, **extra) -> dict:
    a = {"cosine_sim": cos, "rel_l2": rel, "ppl": 21.0 * (1 + pct / 100.0),
         "ppl_delta": 21.0 * pct / 100.0, "ppl_delta_pct": pct, "ce": 3.05}
    a.update(extra)
    return a


def _entry(arch: str, k: int, sample_idx: int, arms: dict, **extra) -> dict:
    e = {"arch": arch, "k": k, "sample_idx": sample_idx,
         "original_ppl": 21.0, "ce_original": 3.04, "arms": arms}
    e.update(extra)
    return e


def results_no_bench(sample_idx: int = 5) -> dict:
    """The Phase-1-OFF path: --arms without --bench. Must render, not crash."""
    return {
        "gpt2_medium@k11": _entry("gpt2_medium", 11, sample_idx, {
            "pca_only": _arm(0.002, cos=1.000000, rel=1e-7),
            "vae": _arm(0.378, cos=0.999998, rel=2.0e-3),
        }),
        "gpt2_medium@k6": _entry("gpt2_medium", 6, sample_idx, {
            "pca_only": _arm(-0.524, cos=0.999972, rel=7.4e-3),
            "vae": _arm(0.51),
        }),
        "pythia_410m@k11": _entry("pythia_410m", 11, sample_idx, {
            "pca_only": _arm(0.001, cos=1.000000, rel=1e-7),
        }),
    }


def results_full() -> dict:
    """All eight arms, all three benchmarks, plus the flow round-trip sweep."""
    bench = {b: {"acc": 0.25, "acc_norm": 0.26, "acc_delta": -0.005,
                 "acc_norm_delta": 0.0, "n_examples": 200}
             for b in ("mmlu", "hellaswag", "gpqa")}
    rt_extra = {"flow_rt_sweep": {"5": {"rel_l2_space": 4.1e-2, "x0_hat_rms": 1.02,
                                        "source_std": 1.0, "n_steps": 5},
                                  "50": {"rel_l2_space": 3.9e-3, "x0_hat_rms": 1.00,
                                         "source_std": 1.0, "n_steps": 50}},
                "flow_trust": "verified"}
    arms = {
        "pca_only": _arm(0.002, bench=bench),
        "vae": _arm(0.378, bench=bench),
        "generate": _arm(8.4, bench=bench),
        "gauss_codes": _arm(7.9, bench=bench),
        "flow_codes": _arm(7.8, bench=bench, flow_trust="verified"),
        "flow_latent": _arm(8.1, bench=bench, flow_trust="verified"),
        "flow_rt_codes": _arm(0.05, bench=bench, **rt_extra),
        "flow_rt_latent": _arm(0.41, bench=bench, **rt_extra),
    }
    return {
        "pythia_410m@k99": _entry(
            "pythia_410m", 99, 5, arms,
            bench_config={"benchmarks": ["mmlu", "hellaswag", "gpqa"],
                          "n_questions": 200, "seed": 0},
            original_bench={b: {"acc": 0.255, "acc_norm": 0.26, "n_examples": 200}
                            for b in ("mmlu", "hellaswag", "gpqa")}),
    }


def manifest(n_samples: int = 12, flat: bool = False,
             unverified: bool = False) -> dict:
    spec = {"gpt2_medium": {"ev0_over_median": 1.006 if flat else 41.2,
                            "effective_rank_ratio": 0.924 if flat else 0.19,
                            "ev0_over_evlast": 12.09,
                            "variance_captured_at_k": 0.51}}
    return {
        "manifest_version": 1, "run_name": "t", "git_commit": "deadbeef",
        "ensemble": {"n_samples": n_samples, "arch_list": ["gpt2_medium",
                                                           "pythia_410m"],
                     "noise_scale": 1e-2, "include_extra": True,
                     "exclude_1d": True, "source": "noise", "mode": "full"},
        "vae": {"11": {"trust": "unverified-legacy" if unverified else "verified"}},
        "flow": {"11": {"codes": {"trust": "verified", "spectrum": spec},
                        "latent": {"trust": "verified", "spectrum": spec}}},
    }


# ---------------------------------------------------------------------------
# Keys and pivoting
# ---------------------------------------------------------------------------

def test_parse_key() -> None:
    print("\n--- parse_key ---")
    check("parses arch@kNN", rs.parse_key("pythia_410m@k99") == ("pythia_410m", 99))
    check("parses a single-digit rank",
          rs.parse_key("gpt2_medium@k6") == ("gpt2_medium", 6))
    check("parses an arch name containing digits and underscores",
          rs.parse_key("smollm2_360m@k50") == ("smollm2_360m", 50))

    # The load-bearing refusal. report.py's files are keyed by bare arch name; if
    # this parsed, every stack column would render '—' and the output would look
    # like a run where nothing worked rather than like the wrong file.
    check("refuses a legacy flat key and names report.py",
          _raises(lambda: rs.parse_key("pythia_410m"), "report.py"))
    check("refuses a missing rank", _raises(lambda: rs.parse_key("gpt2@k"),
                                           "not a stack results key"))
    check("refuses a non-numeric rank",
          _raises(lambda: rs.parse_key("gpt2@kfull"), "not a stack results key"))


def test_pivot() -> None:
    print("\n--- pivot ---")
    rows, cols, cell = rs.pivot(results_no_bench())
    check("rows are (arch, k) sorted by arch then DESCENDING k",
          rows == [("gpt2_medium", 11), ("gpt2_medium", 6), ("pythia_410m", 11)],
          str(rows))
    # Descending k puts the rank bound first, which is where the self-test caveat
    # applies -- so the reader meets the self-test before the real number.
    check("columns follow ARM_ORDER, not dict insertion order",
          cols == ["pca_only", "vae"], str(cols))
    check("cell lookup is keyed (arch, k, arm)",
          abs(cell[("gpt2_medium", 6, "pca_only")]["ppl_delta_pct"] + 0.524) < 1e-9)
    check("a missing (arch, k, arm) is simply absent",
          ("pythia_410m", 11, "vae") not in cell)

    _r, cols8, _c = rs.pivot(results_full())
    check("all eight arms order as declared", cols8 == rs.ARM_ORDER, str(cols8))

    # An arm this file has never heard of must still render, after the known ones.
    _r, cols_x, _c = rs.pivot({"a@k1": _entry("a", 1, 0, {"zzz_new_arm": _arm(1.0),
                                                          "vae": _arm(1.0)})})
    check("an unknown arm sorts after the known ones",
          cols_x == ["vae", "zzz_new_arm"], str(cols_x))

    check("an empty results dict pivots to nothing", rs.pivot({}) == ([], [], {}))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_tables_no_bench() -> None:
    print("\n--- tables with NO benchmark data (the --bench-off path) ---")
    r = results_no_bench()
    check("benchmarks_present is empty", rs.benchmarks_present(r) == [])

    t = rs.table_ppl_delta(r)
    check("ppl delta table renders", t.startswith("### Perplexity delta"))
    check("ppl delta is signed", "+0.378" in t and "-0.524" in t)
    check("a missing arm renders as an em dash", "—" in t)
    check("row count matches the pivot",
          len([ln for ln in t.splitlines() if ln.startswith("| ")]) == 4,
          "3 data rows + 1 header")

    check("absolute ppl table carries the target column",
          "target PPL" in rs.table_ppl_absolute(r))
    check("geometry table restates misstep 11 rather than just printing cosine",
          "Do not gate on these" in rs.table_geometry(r))

    # The whole point: no bench section, no crash, no empty benchmark headings.
    body = rs.build_report(r, manifest())
    check("build_report emits no benchmark section when there is no bench data",
          "— acc delta" not in body)
    check("build_report still emits the ppl tables",
          "Perplexity delta" in body and "Perplexity, absolute" in body)
    check("round-trip table is empty without flow_rt arms",
          rs.table_roundtrip(r) == "")

    # An entry with an EMPTY arms dict is what a crashed-mid-arch run leaves behind.
    check("an entry with no arms does not crash the renderer",
          isinstance(rs.build_report({"gpt2_medium@k11":
                                      _entry("gpt2_medium", 11, 5, {})},
                                     manifest()), str))
    check("build_report on an empty results dict does not crash",
          isinstance(rs.build_report({}, manifest()), str))


def test_tables_full() -> None:
    print("\n--- tables with all arms and all benchmarks ---")
    r = results_full()
    check("benchmarks_present finds all three",
          rs.benchmarks_present(r) == ["gpqa", "hellaswag", "mmlu"], )

    body = rs.build_report(r, manifest(n_samples=100))
    for b in ("MMLU", "HELLASWAG", "GPQA"):
        check(f"{b} appears as its own table", f"### {b} — acc delta" in body)
    check("both acc and acc_norm tables render",
          "— acc delta" in body and "— acc_norm delta" in body)
    check("the benchmark baseline column is populated", "0.2550" in body)
    check("benchmark deltas are signed to 4 dp", "-0.0050" in body)

    rt = rs.table_roundtrip(r)
    # Rows are keyed on the arm name, not the arch: _name() maps through
    # ARCH_DISPLAY_NAMES, so 'pythia_410m' renders as 'EleutherAI/pythia-410m'.
    check("round-trip table renders one row per (arm, n_steps)",
          len([ln for ln in rt.splitlines()
               if ln.startswith("| ") and "flow_rt_" in ln]) == 4,
          "2 arms x 2 step counts")
    check("round-trip steps render ascending", rt.index("| 5 |") < rt.index("| 50 |"))
    check("round-trip table explains the 1/n_steps discriminator",
          "falls like 1/n_steps" in rt)

    # A PARTIAL benchmark dict is the realistic failure: --bench mmlu on one job and
    # --bench mmlu hellaswag on another, merged. The gaps must be visible.
    partial = json.loads(json.dumps(r))
    del partial["pythia_410m@k99"]["arms"]["flow_codes"]["bench"]["mmlu"]
    t = rs.table_bench_delta(partial, "mmlu")
    check("a partial benchmark dict renders an em dash in the gap", "—" in t)
    check("and still renders the arms that do have data", "-0.0050" in t)


# ---------------------------------------------------------------------------
# Caveats -- one check per recorded misstep
# ---------------------------------------------------------------------------

def test_caveats() -> None:
    print("\n--- caveats: each recorded misstep fires independently ---")

    # Misstep 9: member 0 sits at the ensemble centre.
    c0 = " ".join(rs.caveats(results_no_bench(sample_idx=0), manifest()))
    check("misstep 9 fires on sample_idx = 0", "misstep 9" in c0)
    c5 = " ".join(rs.caveats(results_no_bench(sample_idx=5), manifest()))
    check("misstep 9 is silent on a nonzero sample_idx", "misstep 9" not in c5)

    # Misstep 12: pca_only at the rank bound is a self-test.
    check("misstep 12 fires when a row sits at k = N-1",
          "misstep 12" in " ".join(rs.caveats(results_no_bench(),
                                              manifest(n_samples=12))))
    check("misstep 12 is silent when no row is at the rank bound",
          "misstep 12" not in " ".join(rs.caveats(results_no_bench(),
                                                  manifest(n_samples=100))))
    check("misstep 12 is silent when the manifest has no N",
          "misstep 12" not in " ".join(rs.caveats(results_no_bench(), {})))

    # Misstep 14: negative dPPL is truncation denoising a noisy target.
    check("misstep 14 fires on a negative delta and names the worst offender",
          "misstep 14" in " ".join(rs.caveats(results_no_bench(), manifest())))
    positive = {"a@k5": _entry("a", 5, 5, {"pca_only": _arm(0.3)})}
    check("misstep 14 is silent when every delta is positive",
          "misstep 14" not in " ".join(rs.caveats(positive, manifest())))

    # Misstep 15b: gate on the BULK statistic, not ev0/ev[k-1].
    flat = " ".join(rs.caveats(results_full(), manifest(n_samples=100, flat=True)))
    check("the flat-spectrum warning fires on ev0/median < 2",
          "spectrum these flows were trained on is FLAT" in flat)
    check("it says to compare against gauss_codes, not pca_only",
          "against `gauss_codes`, not against `pca_only`" in flat)
    check("it reports the effective-rank ratio too", "effective-rank ratio" in flat)
    steep = " ".join(rs.caveats(results_full(), manifest(n_samples=100, flat=False)))
    check("the flat-spectrum warning is silent on a steep spectrum",
          "trained on is FLAT" not in steep)
    # The trap itself: ev0_over_evlast = 12.09 is in BOTH fixtures. If caveats() ever
    # regresses to reading it, the steep case above would still warn and the flat
    # case would be indistinguishable -- so this pair is the regression guard.
    check("caveats does not read ev0_over_evlast (misstep 15b regression guard)",
          ("trained on is FLAT" in flat) and ("trained on is FLAT" not in steep))

    # The gauss_codes null model.
    no_null = json.loads(json.dumps(results_full()))
    del no_null["pythia_410m@k99"]["arms"]["gauss_codes"]
    check("a missing gauss_codes arm is called out when flow arms are present",
          "No `gauss_codes` arm was run" in
          " ".join(rs.caveats(no_null, manifest(n_samples=100))))
    check("gauss_codes is not demanded when there are no flow arms",
          "No `gauss_codes`" not in
          " ".join(rs.caveats(results_no_bench(), manifest())))

    # Unverified-legacy trust, from the manifest and from an arm.
    check("unverified-legacy in the manifest raises the † footnote",
          "unverified-legacy" in
          " ".join(rs.caveats(results_no_bench(),
                              manifest(unverified=True))))
    tainted = json.loads(json.dumps(results_full()))
    tainted["pythia_410m@k99"]["arms"]["flow_codes"]["flow_trust"] = \
        "unverified-legacy"
    check("unverified-legacy on an arm raises it too",
          "unverified-legacy" in
          " ".join(rs.caveats(tainted, manifest(n_samples=100))))

    check("caveats works with no manifest at all",
          isinstance(rs.caveats(results_no_bench(), None), list))
    check("caveats appear ABOVE the tables in the rendered report",
          rs.build_report(results_no_bench(sample_idx=0), manifest())
          .index("Read this before the tables") <
          rs.build_report(results_no_bench(sample_idx=0), manifest())
          .index("## Tables"))


# ---------------------------------------------------------------------------
# The header block and the CLI
# ---------------------------------------------------------------------------

def test_dispersion_and_collapse() -> None:
    """misstep 19: the caveat that reverses how the dPPL column should be read."""
    print("\n--- dispersion / collapse caveat (misstep 19) ---")

    # The real emb3 numbers: flow_codes beat the gauss_codes null by 43x on dPPL
    # while emitting codes at 0.25x the correct RMS.
    r = {"gpt2_medium@k99": _entry("gpt2_medium", 99, 5, {
        "pca_only": _arm(-0.018),
        "gauss_codes": _arm(-0.177, code_rms=0.9957, real_code_rms=0.995,
                            code_rms_ratio=1.0007),
        "flow_codes": _arm(-0.508, code_rms=0.2475, real_code_rms=0.995,
                           code_rms_ratio=0.249),
        "flow_latent": _arm(-0.427, code_rms=0.6766, real_code_rms=0.995,
                            code_rms_ratio=0.680),
    })}
    c = " ".join(rs.caveats(r, manifest(n_samples=100)))
    check("collapse caveat fires on code_rms_ratio < 0.8",
          "has COLLAPSED toward the family mean" in c)
    check("it names the worst arm and its ratio",
          "flow_codes" in c and "0.25x the correct RMS" in c)
    check("it states the dPPL ranking is anti-correlated with fidelity",
          "anti-correlated with sample fidelity" in c)
    check("it says gauss_codes cannot detect this",
          "gauss_codes` cannot detect it" in c)

    # A generative arm's negative dPPL must NOT be attributed to truncation
    # denoising -- that was a real wording bug the emb3 report exposed, where
    # caveats() blamed `flow_codes: -0.508%` on truncation.
    check("the misstep-14 caveat is NOT raised for a generative-only negative",
          "RECONSTRUCTION arm is not success" not in c)
    r2 = {"a@k99": _entry("a", 99, 5, {"pca_only": _arm(-0.320)})}
    c2 = " ".join(rs.caveats(r2, manifest(n_samples=100)))
    check("the misstep-14 caveat IS raised for a reconstruction-arm negative",
          "RECONSTRUCTION arm is not success" in c2 and "misstep 14" in c2)
    check("and it names the reconstruction arm",
          "`pca_only`" in c2)

    # Healthy dispersion must say so explicitly, not stay silent -- silence reads
    # as "not checked".
    healthy = {"a@k99": _entry("a", 99, 5, {
        "gauss_codes": _arm(0.3, code_rms_ratio=1.0007),
        "flow_codes": _arm(0.2, code_rms_ratio=0.95)})}
    ch = " ".join(rs.caveats(healthy, manifest(n_samples=100)))
    check("a healthy run states no arm collapsed",
          "none is below 0.8" in ch and "at face value" in ch)
    check("and does not raise the collapse warning", "has COLLAPSED" not in ch)

    # Generative arms with NO ratio recorded is the dangerous case: it is exactly
    # the emb3 sample-0 file, produced before the metric existed.
    missing = {"a@k99": _entry("a", 99, 5, {"flow_codes": _arm(0.008)})}
    cm = " ".join(rs.caveats(missing, manifest(n_samples=100)))
    check("a missing code_rms_ratio on a generative arm is called out",
          "No `code_rms_ratio` was recorded" in cm)
    check("and it names both ways to get it",
          "eval_stack.py" in cm and "diag_flow_dispersion.py" in cm)
    recon_only = {"a@k99": _entry("a", 99, 5, {"pca_only": _arm(0.01)})}
    check("no dispersion demand when there are no generative arms",
          "code_rms_ratio" not in
          " ".join(rs.caveats(recon_only, manifest(n_samples=100))))

    t = rs.table_dispersion(r)
    check("dispersion table renders one row per generative arm",
          len([ln for ln in t.splitlines()
               if ln.startswith("| ") and "0." in ln]) == 3, t)
    check("dispersion table flags the collapsed arm in bold",
          "**COLLAPSED**" in t)
    check("dispersion table marks the honest arm ok", "| ok |" in t)
    check("dispersion table says to read it before dPPL",
          "BEFORE the ΔPPL" in t)
    check("dispersion table is empty without any ratios",
          rs.table_dispersion(results_no_bench()) == "")
    check("build_report includes the dispersion table when ratios exist",
          "Generative sample dispersion" in
          rs.build_report(r, manifest(n_samples=100)))


def test_bench_resolution() -> None:
    """A table of two-question deltas must not read as an effect."""
    print("\n--- benchmark resolution caveat ---")

    def with_deltas(d_acc, n=200):
        b = {"mmlu": {"acc": 0.25, "acc_norm": 0.25, "acc_delta": d_acc,
                      "acc_norm_delta": d_acc, "n_examples": n}}
        return {"a@k99": _entry("a", 99, 5, {"pca_only": _arm(0.01, bench=b)},
                                original_bench={"mmlu": {"acc": 0.25,
                                                         "acc_norm": 0.25,
                                                         "n_examples": n}})}

    # emb3 measured deltas of 0.000 to -0.020 at n=200: 0 to 4 questions.
    c = " ".join(rs.caveats(with_deltas(-0.010), manifest(n_samples=100)))
    check("fires when every delta is within a few questions of zero",
          "No resolvable change on MMLU" in c)
    check("it states the quantum", "quantum 1/200" in c)
    check("it converts the delta to questions", "question(s) of zero" in c)
    check("it says not to read a sign or ranking off the column",
          "Do not read a sign or a ranking" in c)
    check("it suggests raising bench_n_questions",
          "--bench_n_questions" in c)

    big = " ".join(rs.caveats(with_deltas(-0.15), manifest(n_samples=100)))
    check("silent when a delta is well above the quantum",
          "No resolvable change" not in big)
    check("silent when there is no benchmark data at all",
          "No resolvable change" not in
          " ".join(rs.caveats(results_no_bench(), manifest())))


def test_acc_norm_dedup() -> None:
    """MMLU/GPQA score single-letter choices, so acc_norm duplicates acc exactly."""
    print("\n--- acc_norm table dedup ---")
    dup = json.loads(json.dumps(results_full()))
    for a in dup["pythia_410m@k99"]["arms"].values():
        for b in a["bench"].values():
            b["acc_norm_delta"] = b["acc_delta"]
    check("_metric_duplicates_acc detects an exact duplicate",
          rs._metric_duplicates_acc(dup, "mmlu"))
    body = rs.build_report(dup, manifest(n_samples=100))
    check("only the acc table renders when acc_norm duplicates it",
          "MMLU — acc delta" in body and "MMLU — acc_norm delta" not in body)

    # HellaSwag's differ in reality (0.330 vs 0.360 baseline), so both must render.
    check("_metric_duplicates_acc is False when they differ",
          not rs._metric_duplicates_acc(results_full(), "mmlu"))
    body2 = rs.build_report(results_full(), manifest(n_samples=100))
    check("both tables render when the metrics genuinely differ",
          "MMLU — acc delta" in body2 and "MMLU — acc_norm delta" in body2)
    check("_metric_duplicates_acc is False when there is no data",
          not rs._metric_duplicates_acc(results_no_bench(), "mmlu"))


def test_header_and_cli() -> None:
    print("\n--- header block and file I/O ---")
    body = rs.build_report(results_no_bench(), manifest(), source="x.json")
    for want in ("ensemble N", "rank bound 11", "ensemble source", "`noise`",
                 "embeddings + LM head in D", "git commit"):
        check(f"header states {want!r}", want in body)
    check("header names the evaluation target member",
          "Evaluation target: ensemble member [5]" in body)
    check("header flags member 0 as the real pretrained model",
          "real pretrained model" in
          rs.build_report(results_no_bench(sample_idx=0), manifest()))
    check("build_report works with no manifest",
          isinstance(rs.build_report(results_no_bench(), None), str))

    tmp = tempfile.mkdtemp()
    try:
        good = os.path.join(tmp, "stack_eval_results.json")
        with open(good, "w") as f:
            json.dump(results_full(), f)
        check("load_results accepts a valid file",
              set(rs.load_results(good)) == {"pythia_410m@k99"})

        # A legacy block-pipeline file must be rejected at LOAD time, before any
        # table is built -- validating every key up front is what makes that true.
        legacy = os.path.join(tmp, "lm_eval_results.json")
        with open(legacy, "w") as f:
            json.dump({"pythia_410m": {"perplexity": 21.3, "ce_loss": 3.06},
                       "gpt2_medium": {"perplexity": 19.4}}, f)
        check("load_results refuses a legacy flat-schema file",
              _raises(lambda: rs.load_results(legacy), "report.py"))

        # One bad key among many good ones must still refuse; a partial render
        # would silently drop a whole architecture.
        mixed = os.path.join(tmp, "mixed.json")
        with open(mixed, "w") as f:
            payload = dict(results_full())
            payload["pythia_410m"] = {"perplexity": 21.3}
            json.dump(payload, f)
        check("one bad key among good ones still refuses",
              _raises(lambda: rs.load_results(mixed), "not a stack results key"))

        check("inspect renders the manifest as json",
              "manifest_version" in rs.inspect(manifest()))
    finally:
        shutil.rmtree(tmp)


def test_stdlib_purity() -> None:
    print("\n--- stdlib purity (so this runs on `cpu`, and locally) ---")
    # report_stack.py must never grow a numpy/torch/llmzoo import: that would break
    # both the `cpu`-partition test and rendering locally from copied-down files.
    src = open(os.path.join(os.path.dirname(os.path.abspath(rs.__file__)),
                            "report_stack.py")).read()
    for mod in ("numpy", "torch", "transformers", "llmzoo"):
        check(f"report_stack.py does not import {mod}",
              f"import {mod}" not in src)
    check("sys.modules has no torch after importing report_stack",
          "torch" not in sys.modules)
    check("sys.modules has no numpy after importing report_stack",
          "numpy" not in sys.modules)

    # The same contract, for the same reason, on every other reporting script.
    # report_retrieval.py is §6.4's control -- the arm that decides whether any
    # generative number means anything -- so it must never become the reason a
    # report needs a GPU.
    here = os.path.dirname(os.path.abspath(rs.__file__))
    for name in ("report_retrieval.py", "report_singleton_probe.py",
                 "report_beta_calibration.py"):
        path = os.path.join(here, name)
        if not os.path.exists(path):
            continue
        body = open(path).read()
        for mod in ("numpy", "torch", "transformers", "llmzoo"):
            check(f"{name} does not import {mod}", f"import {mod}" not in body)


def main() -> int:
    test_parse_key()
    test_pivot()
    test_tables_no_bench()
    test_tables_full()
    test_caveats()
    test_dispersion_and_collapse()
    test_bench_resolution()
    test_acc_norm_dedup()
    test_header_and_cli()
    test_stdlib_purity()

    print("\n" + "=" * 64)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
