"""
Aggregate evaluation results into a single markdown report.

Reads whichever of these exist in --results_dir and renders each as a
markdown table:
  reconstruction_results.json  (evaluate.py::evaluate_all)
  lm_eval_results.json         (eval_lm.py::evaluate_all_families)
  mc_eval_results.json         (eval_mc.py::evaluate_all_families_mc)

Pure stdlib — no torch/transformers import — safe to run locally against
result files copied down from Explorer HPC (e.g. via scp), independent of
whether a training/eval job has actually run.

Usage
-----
    python report.py --results_dir /scratch/biggs.s/llm_vae/results
    python report.py --results_dir ./results --output report.md
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

# Mirrors models/registry.py ARCH_CONFIGS default_model_id — kept as a plain
# dict here (rather than importing models.registry) so this script has zero
# heavy imports.
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


def _display_name(arch: str) -> str:
    return ARCH_DISPLAY_NAMES.get(arch, arch)


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def format_reconstruction_table(results: dict) -> str:
    per_family = results.get("per_family", {})
    lines = ["| Model | Cosine sim | MSE |", "|---|---|---|"]
    for arch, m in per_family.items():
        lines.append(f"| {_display_name(arch)} | {m['cosine_sim']:.4f} | {m['mse']:.3e} |")
    return "\n".join(lines)


def format_ppl_table(results: dict) -> str:
    lines = ["| Model | Original PPL | Reconstructed PPL | Δ PPL | Δ % |", "|---|---|---|---|---|"]
    for arch, r in results.items():
        lines.append(
            f"| {_display_name(arch)} | {r['original_ppl']:.3f} | {r['reconstructed_ppl']:.3f} | "
            f"{r['ppl_delta']:+.3f} | {r['ppl_delta_pct']:+.2f}% |"
        )
    return "\n".join(lines)


def format_mc_table(results: dict, benchmark: str) -> str:
    lines = [
        "| Model | Original acc | Reconstructed acc | Δ acc | "
        "Original acc_norm | Reconstructed acc_norm | Δ acc_norm | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arch, per_arch in results.items():
        if benchmark not in per_arch:
            continue
        r = per_arch[benchmark]
        lines.append(
            f"| {_display_name(arch)} | {r['original_acc']:.3f} | {r['reconstructed_acc']:.3f} | "
            f"{r['acc_delta']:+.3f} | {r['original_acc_norm']:.3f} | {r['reconstructed_acc_norm']:.3f} | "
            f"{r['acc_norm_delta']:+.3f} | {r['n_examples']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(results_dir: str) -> str:
    recon = _load_json(os.path.join(results_dir, "reconstruction_results.json"))
    lm    = _load_json(os.path.join(results_dir, "lm_eval_results.json"))
    mc    = _load_json(os.path.join(results_dir, "mc_eval_results.json"))

    sections = [f"# Evaluation Report\n\nSource: `{results_dir}`"]

    if recon:
        sections.append("## Weight Reconstruction\n\n" + format_reconstruction_table(recon))
    else:
        sections.append("## Weight Reconstruction\n\n_reconstruction_results.json not found_")

    if lm:
        sections.append("## Perplexity (WikiText-2)\n\n" + format_ppl_table(lm))
    else:
        sections.append("## Perplexity (WikiText-2)\n\n_lm_eval_results.json not found_")

    if mc:
        benchmarks = sorted({b for per_arch in mc.values() for b in per_arch if b != "arch"})
        for bench in benchmarks:
            sections.append(f"## {bench.upper()}\n\n" + format_mc_table(mc, bench))
    else:
        sections.append(
            "## Multiple-Choice Benchmarks\n\n"
            "_mc_eval_results.json not found (run with --eval_mc / MC_EVAL=1)_"
        )

    return "\n\n".join(sections) + "\n"


def main():
    p = argparse.ArgumentParser(description="Aggregate LLM-VAE eval results into a markdown report")
    p.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory containing reconstruction_results.json / lm_eval_results.json / mc_eval_results.json",
    )
    p.add_argument("--output", type=str, default=None, help="Write report to this file instead of stdout")
    args = p.parse_args()

    report = build_report(args.results_dir)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
