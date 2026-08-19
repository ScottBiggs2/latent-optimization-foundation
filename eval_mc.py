"""
Multiple-choice benchmark evaluation for reconstructed models.

For each model family:
  1. Load the pretrained model
  2. Measure MMLU / HellaSwag / GPQA accuracy on the original weights
  3. Reconstruct each transformer block through PCA + VAE
  4. Re-measure accuracy on the reconstructed weights
  5. Compare original vs. reconstructed

All three benchmarks are scored with the same log-likelihood ranking
primitive (score_choices): no chat templates, since all registered
architectures are base pretrained models. See data/mc_loader.py for the
per-benchmark prompt construction.
"""

from __future__ import annotations

import gc
import json
import os
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from data.block_dataset import BlockDataset
from data.mc_loader import LOADERS, MCExample
from dual_pca import BatchedCovariancePCA
from eval_lm import generate_model_blocks, reconstruct_model_blocks
from models.registry import get_arch_config, load_model
from vae import ConditionedBlockVAE
import wandb_utils as wb


# ---------------------------------------------------------------------------
# Log-likelihood choice scoring
# ---------------------------------------------------------------------------

def get_max_context_length(model: nn.Module, default: int = 1024) -> int:
    """
    Read the model's max sequence length from its config.

    Different architectures name this differently: GPT-2 uses n_positions/
    n_ctx, Llama/Qwen/OPT-style configs use max_position_embeddings.
    """
    cfg = model.config
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        val = getattr(cfg, attr, None)
        if val:
            return int(val)
    return default


def score_choices(
    model: nn.Module,
    tokenizer,
    context: str,
    choices: list[str],
    device: torch.device,
    max_length: int = 1024,
) -> Tuple[list[float], list[float]]:
    """
    Score each choice as a continuation of context by log-likelihood.

    context and context+choice are encoded jointly and the continuation
    token ids are recovered by slicing off the context-length prefix —
    this avoids BPE boundary mismatches from encoding pieces separately
    (the standard lm-eval-harness trick).

    Few-shot contexts (MMLU/GPQA) can exceed a small model's max position
    embeddings (e.g. GPT-2-medium's 1024), which crashes the forward pass
    with an out-of-range position index. When the joint encoding would
    exceed max_length, the context is truncated from the left (dropping
    the oldest few-shot examples first) so the full continuation is always
    scored intact.

    Returns
    -------
    (sum_logprobs, mean_logprobs) — one value per choice. sum_logprobs
    gives raw accuracy; mean_logprobs (length-normalized) gives acc_norm.
    """
    context_ids = tokenizer.encode(context)
    context_len = len(context_ids)

    sum_logprobs: list[float] = []
    mean_logprobs: list[float] = []

    for choice in choices:
        full_ids = tokenizer.encode(context + choice)
        continuation_ids = full_ids[context_len:]
        if len(continuation_ids) == 0:
            sum_logprobs.append(float("-inf"))
            mean_logprobs.append(float("-inf"))
            continue

        if len(full_ids) > max_length:
            keep_context = max(0, max_length - len(continuation_ids))
            full_ids = context_ids[-keep_context:] + continuation_ids if keep_context > 0 else continuation_ids[-max_length:]

        input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # (1, T, V)

        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # predicts tokens[1:]
        target_ids = input_ids[:, 1:]  # (1, T-1)
        token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T-1,)

        cont_len = len(continuation_ids)
        cont_logprobs = token_logprobs[-cont_len:]
        total = float(cont_logprobs.sum().item())

        sum_logprobs.append(total)
        mean_logprobs.append(total / cont_len)

    return sum_logprobs, mean_logprobs


def compute_mc_accuracy(
    model: nn.Module,
    tokenizer,
    examples: Iterable[MCExample],
    device: torch.device,
    max_length: int = 1024,
) -> dict:
    """
    Score a set of MCExamples and return acc / acc_norm.

    acc      : argmax by raw summed log-likelihood
    acc_norm : argmax by length-normalized log-likelihood
    """
    n = 0
    n_correct = 0
    n_correct_norm = 0

    for ex in examples:
        sum_lp, mean_lp = score_choices(model, tokenizer, ex.context, ex.choices, device, max_length=max_length)
        pred = max(range(len(sum_lp)), key=lambda i: sum_lp[i])
        pred_norm = max(range(len(mean_lp)), key=lambda i: mean_lp[i])
        n += 1
        if pred == ex.gold_idx:
            n_correct += 1
        if pred_norm == ex.gold_idx:
            n_correct_norm += 1

    if n == 0:
        return {"acc": float("nan"), "acc_norm": float("nan"), "n_examples": 0}

    return {"acc": n_correct / n, "acc_norm": n_correct_norm / n, "n_examples": n}


# ---------------------------------------------------------------------------
# Per-family evaluation
# ---------------------------------------------------------------------------

def evaluate_family_mc(
    arch: str,
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    max_block_size: int,
    benchmarks: tuple[str, ...] = ("mmlu", "hellaswag", "gpqa"),
    n_questions: int = 200,
    hf_cache: Optional[str] = None,
    block_transform_fn: Callable = reconstruct_model_blocks,
) -> dict:
    """
    Full before/after multiple-choice accuracy evaluation for one family.

    block_transform_fn replaces the model's blocks in-place before the second
    measurement — pass `reconstruct_model_blocks` (default; VAE round-trips
    each real block) or `generate_model_blocks` (samples fresh blocks from
    the VAE's prior instead, testing generative rather than reconstructive
    quality).

    Returns a dict keyed by benchmark name, each holding original/
    reconstructed acc and acc_norm plus their deltas.
    """
    device = torch.device(
        next(vae.parameters()).device if hasattr(vae, "block_idx_emb") else "cpu"
    )

    model = load_model(arch, cache_dir=hf_cache)
    model = model.to(device)
    model.eval()

    cfg = get_arch_config(arch)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["default_model_id"],
        cache_dir=hf_cache,
        trust_remote_code=True,
    )

    max_length = get_max_context_length(model)
    print(f"  [{arch}] Model max context length: {max_length}")

    examples_by_bench: dict[str, list[MCExample]] = {}
    for bench in benchmarks:
        print(f"  [{arch}] Loading {bench} ({n_questions} questions) …")
        examples_by_bench[bench] = LOADERS[bench](n_questions=n_questions, cache_dir=hf_cache)

    original: dict[str, dict] = {}
    for bench, examples in examples_by_bench.items():
        print(f"  [{arch}] Measuring original {bench} accuracy …")
        original[bench] = compute_mc_accuracy(model, tokenizer, examples, device, max_length=max_length)
        print(f"  [{arch}] Original {bench}: acc={original[bench]['acc']:.3f}  "
              f"acc_norm={original[bench]['acc_norm']:.3f}")

    print(f"  [{arch}] Transforming blocks ({block_transform_fn.__name__}) …")
    block_transform_fn(model, arch, pca, vae, max_block_size, device)

    reconstructed: dict[str, dict] = {}
    for bench, examples in examples_by_bench.items():
        print(f"  [{arch}] Measuring reconstructed {bench} accuracy …")
        reconstructed[bench] = compute_mc_accuracy(model, tokenizer, examples, device, max_length=max_length)
        print(f"  [{arch}] Reconstructed {bench}: acc={reconstructed[bench]['acc']:.3f}  "
              f"acc_norm={reconstructed[bench]['acc_norm']:.3f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {"arch": arch}
    for bench in benchmarks:
        orig, recon = original[bench], reconstructed[bench]
        results[bench] = {
            "original_acc":          orig["acc"],
            "original_acc_norm":     orig["acc_norm"],
            "reconstructed_acc":     recon["acc"],
            "reconstructed_acc_norm": recon["acc_norm"],
            "acc_delta":             recon["acc"] - orig["acc"],
            "acc_norm_delta":        recon["acc_norm"] - orig["acc_norm"],
            "n_examples":            orig["n_examples"],
        }
        wb.log({
            f"mc/{arch}/{bench}/original_acc": orig["acc"],
            f"mc/{arch}/{bench}/reconstructed_acc": recon["acc"],
            f"mc/{arch}/{bench}/acc_delta": results[bench]["acc_delta"],
            f"mc/{arch}/{bench}/acc_norm_delta": results[bench]["acc_norm_delta"],
        })
    return results


# ---------------------------------------------------------------------------
# All families
# ---------------------------------------------------------------------------

def evaluate_all_families_mc(
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    dataset: BlockDataset,
    benchmarks: tuple[str, ...] = ("mmlu", "hellaswag", "gpqa"),
    n_questions: int = 200,
    artifact_dir: str = "/scratch/biggs.s/llm_vae",
    arch_list: Optional[list[str]] = None,
    block_transform_fn: Callable = reconstruct_model_blocks,
) -> dict:
    """Run MC benchmark eval for every arch in arch_list (default: dataset.arch_list)."""
    # Share the same cache as the extract / LM-eval stages (HF_HOME) instead
    # of a separate artifact_dir-local one — see eval_lm.evaluate_all_families.
    hf_cache = os.environ.get("HF_HOME", artifact_dir + "/hf_cache")
    results = {}
    for arch in (arch_list or dataset.arch_list):
        results[arch] = evaluate_family_mc(
            arch, pca, vae,
            max_block_size=dataset.max_block_size,
            benchmarks=benchmarks,
            n_questions=n_questions,
            hf_cache=hf_cache,
            block_transform_fn=block_transform_fn,
        )
    return results


# ---------------------------------------------------------------------------
# Standalone CLI — hook into either raw HF checkpoints or a trained PCA+VAE
# ---------------------------------------------------------------------------

def _load_pca_vae_dataset(artifact_dir: str, pca_dir: Optional[str], vae_dir: Optional[str]):
    """
    Load a trained PCA + VAE + the BlockDataset they were trained on, straight
    from train.py's saved artifacts — no re-extraction/re-training needed.
    Mirrors train.py's own resume path (_load_existing_dataset).
    """
    import argparse as _argparse
    import train as T

    blocks_dir = os.path.join(artifact_dir, "blocks")
    with open(os.path.join(blocks_dir, "dataset_meta.json")) as f:
        meta = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca = BatchedCovariancePCA.load(pca_dir or os.path.join(artifact_dir, "pca"), device=device)

    vdir = vae_dir or os.path.join(artifact_dir, "vae")
    with open(os.path.join(vdir, "vae_config.json")) as f:
        vae_cfg = json.load(f)
    vae = ConditionedBlockVAE(**vae_cfg).to(device)
    vae.load_state_dict(torch.load(os.path.join(vdir, "vae_best.pt"), map_location=device))
    vae.eval()

    fake_args = _argparse.Namespace(
        artifact_dir=artifact_dir, noise_scale=meta["noise_scale"], mode=meta["mode"]
    )
    dataset = T._load_existing_dataset(meta, fake_args)
    return pca, vae, dataset, vdir


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Standalone multiple-choice benchmark evaluation. Hooks into "
                     "either raw HuggingFace checkpoints (--raw, arch_list only — "
                     "no PCA/VAE needed) or a trained PCA+VAE artifact set (default), "
                     "in which case it can test the VAE as an autoencoder "
                     "(--mode reconstruct, default) or as a generator sampling fresh "
                     "block weights straight from its prior (--mode generate)."
    )
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--pca_dir", default=None, help="Defaults to <artifact_dir>/pca")
    p.add_argument("--vae_dir", default=None, help="Defaults to <artifact_dir>/vae")
    p.add_argument("--arch_list", nargs="+", default=None,
                    help="Restrict evaluation to these archs. With --raw: any "
                         "registered arch. Otherwise: must be a subset of the "
                         "archs the loaded PCA/VAE were trained on.")
    p.add_argument("--raw", action="store_true",
                    help="Skip PCA/VAE entirely — evaluate stock HF checkpoints "
                         "directly (the 'hf model path' hook).")
    p.add_argument("--mode", choices=["reconstruct", "generate"], default="reconstruct",
                    help="reconstruct: VAE round-trips each real block (fidelity "
                         "check). generate: sample fresh blocks from the VAE prior "
                         "conditioned on block/family (tests generative quality, "
                         "not just reconstruction). Ignored with --raw.")
    p.add_argument("--skip_simple_eval", action="store_true",
                    help="Skip the cheap block-level cosine-sim/MSE pass "
                         "(evaluate.py) before the expensive MC benchmarks. "
                         "Automatically skipped for --raw or --mode generate "
                         "(no matching real block to diff a generated one against).")
    p.add_argument("--mc_benchmarks", nargs="+", default=["mmlu", "hellaswag", "gpqa"])
    p.add_argument("--mc_n_questions", type=int, default=200)
    p.add_argument("--no_wandb", action="store_true",
                    help="Disable Weights & Biases logging for this run")
    args = p.parse_args()

    hf_cache = os.environ.get("HF_HOME", os.path.join(args.artifact_dir, "hf_cache"))
    res_dir = os.path.join(args.artifact_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    job_type = "eval_mc_raw" if args.raw else f"eval_mc_{args.mode}"
    wb.init_run(
        job_type=job_type,
        config=vars(args),
        tags=[job_type] + (args.arch_list or []),
        enabled=not args.no_wandb,
        artifact_dir=args.artifact_dir,
    )

    if args.raw:
        from eval_baseline import evaluate_baseline_family
        from models.registry import list_archs

        arch_list = args.arch_list or list_archs()
        print(f"{'='*60}\nRaw HF checkpoint MC eval — arch_list={arch_list}\n{'='*60}")
        results = {}
        for arch in arch_list:
            print(f"\n=== {arch} ===")
            results[arch] = evaluate_baseline_family(
                arch, seq_len=512, n_sequences=32,
                mc_benchmarks=tuple(args.mc_benchmarks),
                mc_n_questions=args.mc_n_questions,
                hf_cache=hf_cache, skip_ppl=True, skip_mc=False,
            )
        out_path = os.path.join(res_dir, "mc_eval_raw_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {out_path}")
        wb.finish()
        return

    pca, vae, dataset, vdir = _load_pca_vae_dataset(args.artifact_dir, args.pca_dir, args.vae_dir)

    mc_arch_list = args.arch_list or dataset.arch_list
    missing = set(mc_arch_list) - set(dataset.arch_list)
    if missing:
        raise ValueError(f"--arch_list contains archs the trained VAE wasn't built "
                          f"on: {missing}. Available: {dataset.arch_list}")

    if not args.skip_simple_eval and args.mode == "reconstruct":
        print(f"\n{'='*60}\nStage: block-level reconstruction eval (evaluate.py)\n{'='*60}")
        from evaluate import evaluate_all

        codes_path = os.path.join(vdir, "pca_codes.npy")
        codes_np = np.array(np.memmap(codes_path, dtype=np.float32, mode="r",
                                       shape=(len(dataset), pca.n_components)))
        codes       = torch.from_numpy(codes_np).float()
        block_idxs  = torch.from_numpy(dataset._block_idxs).long()
        family_idxs = torch.from_numpy(dataset._family_idxs).long()
        device = str(next(vae.parameters()).device)

        simple_results = evaluate_all(pca, vae, dataset, codes, block_idxs, family_idxs, device=device)
        simple_path = os.path.join(res_dir, "reconstruction_results.json")
        with open(simple_path, "w") as f:
            json.dump(simple_results, f, indent=2)
        print(f"  global cosine_sim = {simple_results['global']['cosine_sim']:.6f}")
        print(f"  global mse        = {simple_results['global']['mse']:.3e}")
        print(f"  Saved -> {simple_path}")
        wb.log({
            "recon/cosine_sim": simple_results["global"]["cosine_sim"],
            "recon/mse": simple_results["global"]["mse"],
            "recon/kl_divergence": simple_results["global"]["kl_divergence"],
        })

    transform_fn = reconstruct_model_blocks if args.mode == "reconstruct" else generate_model_blocks
    print(f"\n{'='*60}\nStage: MC benchmark eval (mode={args.mode}) — arch_list={mc_arch_list}\n{'='*60}")
    mc_results = evaluate_all_families_mc(
        pca, vae, dataset,
        benchmarks=tuple(args.mc_benchmarks),
        n_questions=args.mc_n_questions,
        artifact_dir=args.artifact_dir,
        arch_list=mc_arch_list,
        block_transform_fn=transform_fn,
    )
    suffix = "" if args.mode == "reconstruct" else "_generated"
    mc_path = os.path.join(res_dir, f"mc_eval_results{suffix}.json")
    with open(mc_path, "w") as f:
        json.dump(mc_results, f, indent=2)
    print(f"\nSaved -> {mc_path}")
    wb.finish()


if __name__ == "__main__":
    main()
