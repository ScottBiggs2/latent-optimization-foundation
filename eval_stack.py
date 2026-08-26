"""
Whole-stack evaluation: PCA-only and PCA+VAE, at one or more ranks.

For each architecture and each rank k:

  PCA-only   codes[0][:k]  ->  inverse_transform  ->  write back  ->  PPL
  PCA+VAE    codes[0][:k]  ->  VAE round trip     ->  inverse     ->  PPL
  generate   z ~ N(0,I)    ->  VAE decode (CFG)   ->  inverse     ->  PPL

Sample 0 of every ensemble is the real pretrained stack, so PCA-only at k = N-1 is
the rank-bound identity and must come back at cosine 1.0 / dPPL ~ 0. Anything else
there is a bug, not a finding. The informative arm is k < N-1.

Read the two arms together: PCA-only isolates what the basis can represent, and the
difference between the arms is what the VAE costs. Gate on dPPL, not cosine --
a measured 0.99939 cosine corresponded to +72,468% PPL on this pipeline, so cosine
below about 0.9999 tells you nothing useful.

    python eval_stack.py --run_name perfam --k 99 50
    python eval_stack.py --run_name perfam --k 50 --arms pca_only vae generate
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from data.ensemble_dataset import EnsembleDataset
from dual_gram_pca import DualGramPCA
from models.registry import get_arch_config, get_layers, load_model, build_tiny_model
from models.weight_extractor import extract_block_flat, reconstruct_block
from vae import StackVAE
import wandb_utils as wb


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------

def write_stack_to_model(stack: np.ndarray, model, arch: str,
                         ds: EnsembleDataset) -> None:
    """
    Slice a flat (D,) stack vector back into per-block parameter tensors.

    The stack is the concatenation of blocks in depth order and every block has the
    same schema (EnsembleDataset enforces both at extraction), so the offsets are
    just block_idx * block_size. reconstruct_block writes only the parameters named
    in the schema, so with exclude_1d the norm gains and biases are never touched and
    keep their pretrained values.
    """
    st = ds.stacks[arch]
    layers = get_layers(model, arch)
    if len(layers) != st.n_layers:
        raise RuntimeError(f"{arch}: model has {len(layers)} layers but the ensemble "
                           f"recorded {st.n_layers}")
    for i, layer in enumerate(layers):
        lo, hi = st.block_slice(i)
        reconstruct_block(np.ascontiguousarray(stack[lo:hi]), layer, st.schema)


def stack_metrics(recon: np.ndarray, w0: np.ndarray) -> dict:
    r = np.asarray(recon, dtype=np.float64)
    w = np.asarray(w0, dtype=np.float64)
    denom = np.linalg.norm(r) * np.linalg.norm(w) + 1e-30
    return {"cosine_sim": float(np.dot(r, w) / denom),
            "mse": float(np.mean((r - w) ** 2)),
            "rel_l2": float(np.linalg.norm(r - w) / (np.linalg.norm(w) + 1e-30))}


# ---------------------------------------------------------------------------
# Per-arch evaluation
# ---------------------------------------------------------------------------

def evaluate_arch(
    arch: str,
    ds: EnsembleDataset,
    pca: DualGramPCA,
    vae: Optional[StackVAE],
    k: int,
    arms: List[str],
    seq_len: int,
    n_sequences: int,
    hf_cache: Optional[str],
    mode: str,
    guidance_scale: float,
    seed: int,
    sample_idx: int = 0,
) -> dict:
    from eval_lm import compute_perplexity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_arch_config(arch)

    model = (build_tiny_model(arch) if mode == "tiny"
             else load_model(arch, cache_dir=hf_cache)).to(device)
    model.eval()

    if mode == "tiny":
        from data.val_loader import get_synthetic_loader
        vocab = cfg["tiny_config"].get("vocab_size",
                                       cfg["tiny_config"].get("n_positions", 1000))
        loader = get_synthetic_loader(vocab_size=vocab, seq_len=seq_len,
                                     n_sequences=n_sequences)
    else:
        from data.val_loader import get_wikitext2_loader
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["default_model_id"],
                                            cache_dir=hf_cache,
                                            trust_remote_code=True)
        loader = get_wikitext2_loader(tok, seq_len=seq_len,
                                     n_sequences=n_sequences, cache_dir=hf_cache)

    st = ds.stacks[arch]
    fidx = torch.tensor([st.family_idx], dtype=torch.long, device=device)

    # The target is ensemble member `sample_idx`. Member 0 is the real pretrained
    # model; any other member is w_0 plus augmentation noise, so it has to be
    # materialised and written into the model FIRST, or "original PPL" would be the
    # real model's while the reconstruction target is a different point.
    target = (np.asarray(ds.w0(arch)) if sample_idx == 0
              else ds.materialize_sample(arch, sample_idx, device=device))
    if sample_idx != 0:
        write_stack_to_model(target, model, arch, ds)

    # Pristine copy AFTER installing the target, so restore() returns to the target.
    pristine = [extract_block_flat(l, exclude_1d=ds.exclude_1d)[0]
                for l in get_layers(model, arch)]

    print(f"  [{arch}] measuring PPL of ensemble member {sample_idx} …")
    base = compute_perplexity(model, loader, device)
    print(f"  [{arch}] member {sample_idx} PPL = {base['perplexity']:.4f}")

    def restore():
        for i, layer in enumerate(get_layers(model, arch)):
            reconstruct_block(pristine[i], layer, st.schema)

    out = {"arch": arch, "k": k, "n_layers": st.n_layers,
           "n_params": st.n_params, "sample_idx": sample_idx,
           "original_ppl": base["perplexity"],
           "ce_original": base["ce_loss"], "arms": {}}

    codes_k = pca.codes(k)[sample_idx]

    for arm in arms:
        if arm == "pca_only":
            recon = pca.inverse_transform(codes_k, ds, arch, k=k)
        elif arm == "vae":
            if vae is None:
                print(f"  [{arch}] skipping 'vae' arm — no checkpoint loaded")
                continue
            with torch.no_grad():
                ct = torch.from_numpy(codes_k.reshape(1, -1)).float().to(device)
                # sample=False: reconstruction fidelity is a deterministic
                # measurement, so use the posterior mean.
                rc, _, _ = vae(ct, fidx, sample=False)
            recon = pca.inverse_transform(rc.cpu().numpy().reshape(-1), ds, arch, k=k)
        elif arm == "generate":
            if vae is None:
                print(f"  [{arch}] skipping 'generate' arm — no checkpoint loaded")
                continue
            g = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                z = torch.randn(1, vae.latent_dim, device=device, generator=g)
                gc_ = vae.decode_cfg(z, fidx, guidance_scale=guidance_scale)
            recon = pca.inverse_transform(gc_.cpu().numpy().reshape(-1), ds, arch, k=k)
        else:
            raise ValueError(f"unknown arm {arm!r}")

        m = stack_metrics(recon, target)
        write_stack_to_model(recon, model, arch, ds)
        res = compute_perplexity(model, loader, device)
        restore()

        delta = res["perplexity"] - base["perplexity"]
        pct = 100.0 * delta / max(base["perplexity"], 1e-9)
        out["arms"][arm] = {**m, "ppl": res["perplexity"], "ppl_delta": delta,
                            "ppl_delta_pct": pct, "ce": res["ce_loss"]}
        print(f"  [{arch}] {arm:9s} k={k:<4d} cos={m['cosine_sim']:.6f}  "
              f"relL2={m['rel_l2']:.4g}  PPL={res['perplexity']:12.4f}  "
              f"Δ={pct:+.3f}%")
        wb.log({f"stack/{arch}/k{k}/{arm}/cosine_sim": m["cosine_sim"],
                f"stack/{arch}/k{k}/{arm}/ppl_delta_pct": pct,
                f"stack/{arch}/k{k}/{arm}/ppl": res["perplexity"]})

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--run_name", default="perfam")
    p.add_argument("--arch_list", nargs="+", default=None)
    p.add_argument("--k", nargs="+", type=int, default=None,
                   help="Ranks to evaluate. Default: the fitted maximum and half it.")
    p.add_argument("--arms", nargs="+", default=["pca_only", "vae"],
                   choices=["pca_only", "vae", "generate"])
    p.add_argument("--mode", choices=["tiny", "full"], default="full")
    p.add_argument("--eval_seq_len", type=int, default=1024)
    p.add_argument("--eval_n_sequences", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Which ensemble member to reconstruct. 0 is the real "
                        "pretrained model, but it sits ~sqrt(N) nearer the ensemble "
                        "mean than a typical member, so a rank sweep on sample 0 "
                        "understates truncation loss. Use a nonzero index for a "
                        "meaningful k sweep.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    ens_meta = os.path.join(run_root, "ensemble", "ensemble_meta.json")
    if not os.path.exists(ens_meta):
        raise SystemExit(f"No ensemble at {ens_meta}. Run train_stack.py first.")
    with open(ens_meta) as f:
        em = json.load(f)

    arch_list = args.arch_list or em["arch_list"]
    hf_cache = os.environ.get("HF_HOME", os.path.join(args.artifact_dir, "hf_cache"))

    # Rebuild the ensemble with the RECORDED parameters. The cache gate in
    # EnsembleDataset refuses any mismatch, so inverse_transform is guaranteed to
    # stream the same ensemble the PCA was fit on.
    ds = EnsembleDataset(arch_list=arch_list, n_samples=em["n_samples"],
                        noise_scale=em["noise_scale"], exclude_1d=em["exclude_1d"],
                        mode=em["mode"], artifact_dir=run_root, seed=em["seed"],
                        chunk_budget_bytes=em["chunk_budget_bytes"])

    pcas = {a: DualGramPCA.load(os.path.join(run_root, "pca", a)) for a in arch_list}
    fitted = min(p.n_components for p in pcas.values())
    ks = args.k or sorted({fitted, max(2, fitted // 2)}, reverse=True)
    for k in ks:
        if k > fitted:
            raise SystemExit(f"--k {k} exceeds the fitted rank ({fitted}).")

    wb.init_run(job_type="eval_stack",
                config={**vars(args), "ks": ks, "fitted_k": fitted},
                tags=["stack"] + arch_list, enabled=not args.no_wandb,
                artifact_dir=args.artifact_dir)

    print("=" * 74)
    print("Whole-stack evaluation")
    print(f"  run    : {args.run_name}")
    print(f"  archs  : {arch_list}")
    print(f"  N      : {em['n_samples']}  (rank bound {em['n_samples'] - 1}, "
          f"fitted {fitted})")
    print(f"  ranks  : {ks}")
    print(f"  arms   : {args.arms}")
    print("=" * 74)

    results: Dict[str, dict] = {}
    for k in ks:
        vae = None
        if "vae" in args.arms or "generate" in args.arms:
            vdir = os.path.join(run_root, f"vae_k{k}")
            cfgp = os.path.join(vdir, "vae_config.json")
            ckpt = os.path.join(vdir, "vae_best.pt")
            if os.path.exists(cfgp) and os.path.exists(ckpt):
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                with open(cfgp) as f:
                    vcfg = json.load(f)
                if vcfg["code_dim"] != k:
                    raise SystemExit(f"{vdir}: VAE code_dim={vcfg['code_dim']} but "
                                     f"evaluating k={k}. Train a VAE for this rank.")
                vae = StackVAE(**vcfg).to(dev)
                vae.load_state_dict(torch.load(ckpt, map_location=dev))
                vae.eval()
                print(f"\nloaded StackVAE for k={k} from {vdir}")
            else:
                print(f"\nNOTE no StackVAE at {vdir} — VAE arms will be skipped "
                      f"for k={k}. Run train_stack.py --k {k} first.")

        for arch in arch_list:
            print(f"\n=== {arch}  (k={k}) ===")
            results[f"{arch}@k{k}"] = evaluate_arch(
                arch, ds, pcas[arch], vae, k, args.arms,
                args.eval_seq_len, args.eval_n_sequences, hf_cache,
                em["mode"], args.guidance_scale, args.seed,
                sample_idx=args.sample_idx)
        del vae
        gc.collect()

    res_dir = os.path.join(run_root, "results")
    os.makedirs(res_dir, exist_ok=True)
    suffix = "" if args.sample_idx == 0 else f"_s{args.sample_idx}"
    out_path = os.path.join(res_dir, f"stack_eval_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 74)
    print(f"{'arch@k':26s} {'arm':10s} {'cosine':>10s} {'PPL orig':>10s} "
          f"{'PPL new':>13s} {'Δ%':>12s}")
    for key, r in results.items():
        for arm, a in r["arms"].items():
            print(f"{key:26s} {arm:10s} {a['cosine_sim']:10.6f} "
                  f"{r['original_ppl']:10.3f} {a['ppl']:13.3f} "
                  f"{a['ppl_delta_pct']:+12.3f}")
    print(f"\nSaved → {out_path}")
    wb.finish()


if __name__ == "__main__":
    main()
