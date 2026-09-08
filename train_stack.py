"""
Whole-stack pipeline: per-family Gram PCA -> codes -> StackVAE.

One PCA sample is one complete decoder stack (DeepWeightFlow framing), so:

  Stage 1  build the per-architecture ensembles (w_0 plus noise-augmented members)
  Stage 2  fit ONE DualGramPCA per architecture at k = N-1 (the rank bound)
  Stage 3  assemble the code matrix at the requested rank k
  Stage 4  train a StackVAE over those codes, conditioned on family only

Stage 2 fits at the maximum rank ONCE. Lower-rank arms are a column prefix of the
same fit -- components are variance-ordered and orthogonal, so a k-prefix IS the
rank-k PCA. That makes the whole k sweep free after one fit.

    python train_stack.py --k 99 --n_samples 100 --noise_scale 1e-2
    python train_stack.py --k 50            # reuses the Stage 2 fits
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from artifact_io import (
    ensemble_fingerprint, pca_fingerprint, provenance_block, read_json,
)
from data.ensemble_dataset import EnsembleDataset
from dual_gram_pca import DualGramPCA
from models.registry import N_FAMILIES
from run_bundle import CodeStats, rebuild_manifest
from vae import BetaScheduler, StackVAE
import wandb_utils as wb


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def spectrum_stats(evals: np.ndarray, k: int) -> dict:
    """
    Flatness of the retained spectrum, measured three ways.

    `ev0_over_evlast` is the ratio this repo reported first, and on its own it
    MISLEADS at k = N-1. Measured on a pure-noise ensemble (N=12): 12.09 at k=N-1 but
    1.006 at k=N/2, for the same data. At the rank bound `ev[k-1]` is the smallest
    numerically-marginal direction left standing after the rank floor, so the ratio
    describes the tail rather than the bulk and reads "steep" for an ensemble that is
    flat by construction. Same trap as `total_variance_captured` being vacuous at
    k = N-1 (misstep 15).

    So two bulk statistics are recorded alongside it, and they are what anything
    downstream should gate on:

    ev0_over_median
        The leading direction's variance as a multiple of the TYPICAL direction's.
        ~1 for isotropic noise, large when a few directions carry the energy.

    effective_rank_ratio
        (sum ev)^2 / (sum ev^2) / k -- the participation ratio, normalised to [0, 1].
        1.0 means every retained direction carries equal variance (perfectly flat);
        small means the variance is concentrated. Unlike any ratio of two individual
        eigenvalues, this cannot be moved by one marginal direction at the tail.
    """
    ev = np.asarray(evals, dtype=np.float64)[:max(int(k), 1)]
    ev = np.clip(ev, 1e-300, None)
    med = float(np.median(ev))
    eff = float((ev.sum() ** 2) / max(float((ev ** 2).sum()), 1e-300))
    return {
        "ev0_over_evlast": float(ev[0] / ev[-1]),
        "ev0_over_median": float(ev[0] / max(med, 1e-300)),
        "effective_rank_ratio": eff / max(len(ev), 1),
        "effective_rank": eff,
    }


def stage_pca(args, ds: EnsembleDataset, pca_root: str) -> Dict[str, DualGramPCA]:
    """Fit (or load) one DualGramPCA per architecture at the maximum rank."""
    out: Dict[str, DualGramPCA] = {}
    k_max = ds.n_samples - 1
    for arch in ds.arch_list:
        d = os.path.join(pca_root, arch)
        meta = os.path.join(d, "gram_pca_meta.json")
        if os.path.exists(meta) and not args.force_pca:
            print(f"{ts()} PCA for {arch} exists — loading …")
            pca = DualGramPCA.load(d)
            if pca.n_samples_ != ds.n_samples:
                raise RuntimeError(
                    f"{arch}: cached PCA was fit on N={pca.n_samples_} but the "
                    f"ensemble now has N={ds.n_samples}. Re-fit with --force_pca.")
            # D changes whenever the stack layout changes -- most sharply when the
            # extra segment is switched on or off. Without this check a stale
            # decoder-only basis is silently reused against a wider ensemble and
            # every code means something different.
            if pca.n_params_ != ds.stacks[arch].n_params:
                raise RuntimeError(
                    f"{arch}: cached PCA was fit on D={pca.n_params_:,} but the "
                    f"ensemble now has D={ds.stacks[arch].n_params:,} "
                    f"(include_extra={ds.include_extra}). Re-fit with --force_pca.")
        else:
            print(f"\n{ts()} Fitting PCA for {arch} at k={k_max} …")
            pca = DualGramPCA(n_components=k_max).fit(ds, arch)
            pca.save(d)
        out[arch] = pca
    return out


def stage_codes(ds: EnsembleDataset, pcas: Dict[str, DualGramPCA], k: int,
                codes_dir: str, provenance: dict):
    """
    Assemble the code matrix at rank k and SEAL it as an artifact.

    Returns (codes (M,k) tensor, family_idxs (M,) tensor, arch_of_row, CodeStats).

    Writing codes_k<k>/ is what lets train_flow.py train on 40 KB without ever
    constructing an EnsembleDataset or touching DualGramPCA. That is the property
    that makes the flow stack re-runnable on a different ensemble source (Pythia
    checkpoint revisions): swap the source, re-run stages 1-3, and the flow trainer
    is unchanged and unaware.
    """
    avail = min(p.n_components for p in pcas.values())
    if k > avail:
        raise ValueError(f"--k {k} exceeds the smallest fitted rank ({avail}). "
                         f"Re-fit with a larger ensemble, or lower --k.")

    blocks, fidxs, arch_of_row = [], [], []
    seen: Dict[int, str] = {}
    arch_to_family: Dict[str, int] = {}
    for arch in ds.arch_list:
        c = torch.from_numpy(pcas[arch].codes(k).copy()).float()
        fi = int(ds.stacks[arch].family_idx)
        # Two archs sharing a family_idx would silently overwrite each other's
        # statistics. Unreachable with today's registry, but the failure is silent,
        # so it is checked rather than assumed.
        if fi in seen:
            raise ValueError(
                f"{arch} and {seen[fi]} both claim family_idx={fi}. Per-family code "
                f"statistics are keyed by family_idx, so one would clobber the "
                f"other. Renumber ARCH_CONFIGS contiguously.")
        seen[fi] = arch
        arch_to_family[arch] = fi
        blocks.append(c)
        fidxs.append(torch.full((c.shape[0],), fi, dtype=torch.long))
        arch_of_row.extend([arch] * c.shape[0])

    codes = torch.cat(blocks)
    fidx = torch.cat(fidxs)

    # CodeStats.from_codes is the single implementation of the reduction: per-dim
    # mean, and one scalar scale per family (RMS over dimensions of the per-dim
    # stds). Each family has its OWN basis, so pooling across families would compare
    # coefficients on unrelated directions.
    cs = CodeStats.from_codes(codes.numpy(), fidx.numpy(), arch_to_family,
                              N_FAMILIES, k=k)
    for arch, fi in arch_to_family.items():
        print(f"  {arch:16s} codes ({int((fidx == fi).sum())}, {k})  "
              f"family_idx={fi}  scale={float(cs.stds[fi, 0]):.4g}")
    cs.save(codes_dir, provenance=provenance)
    return codes, fidx, arch_of_row, cs


def train_vae(args, codes: torch.Tensor, fidxs: torch.Tensor,
              code_stats: CodeStats, vae_dir: str,
              provenance: dict) -> StackVAE:
    os.makedirs(vae_dir, exist_ok=True)
    M, k = codes.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{ts()} Stage 4: StackVAE on {M} samples, code_dim={k}, {device}")

    model = StackVAE(code_dim=k, latent_dim=args.latent_dim,
                     hidden_dim=args.hidden_dim, cond_dim=args.cond_dim,
                     n_families=N_FAMILIES,
                     cond_dropout_p=args.cond_dropout).to(device)
    model.set_code_norm(code_stats.torch_means(device),
                        code_stats.torch_stds(device))

    # vae_config.json is still written, for anything that reads the old layout.
    # vae_meta.json + vae_weights.pt (sealed at the end of this function) are the
    # authoritative artifact.
    with open(os.path.join(vae_dir, "vae_config.json"), "w") as f:
        json.dump(model.config(), f, indent=2)

    loader = DataLoader(TensorDataset(codes, fidxs),
                        batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                        eta_min=1e-5)
    beta_sched = BetaScheduler(beta_max=args.beta, warmup_epochs=args.warmup_epochs)

    # Per-dimension code std, for the augmentation noise.
    per_dim = codes.std(dim=0).clamp(min=1e-8).to(device)

    best, patience, history = float("inf"), 0, []
    vae_path = os.path.join(vae_dir, "vae_best.pt")

    for epoch in range(1, args.epochs + 1):
        beta = beta_sched.get(epoch)
        model.train()
        tot = rec = klv = 0.0
        kl_dims_sum = torch.zeros(args.latent_dim, device=device)
        for cb, fb in loader:
            cb, fb = cb.to(device), fb.to(device)
            cin = cb
            if args.code_noise_std > 0.0:
                cin = cb + torch.randn_like(cb) * (args.code_noise_std * per_dim)
            opt.zero_grad()
            # Reconstruct toward the CLEAN codes -- the noise is input-side
            # augmentation, not a target perturbation.
            recon, mu, logvar = model(cin, fb)
            loss, rl, kl = model.elbo_loss(recon, cb, mu, logvar, fb,
                                           beta=beta, free_bits=args.free_bits)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); rec += rl.item(); klv += kl.item()
            with torch.no_grad():
                kl_dims_sum += model.kl_per_dim(mu, logvar)
        nb = len(loader)
        tot, rec, klv = tot / nb, rec / nb, klv / nb
        kl_dims = kl_dims_sum / nb
        kl_nats = float(kl_dims.sum())
        active = int((kl_dims > 0.01).sum())
        sched.step()

        # Checkpoint selection only once beta has settled: during warmup the
        # objective changes every epoch, so comparing losses across epochs compares
        # different functions.
        in_warmup = epoch <= args.warmup_epochs
        if tot < best or in_warmup:
            best, patience = tot, 0
            torch.save(model.state_dict(), vae_path)
        else:
            patience += 1

        history.append({"epoch": epoch, "beta": round(beta, 4),
                        "loss": round(tot, 6), "recon": round(rec, 6),
                        "kl": round(klv, 8), "kl_total_nats": round(kl_nats, 6),
                        "active_units": active})
        # Namespaced under vae/ so the flow trainer's loss curves can live in the
        # same project without either one shadowing "train/loss". The two gating
        # metrics from RESEARCH_NOTES -- total KL nats and active units -- keep
        # their own kl/ namespace so existing dashboards still resolve.
        wb.log({"vae/beta": beta, "vae/lr": opt.param_groups[0]["lr"],
                "vae/train/loss": tot, "vae/train/recon": rec,
                "vae/train/kl_per_dim": klv,
                "vae/epoch": epoch,
                "kl/total_nats_per_sample": kl_nats,
                "kl/active_units": active,
                "kl/max_dim_nats": float(kl_dims.max()),
                "kl/min_dim_nats": float(kl_dims.min())}, step=epoch)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:5d}/{args.epochs}  loss={tot:.6f}  "
                  f"recon={rec:.6f}  KL={kl_nats:.3f}nats  "
                  f"active={active}/{args.latent_dim}  beta={beta:.3f}"
                  f"{'  [warmup]' if in_warmup else ''}")
        if patience >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best={best:.6f})")
            break

    kl_floor = args.free_bits * args.latent_dim
    print(f"\n  Final KL: {kl_nats:.4f} nats/sample, {active}/{args.latent_dim} "
          f"active dims (free-bits floor {kl_floor:.3f})")
    if kl_nats <= max(0.5, 1.10 * kl_floor):
        print("  WARNING: KL is at the free-bits floor — the latent is carrying no "
              "information beyond what free bits forces. With family-only "
              "conditioning this should NOT happen (a family label cannot identify "
              "a sample), so treat it as a real signal that something is wrong "
              "rather than as expected collapse.")

    # Terminal values into the run summary so the W&B runs table shows the two
    # gating numbers as columns, without having to open each run.
    wb.summary({"vae/best_loss": best,
                "vae/final_kl_total_nats": kl_nats,
                "vae/final_active_units": active,
                "vae/latent_dim": args.latent_dim,
                "vae/code_dim": k,
                "vae/epochs_run": epoch,
                "vae/kl_at_free_bits_floor":
                    bool(kl_nats <= max(0.5, 1.10 * kl_floor))})

    with open(os.path.join(vae_dir, "train_metrics.json"), "w") as f:
        json.dump({"best_loss": best, "final_kl_total_nats": kl_nats,
                   "final_active_units": active,
                   "final_kl_per_dim": [round(float(x), 8) for x in kl_dims.cpu()],
                   "free_bits": args.free_bits,
                   "cond_dropout_p": args.cond_dropout,
                   "history": history}, f, indent=2)

    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()

    # Seal ONCE, after the best weights are back in the model. vae_best.pt is
    # rewritten on every improving epoch and stays raw scratch; rewriting the
    # metadata JSON alongside it up to `epochs` times would buy nothing.
    model.save(vae_dir, provenance=provenance,
               code_stats_fingerprint=code_stats.fingerprint(),
               code_stats_dir=f"codes_k{k}",
               train={"best_loss": best, "final_kl_total_nats": kl_nats,
                      "final_active_units": active, "epochs_run": epoch,
                      "free_bits": args.free_bits,
                      "cond_dropout_p": args.cond_dropout})
    return model


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch_list", nargs="+",
                   default=["gpt2_medium", "smollm2_360m", "pythia_410m"])
    p.add_argument("--mode", choices=["tiny", "full"], default="full")
    p.add_argument("--run_name", default="perfam")
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")

    # Ensemble
    p.add_argument("--n_samples", type=int, default=100,
                   help="N, ensemble size per architecture (sample 0 = real model)")
    p.add_argument("--noise_scale", type=float, default=1e-2,
                   help="Augmentation std as a fraction of the stack's weight std. "
                        "Calibrate with calibrate_noise.py — too small and the Gram "
                        "matrix is roundoff, too large and the members are broken.")
    p.add_argument("--include_extra", action="store_true", default=True,
                   help="Include embeddings, final norm and untied LM head in the "
                        "stack. On by default: DeepWeightFlow flattens whole "
                        "networks, and gpt2_medium's embedding alone is ~17%% of "
                        "its parameters.")
    p.add_argument("--no_include_extra", dest="include_extra",
                   action="store_false",
                   help="Decoder blocks only — the layout_version 1 behaviour. "
                        "Needs a fresh --run_name; the ensemble cache gate refuses "
                        "to mix the two.")
    p.add_argument("--exclude_1d", action="store_true", default=True)
    p.add_argument("--no_exclude_1d", dest="exclude_1d", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk_budget_mb", type=int, default=512)

    # Rank
    p.add_argument("--k", type=int, default=None,
                   help="Code dimension. Default N-1 (the rank bound).")

    # VAE
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--warmup_epochs", type=int, default=100)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--free_bits", type=float, default=0.05)
    p.add_argument("--cond_dropout", type=float, default=0.15)
    p.add_argument("--code_noise_std", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--force_pca", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    pca_root = os.path.join(run_root, "pca")
    os.makedirs(run_root, exist_ok=True)

    k = args.k if args.k is not None else args.n_samples - 1
    vae_dir = os.path.join(run_root, f"vae_k{k}")

    # name_suffix matters: slurm_stack_run.sh calls this script once per rank in a
    # single job, and each call is its own process, so without it both runs land
    # under the identical name train_stack_<jobid>.
    wb.init_run(job_type="train_stack", config=vars(args),
                tags=[args.mode, f"k{k}", args.run_name] + args.arch_list,
                enabled=not args.no_wandb, artifact_dir=args.artifact_dir,
                name_suffix=f"{args.run_name}_k{k}")

    print("=" * 68)
    print("Whole-stack pipeline (per-family Gram PCA + StackVAE)")
    print(f"  run       : {args.run_name}   -> {run_root}")
    print(f"  archs     : {args.arch_list}")
    print(f"  N         : {args.n_samples}   noise_scale={args.noise_scale}")
    print(f"  k         : {k}  (rank bound is N-1 = {args.n_samples - 1})")
    print(f"  extra seg : include_extra={args.include_extra} "
          f"(embeddings / final norm / untied LM head)")
    print("=" * 68)

    print(f"\n{ts()} Stage 1: ensembles")
    ds = EnsembleDataset(arch_list=args.arch_list, n_samples=args.n_samples,
                         noise_scale=args.noise_scale, exclude_1d=args.exclude_1d,
                         include_extra=args.include_extra,
                         mode=args.mode, artifact_dir=run_root, seed=args.seed,
                         chunk_budget_bytes=args.chunk_budget_mb * 1024 * 1024,
                         force_extract=args.force_extract)
    print(ds.summary())

    print(f"\n{ts()} Stage 2: per-family Gram PCA")
    pcas = stage_pca(args, ds, pca_root)

    # Fingerprints bind every downstream artifact to the ensemble and bases it came
    # from, so a stale codes/VAE/flow directory is DETECTED rather than reused.
    ens_meta = read_json(os.path.join(run_root, "ensemble", "ensemble_meta.json"))
    ens_fp = ensemble_fingerprint(ens_meta)
    pca_fps = {
        a: pca_fingerprint(
            read_json(os.path.join(pca_root, a, "gram_pca_meta.json")) or {})
        for a in args.arch_list
    }
    print(f"  ensemble fingerprint : {ens_fp}")

    print(f"\n{ts()} Stage 3: codes at k={k}")
    codes_dir = os.path.join(run_root, f"codes_k{k}")
    codes, fidxs, arch_of_row, code_stats = stage_codes(
        ds, pcas, k, codes_dir,
        provenance_block(args.run_name, k, ens_fp, pca_fps))
    print(f"  code matrix: {tuple(codes.shape)} across {len(set(arch_of_row))} families")

    train_vae(
        args, codes, fidxs, code_stats, vae_dir,
        provenance_block(args.run_name, k, ens_fp, pca_fps,
                         code_stats_fp=code_stats.fingerprint()))

    summary = {
        "run_name": args.run_name, "k": k, "n_samples": args.n_samples,
        "noise_scale": args.noise_scale, "arch_list": args.arch_list,
        "exclude_1d": args.exclude_1d,
        "include_extra": args.include_extra,
        "per_arch": {a: {"n_params": ds.stacks[a].n_params,
                          "n_layers": ds.stacks[a].n_layers,
                          "block_size": ds.stacks[a].block_size,
                          "extra_size": ds.stacks[a].extra_size,
                          "fitted_k": pcas[a].n_components,
                          "variance_captured_at_k": float(np.sum(
                              pcas[a].explained_variance_ratio_[:k])),
                          **{f"spectrum_{key}": val for key, val in
                             spectrum_stats(pcas[a].gram_evals_,
                                            min(k, pcas[a].n_components)).items()}}
                      for a in ds.arch_list},
    }
    summary["ensemble_fingerprint"] = ens_fp
    summary["pca_fingerprints"] = pca_fps
    summary["code_stats_fingerprint"] = code_stats.fingerprint()
    with open(os.path.join(run_root, f"pipeline_summary_k{k}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Derived index over the per-directory metas. Rebuilt by scanning rather than
    # incrementally maintained, so the two train_stack invocations in one Slurm job
    # cannot race each other into a lost section.
    rebuild_manifest(run_root)

    print(f"\n{ts()} Done. Artifacts under {run_root}")
    for a, d in summary["per_arch"].items():
        print(f"  {a:16s} D={d['n_params']:>13,}  fitted_k={d['fitted_k']:>4d}  "
              f"var@k={d['variance_captured_at_k']:.4%}  "
              f"ev0/med={d['spectrum_ev0_over_median']:.4g}  "
              f"effrank={d['spectrum_effective_rank_ratio']:.3f}")

    # Spectrum flatness is the honest caveat, made visible at the end of every run
    # rather than left in a JSON nobody opens. RESEARCH_NOTES Experiment 3: a flow
    # over codes whose aggregate distribution is already Gaussian learns the prior
    # and adds nothing. Reported on the BULK statistic -- ev0/ev[k-1] reads ~12 on a
    # pure-noise ensemble at k=N-1 and would hide exactly this.
    ratios = {a: d["spectrum_ev0_over_median"]
              for a, d in summary["per_arch"].items()}
    effs = {a: d["spectrum_effective_rank_ratio"]
            for a, d in summary["per_arch"].items()}
    if ratios and min(ratios.values()) < 2.0:
        print(f"\n  NOTE the retained spectrum is FLAT "
              f"(min ev0/median = {min(ratios.values()):.3g}, "
              f"max effective-rank ratio = {max(effs.values()):.3f} of 1.0). For a "
              f"manufactured noise ensemble that is the expected result, not a "
              f"problem — but it means the per-family code distribution is "
              f"near-Gaussian by construction, so any flow fit to it may learn only "
              f"the prior. Compare flow_codes against gauss_codes, not against "
              f"pca_only.")

    wb.summary({"pipeline/ensemble_fingerprint": ens_fp,
                "pipeline/code_stats_fingerprint": code_stats.fingerprint(),
                "pipeline/min_spectrum_ev0_over_median":
                    min(ratios.values()) if ratios else None,
                "pipeline/max_effective_rank_ratio":
                    max(effs.values()) if effs else None,
                "pipeline/k": k,
                "pipeline/include_extra": args.include_extra})
    wb.finish()


if __name__ == "__main__":
    main()
