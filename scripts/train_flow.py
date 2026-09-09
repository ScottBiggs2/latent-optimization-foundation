"""
Train a conditional rectified flow over stack PCA codes, or over StackVAE latents.

Two interchangeable spaces, ONE code path -- the only thing that differs is which
FlowSpace gets built (flow.build_space):

    --space codes    target = per-family-normalized PCA codes   (k dims)
    --space latent   target = StackVAE posterior means mu(codes) (latent_dim dims)

This script reads ONLY runs/<run>/codes_k<k>/ -- codes, family labels, per-family
statistics, about 40 KB -- plus runs/<run>/vae_k<k>/ for the latent space. It never
constructs an EnsembleDataset and never touches DualGramPCA, because decoding codes
back to weights is eval_stack's job, not training's. That separation is deliberate:
swapping the ensemble source (manufactured noise -> Pythia checkpoint revisions)
changes stages 1 to 3 of train_stack.py and leaves this file untouched and unaware.

`--k` is a SCALAR here, matching train_stack.py, not variadic like eval_stack.py. In
code space a flow's input width IS k, so one flow cannot span ranks; passing
`--k 99 50` would silently train only at 50.

The honest caveat
-----------------
On the current manufactured ensemble the per-family code distribution is
near-Gaussian by construction, so a flow fit to it may learn nothing beyond the
prior. RESEARCH_NOTES Experiment 3 says not to start this until the spectrum is
steep. `--require_spectrum_ratio` turns that into a machine-checkable gate; it is OFF
by default so the machinery is smoke-testable now, and should be set on a real
ensemble so a flat-spectrum run refuses instead of burning GPU-hours fitting a
Gaussian. Either way, the measured spectrum is sealed into the checkpoint and
`eval_stack`'s `gauss_codes` arm is the null model to compare against.

    python scripts/train_flow.py --run_name perfam --k 99 --space codes
    python scripts/train_flow.py --run_name perfam --k 99 --space latent
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from llmzoo.artifacts.io import (
    ensemble_fingerprint, pca_fingerprint, provenance_block, read_json,
)
from llmzoo.gen.flow import FlowVelocityNet, RectifiedFlow, build_space, save_flow
from llmzoo.models.registry import N_FAMILIES
from llmzoo.artifacts.bundle import (
    load_pi_matrix, load_run, update_manifest_section,
)
import llmzoo.wandb_utils as wb


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


def spectrum_from_summary(run_root: str, k: int) -> Dict[str, dict]:
    """
    Read the per-family spectrum out of pipeline_summary_k<k>.json.

    Copied into every flow checkpoint so the flatness caveat travels WITH the model
    rather than living in a sibling file nobody opens. report_stack renders a warning
    banner whenever it sees a flat one.
    """
    summary = read_json(os.path.join(run_root, f"pipeline_summary_k{k}.json")) or {}
    return {
        arch: {
            # ev0_over_median is the one anything should gate on. ev0_over_evlast is
            # carried for continuity but MISLEADS at k = N-1, where ev[k-1] is the
            # smallest direction surviving the rank floor: measured 12.09 there vs
            # 1.006 at k=N/2 on the same pure-noise ensemble.
            "ev0_over_median": d.get("spectrum_ev0_over_median"),
            "effective_rank_ratio": d.get("spectrum_effective_rank_ratio"),
            "ev0_over_evlast": d.get("spectrum_ev0_over_evlast"),
            "variance_captured_at_k": d.get("variance_captured_at_k"),
        }
        for arch, d in (summary.get("per_arch") or {}).items()
    }


def train(args) -> None:
    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    ens_meta = read_json(os.path.join(run_root, "ensemble", "ensemble_meta.json"))
    if ens_meta is None:
        raise SystemExit(f"No ensemble under {run_root}. Run train_stack.py first.")

    k = args.k if args.k is not None else ens_meta["n_samples"] - 1
    # The suffix keeps a pi flow beside its family-conditioned twin instead of
    # overwriting it. That pair IS section 6.2's control: identical codes,
    # identical net, identical epochs, conditioning present or absent -- and on a
    # one-architecture zoo the family flow is genuinely unconditional.
    suffix = "" if args.cond_mode == "family" else f"_{args.cond_mode}"
    out_dir = os.path.join(run_root, f"flow_k{k}_{args.space}{suffix}")
    if os.path.exists(os.path.join(out_dir, "flow_meta.json")) and not args.force:
        raise SystemExit(
            f"{out_dir} already holds a sealed flow. Pass --force to overwrite.")

    # want excludes 'dataset' and 'pca' on purpose: this trainer needs neither, and
    # constructing them would memmap a ~1.2 GB mean per family for nothing.
    want = ("codes", "vae") if args.space == "latent" else ("codes",)
    bundle = load_run(run_root, k, want=want,
                      allow_legacy_vae=args.allow_legacy_vae)
    cs = bundle.code_stats
    if cs is None:
        raise SystemExit(
            f"No codes_k{k}/ under {run_root}. Run train_stack.py --k {k} first.")
    if cs.codes is None or cs.family_idxs is None:
        raise SystemExit(
            f"codes_k{k}/ has statistics but no codes.npy / family_idxs.npy. "
            f"Re-run train_stack.py --k {k}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codes = torch.from_numpy(cs.codes).float().to(device)
    fidx = torch.from_numpy(cs.family_idxs).long().to(device)

    # --- the conditioning variable ----------------------------------------
    # In a single-arch zoo every member shares one family_idx, so cond_mode="family"
    # is literally unconditional. pi is the mixture the member was actually trained
    # on, joined row -> member -> zoo_meta (bundle.load_pi_matrix).
    pi_all = pi_meta = None
    if args.cond_mode == "pi":
        pi_meta = load_pi_matrix(ens_meta)
        pi_np = pi_meta["pi"]
        if pi_np.shape[0] != codes.shape[0]:
            raise SystemExit(
                f"pi has {pi_np.shape[0]} rows but codes has {codes.shape[0]}. "
                f"The row->member join is broken; refusing to train on a scrambled "
                f"pairing, which would look like 'conditioning does not generalise'.")
        pi_all = torch.from_numpy(pi_np).float().to(device)
        print(f"  pi         : {tuple(pi_np.shape)} over {pi_meta['domains']}\n"
              f"               fingerprint {pi_meta['fingerprint']}")

    spectrum = spectrum_from_summary(run_root, k)
    ratios = {a: d["ev0_over_median"] for a, d in spectrum.items()
              if d.get("ev0_over_median") is not None}
    min_ratio = min(ratios.values()) if ratios else None
    effs = {a: d["effective_rank_ratio"] for a, d in spectrum.items()
            if d.get("effective_rank_ratio") is not None}

    if args.require_spectrum_ratio is not None and min_ratio is not None \
            and min_ratio < args.require_spectrum_ratio:
        raise SystemExit(
            f"Refusing to train: the flattest retained spectrum is "
            f"ev0/median = {min_ratio:.4g}, below --require_spectrum_ratio "
            f"{args.require_spectrum_ratio}. A flat spectrum means the codes are "
            f"close to mutually orthogonal and their aggregate distribution is "
            f"near-Gaussian, so a flow over them learns the prior and adds nothing "
            f"(RESEARCH_NOTES Experiment 3). Lower the threshold to override, or "
            f"fit the PCA on an ensemble of genuinely different complete models.")

    # --- the space ---------------------------------------------------------
    space = build_space(args.space, code_stats=cs, vae=bundle.vae,
                        codes=codes, family_idxs=fidx)
    with torch.no_grad():
        x1_all = space.to_flow(codes, fidx)

    net = FlowVelocityNet(
        dim=space.dim, n_families=N_FAMILIES, cond_dim=args.cond_dim,
        hidden_dims=tuple(args.hidden_dims), time_embed_dim=args.time_embed_dim,
        cond_dropout_p=args.cond_dropout,
        cond_mode=args.cond_mode,
        pi_dim=(pi_all.shape[1] if pi_all is not None else 5)).to(device)
    flow = RectifiedFlow(net, source_std=args.source_std,
                         path_noise=args.path_noise).to(device)

    n_params = sum(p.numel() for p in flow.parameters())
    print("=" * 70)
    print("Conditional rectified flow")
    print(f"  run        : {args.run_name} -> {out_dir}")
    print(f"  space      : {args.space}   dim={space.dim}")
    print(f"  k          : {k}")
    print(f"  data       : {tuple(x1_all.shape)} rows, "
          f"{len(set(fidx.tolist()))} families")
    print(f"  target rms : {float(x1_all.pow(2).mean().sqrt()):.4g}  "
          f"(source_std={args.source_std})")
    print(f"  net        : {list(args.hidden_dims)}  {n_params:,} params")
    print(f"  spectrum   : min ev0/median = "
          f"{'n/a' if min_ratio is None else f'{min_ratio:.4g}'}"
          f"   max eff-rank ratio = "
          f"{'n/a' if not effs else f'{max(effs.values()):.3f}'}")
    print("=" * 70)
    if min_ratio is not None and min_ratio < 2.0:
        print("  NOTE the spectrum is FLAT. The per-family code distribution is\n"
              "       near-Gaussian by construction, so this flow may learn only\n"
              "       the prior. Compare flow_codes against gauss_codes, NOT\n"
              "       against pca_only, when reading the eval table.")

    # --- optional held-out rows -------------------------------------------
    # Held out PER FAMILY, not uniformly at random: a random split can starve a
    # whole family, and the conditioning would then be evaluated on a label the
    # flow never trained under.
    n = x1_all.shape[0]
    train_mask = torch.ones(n, dtype=torch.bool, device=device)
    if args.holdout_frac > 0.0:
        g = np.random.default_rng(args.seed)
        for f in sorted(set(fidx.tolist())):
            rows = torch.nonzero(fidx == f, as_tuple=True)[0].cpu().numpy()
            n_hold = max(1, int(round(args.holdout_frac * len(rows))))
            for r in g.choice(rows, size=min(n_hold, len(rows) - 1), replace=False):
                train_mask[int(r)] = False
        print(f"  holdout    : {int((~train_mask).sum())} of {n} rows")

    x_tr, f_tr = x1_all[train_mask], fidx[train_mask]
    x_va, f_va = x1_all[~train_mask], fidx[~train_mask]
    # A third tensor rather than a lookup: the loader shuffles, so pi has to travel
    # WITH its row or the pairing is lost the first time a batch is drawn.
    pi_tr = pi_all[train_mask] if pi_all is not None else None
    pi_va = pi_all[~train_mask] if pi_all is not None else None

    parts = [x_tr, f_tr] + ([pi_tr] if pi_tr is not None else [])
    loader = DataLoader(TensorDataset(*parts),
                        batch_size=min(args.batch_size, x_tr.shape[0]),
                        shuffle=True)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)

    best, patience, history = float("inf"), 0, []
    ckpt = os.path.join(out_dir, "flow_best.pt")
    os.makedirs(out_dir, exist_ok=True)
    epoch = 0

    # Gating on TRAIN loss actively selects a collapsed model. Measured by
    # diag_flow_capacity.py at k=99, M=300 rows, everything else fixed:
    #
    #   width           epochs   train loss   code rms ratio
    #   64,128,64          500        1.846            1.039   <- correct
    #   256,512,256       2000        0.904            0.360
    #   512,1024,512      2000        0.650            0.158   <- worst
    #
    # Train loss falls monotonically while the samples die, so "best train loss"
    # picks the most collapsed checkpoint on the curve. That is not a subtle effect:
    # it is a 6x error in sample spread, and dPPL cannot see it because a collapsed
    # generator lands near the ensemble mean, which is essentially w_0 (misstep 19).
    if x_va.shape[0] == 0:
        print("  " + "!" * 68)
        print("  WARNING no holdout rows, so early stopping gates on TRAIN loss.")
        print("  Train loss falls as sample dispersion COLLAPSES, so this selects")
        print("  the most degenerate checkpoint. Pass --holdout_frac > 0, and read")
        print("  the rms_ratio column below before trusting any sample.")
        print("  " + "!" * 68)

    target_rms = float(x1_all.pow(2).mean().sqrt())

    # --- what "correct dispersion" means, and why it depends on the mode ----
    #
    # The pooled ratio compares the spread of SAMPLES to the spread of the whole
    # training set. Under cond_mode="family" on a one-family zoo the conditioning is
    # constant, so those are the same population and 1.0 is the right answer.
    #
    # Under cond_mode="pi" they are NOT. Samples at a FIXED pi have the conditional
    # spread; the target has the pooled spread, which also contains all the
    # between-mixture variation. PHASE2_RESULTS section 5 measures between/within at
    # ~3.96, so a PERFECT pi-conditioned flow lands near
    #
    #     1 / sqrt(1 + 3.96^2) ~ 0.25
    #
    # and 0.25 is numerically the same number misstep 19 recorded for a genuinely
    # collapsed flow. Reporting only the pooled ratio would therefore tag a correct
    # result [COLLAPSED] and a real collapse [COLLAPSED] identically.
    #
    # So emit BOTH, and say which reference each used:
    #   pooled      -- sample at the TRAINING pi's, compare to the pooled target.
    #                  Continuity with every historical number. ~1.0 is correct in
    #                  either mode, because the conditioning distribution matches.
    #   at_fixed_pi -- sample at ONE anchor's pi, compare to that anchor's own
    #                  within-group spread. This is the number that catches
    #                  memorisation-as-point-mass, which is pi-conditioning's own
    #                  failure mode (CcGAN: one real sample per label).
    anchor_rows = None
    if pi_all is not None:
        # An anchor group is >= 2 rows sharing one pi. That is exactly what the 20
        # anchors were provisioned for (RESEARCH_PLAN 6.3's within-mixture noise
        # floor); the 80 singletons have one member each and cannot supply it.
        keys: Dict[tuple, list] = {}
        for j, row in enumerate(pi_all.detach().cpu().numpy()):
            keys.setdefault(tuple(np.round(row, 6)), []).append(j)
        groups = sorted((v for v in keys.values() if len(v) > 1), key=len,
                        reverse=True)
        if groups:
            anchor_rows = groups[0]
            blk = x1_all[anchor_rows]
            within_rms = float((blk - blk.mean(dim=0, keepdim=True))
                               .pow(2).mean().sqrt())
            print(f"  within-pi  : anchor group of {len(anchor_rows)} rows, "
                  f"reference rms {within_rms:.4g} (pooled {target_rms:.4g}, "
                  f"ratio {within_rms / max(target_rms, 1e-12):.3f})")
        else:
            within_rms = None
            print("  within-pi  : no repeated pi in the training rows — the "
                  "conditional dispersion reference is unavailable.")
    else:
        within_rms = None

    def _rms_ratio(fixed_pi: bool = False) -> Optional[float]:
        """Sample spread against a training-set reference, in flow-space units.

        This is the only number in the loop that goes the WRONG way as the loss
        improves, which is why it is logged rather than computed at the end.
        """
        if args.dispersion_n < 1:
            return None
        n = args.dispersion_n
        flow.eval()
        with torch.no_grad():
            if pi_all is None:
                fv = f_tr[:1].expand(n).contiguous()
                xs = flow.sample(n, fv, n_steps=args.n_steps)
                return float(xs.pow(2).mean().sqrt()) / max(target_rms, 1e-12)
            if fixed_pi:
                if anchor_rows is None or not within_rms:
                    return None
                fv = fidx[anchor_rows[:1]].expand(n).contiguous()
                pv = pi_all[anchor_rows[:1]].expand(n, -1).contiguous()
                xs = flow.sample(n, fv, n_steps=args.n_steps, pi=pv)
                dev = xs - xs.mean(dim=0, keepdim=True)
                return float(dev.pow(2).mean().sqrt()) / max(within_rms, 1e-12)
            # Pooled: draw the conditioning from the SAME distribution the target
            # was pooled over, so the two are comparable again.
            sel = torch.randint(0, x_tr.shape[0], (n,), device=x_tr.device)
            xs = flow.sample(n, f_tr[sel], n_steps=args.n_steps, pi=pi_tr[sel])
        return float(xs.pow(2).mean().sqrt()) / max(target_rms, 1e-12)

    for epoch in range(1, args.epochs + 1):
        flow.train()
        tot, nb = 0.0, 0
        for batch in loader:
            xb, fb = batch[0], batch[1]
            pb = batch[2] if len(batch) > 2 else None
            opt.zero_grad()
            loss, _ = flow.loss(xb, fb, pi=pb)
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        tot /= max(nb, 1)
        sched.step()

        val = None
        if x_va.shape[0] > 0:
            flow.eval()
            with torch.no_grad():
                # Averaged over several t draws: a single draw makes the val curve
                # dominated by which t happened to be sampled, not by the model.
                val = float(np.mean([float(flow.loss(x_va, f_va, pi=pi_va)[0])
                                     for _ in range(8)]))

        gate = val if val is not None else tot
        if gate < best:
            best, patience = gate, 0
            torch.save(flow.state_dict(), ckpt)
        else:
            patience += 1

        row = {"epoch": epoch, "loss": round(tot, 8)}
        if val is not None:
            row["val_loss"] = round(val, 8)
        history.append(row)
        log = {f"flow/{args.space}/k{k}/train/loss": tot,
               f"flow/{args.space}/k{k}/train/lr": opt.param_groups[0]["lr"],
               f"flow/{args.space}/epoch": epoch}
        if val is not None:
            log[f"flow/{args.space}/k{k}/val/loss"] = val

        # Only measured on print epochs: sampling is 50 Euler steps, so doing it
        # every epoch would dominate a 4000-epoch run for a number nobody reads
        # between checkpoints.
        rr = _rms_ratio() if (epoch % 200 == 0 or epoch == 1) else None
        if rr is not None:
            row["rms_ratio"] = round(rr, 5)
            log[f"flow/{args.space}/k{k}/train/code_rms_ratio"] = rr
        wb.log(log, step=epoch)

        if epoch % 200 == 0 or epoch == 1:
            extra = f"  val={val:.6f}" if val is not None else ""
            tag = ""
            if rr is not None:
                tag = (f"  rms={rr:.3f}x"
                       f"{'[COLLAPSED]' if rr < 0.5 else '[shrunken]' if rr < 0.8 else ''}")
            print(f"  epoch {epoch:5d}/{args.epochs}  loss={tot:.6f}{extra}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}{tag}")
        if patience >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best={best:.6f})")
            break

    flow.load_state_dict(torch.load(ckpt, map_location=device))
    flow.eval()

    # Measured on the SELECTED weights, not the last epoch's -- the whole point is
    # that checkpoint selection can pick a collapsed model.
    final_rms_ratio = _rms_ratio()
    final_rms_at_pi = _rms_ratio(fixed_pi=True)
    if final_rms_at_pi is not None:
        v2 = ("COLLAPSED" if final_rms_at_pi < 0.5 else
              "shrunken" if final_rms_at_pi < 0.8 else
              "over-dispersed" if final_rms_at_pi > 1.25 else "ok")
        print(f"  rms at a fixed pi : {final_rms_at_pi:.3f}x that mixture's own "
              f"within-group spread  [{v2}]")
        print("    This is the memorisation read, and it is the one the pooled "
              "ratio below\n    cannot make: a flow that has learned one point per "
              "mixture still spreads\n    correctly ACROSS mixtures, so its pooled "
              "ratio is a healthy 1.0.")
    if final_rms_ratio is not None:
        verdict = ("COLLAPSED" if final_rms_ratio < 0.5 else
                   "shrunken" if final_rms_ratio < 0.8 else
                   "over-dispersed" if final_rms_ratio > 1.25 else "ok")
        print(f"  sample rms : {final_rms_ratio:.3f}x the training target  "
              f"[{verdict}]")
        if final_rms_ratio < 0.8:
            print("  " + "!" * 68)
            print("  This flow's samples are CONTRACTED toward the family mean. On "
                  "this")
            print("  ensemble mean(w) ~ w_0, so it will post a FLATTERING dPPL while")
            print("  generating nothing. Do not read its dPPL against gauss_codes.")
            print("  Try a smaller --hidden_dims and fewer --epochs: measured at "
                  "k=99,")
            print("  (64,128,64) for 500 epochs gives 1.04x where (512,1024,512) "
                  "gives 0.16x.")
            print("  " + "!" * 68)

    # --- a sanity number worth having before any eval ---------------------
    # Round-trip residual in FLOW SPACE at the sealed step count. Not a gate -- L2
    # does not predict functional damage (misstep 13) -- but a large value here means
    # the ODE is not self-consistent and no dPPL number downstream will be readable.
    # --- did pi reach the right rows? -------------------------------------
    # A scrambled row->member join trains on shuffled conditioning and yields a FLAT
    # section 6.5 slope, which is indistinguishable after the fact from an honest
    # negative result. This correlates pairwise mixture distance against pairwise
    # code distance: if the pairing is real it is clearly positive, and if the rows
    # were shuffled it is ~0. It is a PAIRING check, independent of whether the flow
    # learned anything, and at M=100 it is a 100x100 computation, i.e. free.
    pi_corr = None
    if pi_all is not None and x1_all.shape[0] > 2:
        with torch.no_grad():
            dp = torch.cdist(pi_all, pi_all, p=1)
            dx = torch.cdist(x1_all, x1_all, p=2)
            iu = torch.triu_indices(dp.shape[0], dp.shape[0], offset=1)
            a, b = dp[iu[0], iu[1]], dx[iu[0], iu[1]]
            a = a - a.mean(); b = b - b.mean()
            pi_corr = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        print(f"\n  pi<->code distance corr : {pi_corr:+.4f}  "
              f"{'OK' if pi_corr > 0.1 else 'SUSPECT — is the row join right?'}")

    with torch.no_grad():
        _, _, diag = flow.round_trip(x1_all, fidx, n_steps=args.n_steps, pi=pi_all)
    print(f"\n  round-trip rel_l2 in {args.space} space at {args.n_steps} steps: "
          f"{diag['rel_l2_space']:.4g}")
    print(f"  reverse pass landed at rms {diag['x0_hat_rms']:.4g} "
          f"(source_std {flow.source_std})")

    ens_fp = ensemble_fingerprint(ens_meta)
    pca_fps = {
        a: pca_fingerprint(
            read_json(os.path.join(run_root, "pca", a, "gram_pca_meta.json")) or {})
        for a in ens_meta["arch_list"]
    }
    # Trust propagates: a flow trained on an unprovenanced VAE is itself
    # unprovenanced, and report_stack footnotes every row derived from it.
    trust = "verified"
    if args.space == "latent" and bundle.vae is not None:
        trust = bundle.vae.loaded_meta.get("provenance", {}).get("trust", "verified")

    meta = save_flow(
        flow, space, out_dir,
        provenance=provenance_block(args.run_name, k, ens_fp, pca_fps,
                                    code_stats_fp=cs.fingerprint(), trust=trust),
        train_meta={"epochs_run": epoch, "best_loss": best,
                    "n_rows": int(x_tr.shape[0]),
                    "n_holdout": int(x_va.shape[0]),
                    "target_rms": target_rms,
                    # Sealed so a collapsed flow carries the evidence forever, the
                    # same way it carries the flat spectrum it was trained on.
                    "code_rms_ratio": final_rms_ratio,
                    "code_rms_reference": ("pooled_at_training_pi"
                                           if pi_all is not None else "pooled"),
                    "code_rms_ratio_at_fixed_pi": final_rms_at_pi,
                    "n_anchor_rows": (len(anchor_rows) if anchor_rows else 0),
                    "pi_code_distance_corr": pi_corr,
                    "gated_on": "val_loss" if x_va.shape[0] > 0 else "train_loss",
                    "round_trip": diag,
                    "spectrum": spectrum,
                    "history": history},
        defaults={"n_steps": args.n_steps,
                  "guidance_scale": args.eval_guidance_scale},
        cond_meta=({"domains": pi_meta["domains"],
                    "pi_fingerprint": pi_meta["fingerprint"],
                    "zoo_dir": pi_meta["zoo_dir"]} if pi_meta else None))

    update_manifest_section(run_root, "flow", str(k), {
        **( (read_json(os.path.join(run_root, "run_manifest.json")) or {})
            .get("flow", {}).get(str(k), {}) ),
        args.space: {"dir": os.path.basename(out_dir),
                     "layout_version": meta["layout_version"],
                     "dim": meta["dim"], "trust": trust,
                     "n_steps_default": args.n_steps,
                     "cond_mode": args.cond_mode,
                     "code_rms_ratio": final_rms_ratio,
                     "code_rms_ratio_at_fixed_pi": final_rms_at_pi,
                     "spectrum": spectrum},
    })

    wb.summary({f"flow/{args.space}/best_loss": best,
                f"flow/{args.space}/epochs_run": epoch,
                f"flow/{args.space}/dim": space.dim,
                f"flow/{args.space}/round_trip_rel_l2": diag["rel_l2_space"],
                f"flow/{args.space}/k": k,
                "pipeline/min_spectrum_ev0_over_median": min_ratio,
                "pipeline/max_effective_rank_ratio":
                    max(effs.values()) if effs else None,
                "pipeline/trust": trust})
    print(f"\n{ts()} Done. Flow under {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir",
                   default=os.environ.get("ARTIFACT_DIR", "./artifacts"),
                   help="Run artifacts root. slurm/aicr_env.sh exports "
                        "ARTIFACT_DIR; this flag overrides it.")
    p.add_argument("--run_name", default="perfam")
    p.add_argument("--k", type=int, default=None,
                   help="Rank. SCALAR, unlike eval_stack.py's --k: in code space a "
                        "flow's input width IS k, so one flow cannot span ranks. "
                        "Default: N-1.")
    p.add_argument("--space", required=True, choices=["codes", "latent"])
    # net
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 512, 256])
    p.add_argument("--time_embed_dim", type=int, default=64)
    p.add_argument("--cond_dim", type=int, default=64)
    p.add_argument("--cond_mode", choices=["family", "pi"], default="family",
                   help="What the velocity field is conditioned on. 'family' "
                        "is an nn.Embedding row per architecture -- which in a "
                        "single-arch zoo is a CONSTANT, i.e. unconditional. "
                        "'pi' feeds the member's pretraining mixture through a "
                        "small MLP (RESEARCH_PLAN 6.2), which is what makes an "
                        "unseen mixture the same kind of object as a seen one. "
                        "Needs --source zoo; pi is read from zoo_meta.json.")
    p.add_argument("--cond_dropout", type=float, default=0.15,
                   help="Blanks the family embedding during training so a learned "
                        "null branch exists. That branch is what makes CFG possible "
                        "at sampling time.")
    # flow
    p.add_argument("--source_std", type=float, default=1.0,
                   help="Std of the N(0, s^2 I) source. 1.0 (not DeepWeightFlow's "
                        "0.001) so flow_latent starts from the same distribution as "
                        "the existing `generate` arm and the two are comparable. At "
                        "0.001 the source is a near-point-mass and CFG has almost "
                        "nothing to interpolate between.")
    p.add_argument("--path_noise", type=float, default=0.0,
                   help="Extra Gaussian noise on the interpolant. Keep at 0: "
                        "nonzero makes the probability path non-deterministic, so "
                        "the flow_rt_* round-trip diagnostic stops measuring ODE "
                        "error.")
    p.add_argument("--dispersion_n", type=int, default=64,
                   help="Draws used to log the sample-spread ratio during training. "
                        "This is the one logged number that gets WORSE as the loss "
                        "improves, so it is on by default. 0 disables it.")
    # optimisation
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--patience", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--holdout_frac", type=float, default=0.15,
                   help="Hold out this fraction of rows PER FAMILY and gate early "
                        "stopping on VAL loss. Nonzero by default. The previous "
                        "default of 0.0 was justified as 'the flow needs to fit the "
                        "training set before generalization is a meaningful "
                        "question' -- and that reasoning is exactly backwards: "
                        "fitting the training set IS the failure mode here. With no "
                        "holdout the gate is train loss, which falls monotonically "
                        "while sample dispersion collapses, so best-train-loss "
                        "selects the most degenerate checkpoint (misstep 21).")
    # sampling defaults, sealed into the checkpoint
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--eval_guidance_scale", type=float, default=1.0)
    # gates
    p.add_argument("--require_spectrum_ratio", type=float, default=None,
                   help="Refuse to train when the flattest retained spectrum "
                        "ev0/MEDIAN falls below this. Deliberately the bulk "
                        "statistic, not ev0/ev[k-1]: at k=N-1 that ratio is set by "
                        "the smallest direction surviving the rank floor and reads "
                        "~12 on a pure-noise ensemble, so gating on it would let a "
                        "flat spectrum straight through. Off by default so the "
                        "machinery is smoke-testable on the noise ensemble; set it "
                        "on a real ensemble.")
    p.add_argument("--allow_legacy_vae", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_wandb", action="store_true")
    wb.add_wandb_args(p)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_tag = "auto" if args.k is None else str(args.k)
    wb.init_run(job_type="train_flow", config=vars(args),
                tags=["flow", args.space, args.run_name, f"k{k_tag}"],
                enabled=not args.no_wandb, artifact_dir=args.artifact_dir,
                name_suffix=f"{args.run_name}_k{k_tag}_{args.space}",
                project=args.wandb_project, group=args.wandb_group)
    train(args)
    wb.finish()


if __name__ == "__main__":
    main()
