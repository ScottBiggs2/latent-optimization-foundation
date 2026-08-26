"""
Full training pipeline: block extraction → PCA → VAE training.

Pipeline stages (each stage is skipped if its checkpoint already exists):
  1. Build BlockDataset — extract transformer blocks from all model families
  2. Fit BatchedCovariancePCA on all blocks (Gram-matrix dual trick)
  3. Encode all blocks → PCA code matrix (N, k)
  4. Train ConditionedBlockVAE on PCA codes
  5. Save all artifacts to artifact_dir

All large artifacts (blocks, PCA components, codes) live under artifact_dir
which defaults to /scratch/biggs.s/llm_vae — never written to $HOME.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from data.block_dataset import BLOCK_LAYOUT_VERSION, BlockDataset
from dual_pca import BatchedCovariancePCA, load_codes
from models.registry import MAX_BLOCKS, N_FAMILIES, list_archs
from vae import BetaScheduler, ConditionedBlockVAE
import wandb_utils as wb


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def stage_extract(
    args, arch_list: list[str], blocks_dir: str
) -> BlockDataset:
    """Build BlockDataset. Skips if all_blocks.npy already exists."""
    blocks_file = os.path.join(blocks_dir, "all_blocks.npy")
    meta_file   = os.path.join(blocks_dir, "dataset_meta.json")

    if os.path.exists(blocks_file) and os.path.exists(meta_file) and not args.force_extract:
        print(f"{ts()} Found existing blocks at {blocks_file} — loading metadata …")
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"  {meta['total_blocks']} blocks × {meta['max_block_size']:,} params")
        # Re-create dataset without re-extracting (loads from memmap)
        dataset = _load_existing_dataset(meta, args)
        return dataset

    print(f"{ts()} Stage 1: Extracting transformer blocks …")
    dataset = BlockDataset(
        arch_list=arch_list,
        noise_scale=args.noise_scale,
        mode=args.mode,
        artifact_dir=args.artifact_dir,
        augment=True,
        exclude_1d=getattr(args, "exclude_1d", False),
    )
    return dataset


def _load_existing_dataset(meta: dict, args) -> BlockDataset:
    """Re-instantiate BlockDataset from existing memmap files (fast path)."""
    # Version gate. Every stage in this pipeline is guarded only by
    # os.path.exists, so an artifact directory left over from a run with a
    # different layout is resumed rather than rejected. That is not a hypothetical:
    # the shapes are read from meta and applied to raw memmaps, so a mismatch
    # produces plausible-looking numbers instead of an error.
    found = meta.get("layout_version")
    if found != BLOCK_LAYOUT_VERSION:
        raise RuntimeError(
            f"Refusing to resume from {os.path.join(args.artifact_dir, 'blocks')}: "
            f"dataset_meta.json reports layout_version={found!r}, but this code "
            f"expects {BLOCK_LAYOUT_VERSION}. "
            f"{'Artifacts written before 2026-08-25 carry no version field and are '
               'from the padded six-family block layout.' if found is None else ''} "
            f"Re-extract with --force_extract, or point --artifact_dir at a fresh "
            f"directory."
        )
    dataset = BlockDataset.__new__(BlockDataset)
    blocks_dir = os.path.join(args.artifact_dir, "blocks")

    dataset.arch_list       = meta["arch_list"]
    dataset.noise_scale     = args.noise_scale
    dataset.mode            = meta["mode"]
    dataset.artifact_dir    = args.artifact_dir
    dataset.augment         = True
    dataset.max_block_size  = meta["max_block_size"]
    dataset.exclude_1d      = bool(meta.get("exclude_1d", False))

    total   = meta["total_blocks"]
    max_sz  = meta["max_block_size"]
    dataset._blocks = np.memmap(
        os.path.join(blocks_dir, "all_blocks.npy"),
        dtype=np.float32, mode="r", shape=(total, max_sz))
    dataset._masks = np.memmap(
        os.path.join(blocks_dir, "all_masks.npy"),
        dtype=np.uint8, mode="r", shape=(total, max_sz))

    # Rebuild the per-block metadata lists.
    #
    # Schemas: built from TINY models (hidden=128) as a structural template. The
    # parameter *names* and their order match the real model; the *shapes* do
    # not. That is fine only because nothing reads these schemas on the resume
    # path — eval_lm/eval_mc re-extract the schema from the freshly loaded real
    # model (eval_lm.reconstruct_model_blocks), and evaluate.py only touches
    # _blocks/_masks. `_real_sizes` and `_block_stds` used to be derived from
    # those tiny shapes too, which made them silently wrong; they now come from
    # dataset_meta.json where the extraction stage recorded the real values.
    from models.registry import build_tiny_model, get_arch_config, get_layers
    from models.weight_extractor import extract_block_flat

    exclude_1d = bool(meta.get("exclude_1d", False))
    block_idxs, family_idxs, arch_names, schemas = [], [], [], []
    tiny_sizes = []
    for arch in meta["arch_list"]:
        model = build_tiny_model(arch)
        cfg = get_arch_config(arch)
        layers = get_layers(model, arch)
        n_this = meta["blocks_per_arch"][arch]
        for i in range(n_this):
            flat, schema = extract_block_flat(
                layers[min(i, len(layers) - 1)], exclude_1d=exclude_1d)
            block_idxs.append(i)
            family_idxs.append(int(cfg["family_idx"]))
            arch_names.append(arch)
            schemas.append(schema)
            tiny_sizes.append(len(flat))
        del model; gc.collect()

    dataset._block_idxs  = np.array(block_idxs,  dtype=np.int64)
    dataset._family_idxs = np.array(family_idxs, dtype=np.int64)
    dataset._arch_names  = arch_names
    dataset._schemas     = schemas

    # Prefer the recorded real values; fall back to the tiny-model sizes only for
    # datasets extracted before those fields were written, and say so loudly.
    if "real_sizes" in meta and "block_stds" in meta:
        dataset._real_sizes = np.array(meta["real_sizes"], dtype=np.int64)
        dataset._block_stds = np.array(meta["block_stds"], dtype=np.float32)
    else:
        print("  WARNING: dataset_meta.json predates real_sizes/block_stds. "
              "Falling back to tiny-model parameter counts, which are WRONG for "
              "full mode. Re-extract with --force_extract if anything reads "
              "get_real_size()/get_block_std().")
        dataset._real_sizes = np.array(tiny_sizes, dtype=np.int64)
        dataset._block_stds = np.zeros(len(tiny_sizes), dtype=np.float32)
    return dataset


def stage_pca(args, dataset: BlockDataset, pca_dir: str) -> BatchedCovariancePCA:
    """Fit or load PCA on all blocks."""
    meta_path = os.path.join(pca_dir, "pca_meta.json")

    if os.path.exists(meta_path) and not args.force_pca:
        print(f"{ts()} Found existing PCA at {pca_dir} — loading …")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return BatchedCovariancePCA.load(pca_dir, device=device)

    print(f"{ts()} Stage 2: Fitting PCA on {len(dataset)} blocks "
          f"(max_block_size={dataset.max_block_size:,}) …")

    N = len(dataset)
    n_comp = min(args.n_components, N - 1)
    if n_comp != args.n_components:
        print(f"  --n_components={args.n_components} capped to {n_comp} (= N_blocks - 1). "
              f"At n_comp = N-1 the basis spans the entire affine hull of the "
              f"training blocks, so projection is lossless in-sample and any "
              f"decoded sample is confined to that hull.")
    loader = dataset.make_loader()   # creates one in-memory copy of all blocks

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca = BatchedCovariancePCA(n_components=n_comp, device=device)
    pca.fit(loader, n_models=N, batch_size=args.pca_batch_size)
    pca.save(pca_dir)
    gc.collect()
    return pca


def stage_encode(
    args, dataset: BlockDataset, pca: BatchedCovariancePCA, vae_dir: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode all blocks to PCA codes. Returns (codes, block_idxs, family_idxs)."""
    codes_path = os.path.join(vae_dir, "pca_codes.npy")

    if os.path.exists(codes_path) and not args.force_encode:
        print(f"{ts()} Found existing PCA codes at {codes_path} — loading …")
        # load_codes validates the shape instead of silently reinterpreting a stale
        # file — see dual_pca.load_codes for why that mattered.
        codes_np = load_codes(codes_path, n_models=len(dataset),
                              n_components=pca.n_components)
    else:
        print(f"{ts()} Stage 3: Encoding {len(dataset)} blocks → PCA codes …")
        N = len(dataset)
        loader = dataset.make_loader()   # one in-memory copy for encoding

        # Use PCA.transform() which streams batches
        codes_file = os.path.join(vae_dir, "pca_codes.npy")
        pca.transform(loader, n_models=N, batch_size=args.pca_batch_size,
                      output_file=codes_file)
        codes_np = load_codes(codes_file, n_models=N,
                              n_components=pca.n_components)
        gc.collect()

    codes      = torch.from_numpy(codes_np).float()
    block_idxs = torch.from_numpy(dataset._block_idxs).long()
    family_idxs = torch.from_numpy(dataset._family_idxs).long()
    print(f"  codes shape: {codes.shape}  (N={codes.shape[0]}, k={codes.shape[1]})")
    return codes, block_idxs, family_idxs


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_vae(
    args,
    codes: torch.Tensor,
    block_idxs: torch.Tensor,
    family_idxs: torch.Tensor,
    vae_dir: str,
    block_stds: "np.ndarray | None" = None,
) -> ConditionedBlockVAE:
    """
    Train ConditionedBlockVAE on PCA codes.

    block_stds : optional (N,) per-block weight std, used to scale the
                 `--noise_scale` augmentation. Pass dataset.block_stds_numpy().
    """
    vae_path   = os.path.join(vae_dir, "vae_best.pt")
    cfg_path   = os.path.join(vae_dir, "vae_config.json")
    metrics_path = os.path.join(vae_dir, "train_metrics.json")

    N, k = codes.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} Stage 4: Training VAE on {N} blocks, code_dim={k}, device={device}")

    # Normalization stats from the full code matrix. Stored as VAE buffers so
    # they travel with the checkpoint and are available to evaluation code
    # without a separate file. set_code_norm() reduces the std to a single
    # scalar on purpose — see the vae.py module docstring.
    code_mean   = codes.mean(dim=0)
    code_std    = codes.std(dim=0).clamp(min=1e-8)
    per_dim_std = code_std                      # kept for the noise augmentation
    print(f"  Code scale: mean_abs={code_mean.abs().mean():.2f}, "
          f"per-dim std {code_std.max():.2f} (PC 0) → {code_std.min():.2f} (PC {k-1}), "
          f"ratio {float(code_std.max() / code_std.min()):.2f}x")
    print(f"  ELBO uses a single global scale ({float(code_std.pow(2).mean().sqrt()):.2f}) "
          f"so relative PC magnitudes survive into the loss.")

    cond_dropout_p = getattr(args, "cond_dropout", 0.15)
    free_bits      = getattr(args, "free_bits", 0.05)

    model = ConditionedBlockVAE(
        code_dim=k,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        max_blocks=MAX_BLOCKS,
        n_families=N_FAMILIES,
        cond_dropout_p=cond_dropout_p,
    ).to(device)

    cfg_dict = {
        "code_dim": k,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "cond_dim": args.cond_dim,
        "max_blocks": MAX_BLOCKS,
        "n_families": N_FAMILIES,
        "cond_dropout_p": cond_dropout_p,
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Attach normalization stats — saved in checkpoint, used by eval code
    model.set_code_norm(code_mean, code_std)

    # Train/val split.
    # With only ~100 blocks, val_fraction=0 (train on all data) is correct for
    # exploration: we care about reconstruction fidelity on known weights, not
    # generalization to unseen architectures.  Use train-loss plateau stopping.
    val_fraction = getattr(args, "val_fraction", 0.0)
    # row_idx rides along so the augmentation can look up each block's own weight
    # std; random_split shuffles rows, so positional indexing is not enough.
    row_idx = torch.arange(N, dtype=torch.long)
    full_ds = TensorDataset(codes, block_idxs, family_idxs, row_idx)

    if val_fraction <= 0.0:
        train_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = None
        print(f"  Training on all {N} blocks (val_fraction=0, train-loss plateau stopping)")
    else:
        n_val   = max(1, int(val_fraction * N))
        n_train = N - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
        print(f"  Train/val split: {n_train} train / {n_val} val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )
    beta_scheduler = BetaScheduler(beta_max=args.beta, warmup_epochs=args.warmup_epochs)

    best_val_loss = float("inf")
    patience_count = 0
    history: list[dict] = []

    # Augmentation noise, applied to the PCA codes each batch.
    #
    # The old weight-space path in BlockDataset.__getitem__ never ran: PCA reads
    # blocks via make_loader() and the VAE trains on precomputed codes, so
    # nothing called __getitem__. Doing it in code space instead is justified by
    # linearity: the components V are unit-norm and orthogonal, so isotropic
    # weight-space noise eps ~ N(0, s^2 I_D) projects to V eps ~ N(0, s^2 I_k) —
    # isotropic with the SAME std. That equivalence is exact only when the block
    # occupies the whole padded vector; for a padded block the true noise lives in
    # a d < D subspace and V_real V_real^T is a scaled projection rather than the
    # identity, so this slightly overstates the magnitude there. With the 350-400M
    # roster padding is 0% / 22% / 0%, so the gap is small. Revisit if a
    # heavily-padded family comes back.
    #
    # `--noise_scale` keeps its original meaning (std relative to the block's own
    # weight std). `--code_noise_std` is the same idea expressed relative to the
    # per-PC code std, which is the scale that actually bites here — the old 1e-7
    # weight-relative default sits ~9 orders of magnitude below the code scale,
    # i.e. it was a no-op even before the dead-code bug.
    noise_scale     = float(getattr(args, "noise_scale", 0.0) or 0.0)
    code_noise_std  = float(getattr(args, "code_noise_std", 0.0) or 0.0)
    block_std_t = None
    if noise_scale > 0.0 and block_stds is not None:
        block_std_t = torch.as_tensor(block_stds, dtype=torch.float32, device=device)
    per_dim_std_t = per_dim_std.to(device)
    augment_on = (noise_scale > 0.0 and block_std_t is not None) or code_noise_std > 0.0
    if augment_on:
        print(f"  Code-space augmentation: noise_scale={noise_scale:g} "
              f"(weight-relative), code_noise_std={code_noise_std:g} (PC-relative)")
    else:
        print("  Code-space augmentation: OFF")

    for epoch in range(1, args.epochs + 1):
        beta = beta_scheduler.get(epoch)

        # ---- train ----
        model.train()
        train_loss = train_recon = train_kl = 0.0
        kl_dims_sum = torch.zeros(args.latent_dim, device=device)
        for codes_b, bidx_b, fidx_b, row_b in train_loader:
            codes_b = codes_b.to(device)
            bidx_b  = bidx_b.to(device)
            fidx_b  = fidx_b.to(device)

            if augment_on:
                noise = torch.zeros_like(codes_b)
                if code_noise_std > 0.0:
                    noise += torch.randn_like(codes_b) * (code_noise_std * per_dim_std_t)
                if block_std_t is not None:
                    sig = (noise_scale * block_std_t[row_b.to(device)]).unsqueeze(1)
                    noise += torch.randn_like(codes_b) * sig
                codes_in = codes_b + noise
            else:
                codes_in = codes_b

            optimizer.zero_grad()
            recon, mu, logvar = model(codes_in, bidx_b, fidx_b)
            # Reconstruct toward the CLEAN codes — the noise is input-side
            # augmentation, not a target perturbation.
            loss, rl, kl = model.elbo_loss(recon, codes_b, mu, logvar,
                                            beta=beta, free_bits=free_bits)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss  += loss.item()
            train_recon += rl.item()
            train_kl    += kl.item()
            with torch.no_grad():
                kl_dims_sum += model.kl_per_dim(mu, logvar)
        n_batches = len(train_loader)
        train_loss  /= n_batches
        train_recon /= n_batches
        train_kl    /= n_batches
        kl_dims = (kl_dims_sum / n_batches).detach()
        kl_total_nats = float(kl_dims.sum())
        active_units  = int((kl_dims > 0.01).sum())

        # ---- val (optional) ----
        model.eval()
        if val_loader is not None:
            val_loss = val_recon = val_kl = 0.0
            with torch.no_grad():
                for codes_b, bidx_b, fidx_b, _row_b in val_loader:
                    codes_b = codes_b.to(device)
                    bidx_b  = bidx_b.to(device)
                    fidx_b  = fidx_b.to(device)
                    # sample=True keeps the val ELBO the same objective as train.
                    recon, mu, logvar = model(codes_b, bidx_b, fidx_b, sample=True)
                    loss, rl, kl = model.elbo_loss(recon, codes_b, mu, logvar,
                                                    beta=beta, free_bits=free_bits)
                    val_loss  += loss.item()
                    val_recon += rl.item()
                    val_kl    += kl.item()
            val_loss  /= len(val_loader)
            val_recon /= len(val_loader)
            val_kl    /= len(val_loader)
            monitor_loss = val_loss
        else:
            # No val split: use train loss for plateau stopping
            val_loss = val_recon = val_kl = float("nan")
            monitor_loss = train_loss

        scheduler.step(monitor_loss)

        # Early stopping and best-checkpoint selection only start once beta has
        # reached its final value. During warmup the objective itself changes
        # every epoch, so comparing monitor_loss across epochs compares different
        # functions — beta ramping in pushes the loss UP while reconstruction is
        # still improving, which previously let the patience counter fill up
        # during warmup and stop the run before it had really started.
        in_warmup = epoch <= args.warmup_epochs

        if monitor_loss < best_val_loss or in_warmup:
            best_val_loss = monitor_loss
            torch.save(model.state_dict(), vae_path)
            patience_count = 0
        else:
            patience_count += 1

        row = {
            "epoch": epoch, "beta": round(beta, 4),
            "train_loss": round(train_loss, 6), "train_recon": round(train_recon, 6),
            "train_kl": round(train_kl, 6),
            "kl_total_nats": round(kl_total_nats, 6),
            "active_units": active_units,
            "val_loss": round(val_loss, 6) if not (val_loss != val_loss) else None,
            "val_recon": round(val_recon, 6) if not (val_recon != val_recon) else None,
            "val_kl": round(val_kl, 6) if not (val_kl != val_kl) else None,
        }
        history.append(row)

        wb_row = {
            "beta": beta,
            "lr": optimizer.param_groups[0]["lr"],
            "patience": patience_count,
            "train/loss": train_loss, "train/recon": train_recon, "train/kl": train_kl,
            # The collapse diagnostics. train/kl alone cannot distinguish a
            # healthy latent from a dead one; these two can.
            "kl/total_nats_per_sample": kl_total_nats,
            "kl/active_units": active_units,
            "kl/max_dim_nats": float(kl_dims.max()),
        }
        if val_loader is not None:
            wb_row.update({"val/loss": val_loss, "val/recon": val_recon, "val/kl": val_kl})
        wb.log(wb_row, step=epoch)

        if epoch % 50 == 0 or epoch == 1:
            head = (f"  epoch {epoch:4d}/{args.epochs}  train={train_loss:.5f}  ")
            if val_loader is not None:
                head += f"val={val_loss:.5f}  recon={val_recon:.5f}  "
            else:
                head += f"recon={train_recon:.5f}  "
            print(head +
                  f"kl={train_kl:.6f}  KL={kl_total_nats:.3f}nats  "
                  f"active={active_units}/{args.latent_dim}  beta={beta:.3f}  "
                  f"patience={patience_count}/{args.patience}"
                  f"{'  [warmup]' if in_warmup else ''}")

        if patience_count >= args.patience:
            label = "val" if val_loader is not None else "train"
            print(f"  Early stopping at epoch {epoch} (best {label}={best_val_loss:.6f})")
            break

    kl_dims_final = [round(float(x), 6) for x in kl_dims.cpu()]
    with open(metrics_path, "w") as f:
        json.dump({
            "best_monitor_loss": best_val_loss,
            "final_kl_total_nats": kl_total_nats,
            "final_active_units": active_units,
            "final_kl_per_dim": kl_dims_final,
            "cond_dropout_p": cond_dropout_p,
            "free_bits": free_bits,
            "history": history,
        }, f, indent=2)
    print(f"{ts()} VAE training done — best loss={best_val_loss:.6f}")
    # Collapse check. An absolute threshold is misleading once free_bits is on,
    # because free_bits alone guarantees a KL of free_bits * latent_dim nats even
    # when every dimension is pinned at the floor and carries no information. So
    # compare against the floor, not against a constant.
    kl_floor = free_bits * args.latent_dim
    print(f"  Final KL: {kl_total_nats:.4f} nats/sample across "
          f"{active_units}/{args.latent_dim} active dims"
          f"{f' (free-bits floor: {kl_floor:.3f})' if kl_floor > 0 else ''}")
    if kl_total_nats <= max(0.5, 1.10 * kl_floor):
        print(f"  WARNING: KL ({kl_total_nats:.4f} nats) is at or near the "
              f"free-bits floor ({kl_floor:.3f}) — every latent dim is pinned at "
              f"the floor, so the latent is carrying no real information and the "
              f"decoder is reconstructing from its conditioning alone. Raise "
              f"--cond_dropout, or lower --beta. Generative machinery built on "
              f"this latent cannot work.")
    print(f"  Checkpoint → {vae_path}")

    # Load best weights
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="LLM-VAE block-wise training pipeline")

    # Data & model selection
    p.add_argument("--arch_list", nargs="+", default=list_archs(),
                   help="Architectures to include (default: all 4)")
    p.add_argument("--mode", choices=["tiny", "full"], default="full",
                   help="'tiny'=random-init local test; 'full'=pretrained")
    p.add_argument("--noise_scale", type=float, default=1e-7,
                   help="Augmentation noise std relative to each block's weight "
                        "std. Applied to PCA codes (exact by PCA linearity — see "
                        "train_vae). NOTE 1e-7 is ~9 orders of magnitude below the "
                        "code scale, i.e. effectively off; use --code_noise_std.")
    p.add_argument("--code_noise_std", type=float, default=0.02,
                   help="Augmentation noise std relative to each PCA dimension's "
                        "own std. This is the knob that actually bites at this "
                        "dataset size. 0 disables.")
    p.add_argument("--exclude_1d", action="store_true",
                   help="Exclude 1-D params (norm gains, biases) from the PCA/VAE. "
                        "They keep their pretrained values on reconstruction.")

    # PCA
    p.add_argument("--n_components", type=int, default=97,
                   help="Max PCA components (capped at N_blocks-1)")
    p.add_argument("--pca_batch_size", type=int, default=10,
                   help="Batch size for streaming PCA passes (memory: batch×max_block_size×4B)")

    # VAE architecture
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--cond_dim",   type=int, default=64)

    # VAE training
    p.add_argument("--val_fraction",   type=float, default=0.0,
                   help="Fraction of blocks for validation (0=train on all data, "
                        "use train-loss plateau stopping — correct for ≤200 blocks)")
    p.add_argument("--epochs",         type=int,   default=500)
    p.add_argument("--patience",       type=int,   default=50)
    p.add_argument("--warmup_epochs",  type=int,   default=50,
                   help="Epochs to linearly ramp beta from 0 → beta")
    p.add_argument("--beta",           type=float, default=1.0)
    p.add_argument("--free_bits",       type=float, default=0.05,
                   help="Per-latent-dim KL floor in nats. Dims below the floor "
                        "incur no penalty, so beta cannot crush them to zero. "
                        "0.05 x latent_dim is the resulting KL floor.")
    p.add_argument("--cond_dropout",    type=float, default=0.15,
                   help="Probability of replacing the conditioning vector with the "
                        "learned null embedding during training. Stops the decoder "
                        "using (family_idx, block_idx) as a lookup key, and gives "
                        "classifier-free guidance its unconditional branch.")
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--batch_size",     type=int,   default=32)

    # Paths
    p.add_argument("--artifact_dir", type=str,
                   default="/scratch/biggs.s/llm_vae",
                   help="Root output directory (all artifacts written here)")

    # Force re-run flags
    p.add_argument("--force_extract", action="store_true",
                   help="Re-extract blocks even if they exist")
    p.add_argument("--force_pca",     action="store_true",
                   help="Re-fit PCA even if checkpoint exists")
    p.add_argument("--force_encode",  action="store_true",
                   help="Re-encode blocks even if codes exist")
    p.add_argument("--force_train",   action="store_true",
                   help="Re-train VAE even if checkpoint exists")

    # Evaluation
    p.add_argument("--eval_lm", action="store_true",
                   help="Run LM perplexity evaluation after training")
    p.add_argument("--eval_seq_len",    type=int, default=512)
    p.add_argument("--eval_n_sequences", type=int, default=16)

    # Reporting
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable Weights & Biases logging for this run")

    args = p.parse_args()

    wb.init_run(
        job_type="train",
        config=vars(args),
        tags=[args.mode] + args.arch_list,
        enabled=not args.no_wandb,
        artifact_dir=args.artifact_dir,
    )

    # Create artifact directories
    pca_dir  = os.path.join(args.artifact_dir, "pca")
    vae_dir  = os.path.join(args.artifact_dir, "vae")
    res_dir  = os.path.join(args.artifact_dir, "results")
    for d in [args.artifact_dir, pca_dir, vae_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    blocks_dir = os.path.join(args.artifact_dir, "blocks")

    print(f"\n{'='*60}")
    print(f"LLM-VAE Training Pipeline")
    print(f"  arch_list    : {args.arch_list}")
    print(f"  mode         : {args.mode}")
    print(f"  artifact_dir : {args.artifact_dir}")
    print(f"  n_components : {args.n_components}")
    print(f"  latent_dim   : {args.latent_dim}")
    print(f"{'='*60}\n")

    # ---- Stage 1: Block extraction ----
    dataset = stage_extract(args, args.arch_list, blocks_dir)

    # ---- Stage 2: PCA fit ----
    pca = stage_pca(args, dataset, pca_dir)

    # ---- Stage 3: Encode ----
    codes, block_idxs, family_idxs = stage_encode(args, dataset, pca, vae_dir)

    # ---- Stage 4: Train VAE ----
    vae_path = os.path.join(vae_dir, "vae_best.pt")
    if os.path.exists(vae_path) and not args.force_train:
        print(f"{ts()} Found existing VAE checkpoint — loading …")
        cfg_path = os.path.join(vae_dir, "vae_config.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = ConditionedBlockVAE(**cfg_dict).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
    else:
        vae = train_vae(args, codes, block_idxs, family_idxs, vae_dir,
                        block_stds=dataset.block_stds_numpy())

    # ---- Stage 5: Evaluate ----
    print(f"\n{ts()} Stage 5: Evaluating block reconstruction …")
    from evaluate import evaluate_all
    results = evaluate_all(pca, vae, dataset, codes, block_idxs, family_idxs,
                           device=str(vae.block_idx_emb.weight.device))
    results_path = os.path.join(res_dir, "reconstruction_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  cosine_sim (global): {results['global']['cosine_sim']:.6f}")
    print(f"  mse        (global): {results['global']['mse']:.3e}")
    print(f"  Saved → {results_path}")
    wb.log({
        "recon/cosine_sim": results["global"]["cosine_sim"],
        "recon/mse": results["global"]["mse"],
        "recon/kl_divergence": results["global"]["kl_divergence"],
        **{f"recon/{arch}/cosine_sim": fm["cosine_sim"] for arch, fm in results["per_family"].items()},
        **{f"recon/{arch}/mse": fm["mse"] for arch, fm in results["per_family"].items()},
    })

    if args.eval_lm:
        print(f"\n{ts()} Stage 6: LM perplexity evaluation …")
        from eval_lm import evaluate_all_families
        lm_results = evaluate_all_families(
            pca, vae, dataset,
            seq_len=args.eval_seq_len,
            n_sequences=args.eval_n_sequences,
            mode=args.mode,
            artifact_dir=args.artifact_dir,
        )
        lm_path = os.path.join(res_dir, "lm_eval_results.json")
        with open(lm_path, "w") as f:
            json.dump(lm_results, f, indent=2)
        print(f"  Saved → {lm_path}")
        for arch, res in lm_results.items():
            print(f"  {arch}: ppl {res['original_ppl']:.2f} → {res['reconstructed_ppl']:.2f} "
                  f"(Δ={res['ppl_delta']:+.3f})")

    print(f"\n{ts()} Done. All artifacts in {args.artifact_dir}")
    wb.finish()


if __name__ == "__main__":
    main()
