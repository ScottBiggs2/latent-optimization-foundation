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

from data.block_dataset import BlockDataset
from dual_pca import BatchedCovariancePCA
from models.registry import MAX_BLOCKS, N_FAMILIES, list_archs
from vae import BetaScheduler, ConditionedBlockVAE


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
    )
    return dataset


def _load_existing_dataset(meta: dict, args) -> BlockDataset:
    """Re-instantiate BlockDataset from existing memmap files (fast path)."""
    dataset = BlockDataset.__new__(BlockDataset)
    blocks_dir = os.path.join(args.artifact_dir, "blocks")

    dataset.arch_list       = meta["arch_list"]
    dataset.noise_scale     = args.noise_scale
    dataset.mode            = meta["mode"]
    dataset.artifact_dir    = args.artifact_dir
    dataset.augment         = True
    dataset.max_block_size  = meta["max_block_size"]

    total   = meta["total_blocks"]
    max_sz  = meta["max_block_size"]
    dataset._blocks = np.memmap(
        os.path.join(blocks_dir, "all_blocks.npy"),
        dtype=np.float32, mode="r", shape=(total, max_sz))
    dataset._masks = np.memmap(
        os.path.join(blocks_dir, "all_masks.npy"),
        dtype=np.uint8, mode="r", shape=(total, max_sz))

    # Re-build block extraction is required to get schemas — use tiny approach
    # for metadata reconstruction or load from a secondary metadata file.
    # For now: rebuild tiny models just to get schemas (very fast).
    from models.registry import build_tiny_model, get_arch_config, get_layers
    from models.weight_extractor import extract_block_flat

    block_idxs, family_idxs, arch_names, schemas, real_sizes = [], [], [], [], []
    for arch in meta["arch_list"]:
        if meta["mode"] == "tiny":
            model = build_tiny_model(arch)
        else:
            # For schema reconstruction from full models we use the tiny model
            # as a structural template — parameter names/shapes are the same.
            model = build_tiny_model(arch)
        cfg = get_arch_config(arch)
        layers = get_layers(model, arch)
        n_this = meta["blocks_per_arch"][arch]
        for i in range(n_this):
            flat, schema = extract_block_flat(layers[min(i, len(layers)-1)])
            block_idxs.append(i)
            family_idxs.append(int(cfg["family_idx"]))
            arch_names.append(arch)
            schemas.append(schema)
            real_sizes.append(len(flat))
        del model; gc.collect()

    dataset._block_idxs  = np.array(block_idxs,  dtype=np.int64)
    dataset._family_idxs = np.array(family_idxs, dtype=np.int64)
    dataset._arch_names  = arch_names
    dataset._schemas     = schemas
    dataset._real_sizes  = np.array(real_sizes, dtype=np.int64)
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
        codes_np = np.load(codes_path)
    else:
        print(f"{ts()} Stage 3: Encoding {len(dataset)} blocks → PCA codes …")
        N = len(dataset)
        loader = dataset.make_loader()   # one in-memory copy for encoding

        # Use PCA.transform() which streams batches
        codes_file = os.path.join(vae_dir, "pca_codes.npy")
        pca.transform(loader, n_models=N, batch_size=args.pca_batch_size,
                      output_file=codes_file)
        codes_np = np.array(np.memmap(codes_file, dtype=np.float32, mode="r",
                                      shape=(N, pca.n_components)))
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
) -> ConditionedBlockVAE:
    """Train ConditionedBlockVAE on PCA codes."""
    vae_path   = os.path.join(vae_dir, "vae_best.pt")
    cfg_path   = os.path.join(vae_dir, "vae_config.json")
    metrics_path = os.path.join(vae_dir, "train_metrics.json")

    N, k = codes.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} Stage 4: Training VAE on {N} blocks, code_dim={k}, device={device}")

    model = ConditionedBlockVAE(
        code_dim=k,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        max_blocks=MAX_BLOCKS,
        n_families=N_FAMILIES,
    ).to(device)

    cfg_dict = {
        "code_dim": k,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "cond_dim": args.cond_dim,
        "max_blocks": MAX_BLOCKS,
        "n_families": N_FAMILIES,
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # 80/20 split
    n_val   = max(1, int(0.2 * N))
    n_train = N - n_val
    full_ds = TensorDataset(codes, block_idxs, family_idxs)
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )
    beta_scheduler = BetaScheduler(beta_max=args.beta, warmup_epochs=args.warmup_epochs)

    best_val_loss = float("inf")
    patience_count = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        beta = beta_scheduler.get(epoch)

        # ---- train ----
        model.train()
        train_loss = train_recon = train_kl = 0.0
        for codes_b, bidx_b, fidx_b in train_loader:
            codes_b = codes_b.to(device)
            bidx_b  = bidx_b.to(device)
            fidx_b  = fidx_b.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(codes_b, bidx_b, fidx_b)
            loss, rl, kl = model.elbo_loss(recon, codes_b, mu, logvar, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss  += loss.item()
            train_recon += rl.item()
            train_kl    += kl.item()
        train_loss  /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl    /= len(train_loader)

        # ---- val ----
        model.eval()
        val_loss = val_recon = val_kl = 0.0
        with torch.no_grad():
            for codes_b, bidx_b, fidx_b in val_loader:
                codes_b = codes_b.to(device)
                bidx_b  = bidx_b.to(device)
                fidx_b  = fidx_b.to(device)
                recon, mu, logvar = model(codes_b, bidx_b, fidx_b)
                loss, rl, kl = model.elbo_loss(recon, codes_b, mu, logvar, beta=beta)
                val_loss  += loss.item()
                val_recon += rl.item()
                val_kl    += kl.item()
        val_loss  /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl    /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), vae_path)
            patience_count = 0
        else:
            patience_count += 1

        row = {
            "epoch": epoch, "beta": round(beta, 4),
            "train_loss": round(train_loss, 6), "train_recon": round(train_recon, 6),
            "train_kl": round(train_kl, 6),
            "val_loss": round(val_loss, 6), "val_recon": round(val_recon, 6),
            "val_kl": round(val_kl, 6),
        }
        history.append(row)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"recon={val_recon:.5f}  kl={val_kl:.5f}  beta={beta:.3f}  "
                  f"patience={patience_count}/{args.patience}")

        if patience_count >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.6f})")
            break

    with open(metrics_path, "w") as f:
        json.dump({"best_val_loss": best_val_loss, "history": history}, f, indent=2)
    print(f"{ts()} VAE training done — best val_loss={best_val_loss:.6f}")
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
                   help="Augmentation noise std relative to block std (≤1e-6)")

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
    p.add_argument("--epochs",         type=int,   default=500)
    p.add_argument("--patience",       type=int,   default=50)
    p.add_argument("--warmup_epochs",  type=int,   default=50,
                   help="Epochs to linearly ramp beta from 0 → beta")
    p.add_argument("--beta",           type=float, default=1.0)
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

    args = p.parse_args()

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
        vae = train_vae(args, codes, block_idxs, family_idxs, vae_dir)

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


if __name__ == "__main__":
    main()
