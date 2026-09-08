"""
Conditioned β-VAE operating on PCA codes of transformer blocks.

Architecture
------------
The VAE takes k-dimensional PCA codes as input (not raw block weights —
those are ~15M params and would require billions of parameters in the
first MLP layer). The VAE is conditioned on:
  - block_idx  : integer position of the block within its model (0 … L-1)
  - family_idx : which model family the block comes from (0 … n_families-1)

Conditioning is implemented as learned embeddings concatenated to the
encoder/decoder inputs (simple and effective for small datasets).

KL warmup (beta annealing) is applied to avoid posterior collapse when
training on the ~100-block dataset: beta ramps linearly from 0 to beta_max
over the first `warmup_epochs` epochs.

Posterior collapse (measured, 2026-08-21 run)
---------------------------------------------
The 150-block run drove total KL to 3.3e-05 * latent_dim = 0.00106 nats/sample,
i.e. the decoder ignored z entirely and reconstructed each block from its
conditioning alone.  That is possible because (family_idx, block_idx) is a
UNIQUE KEY over the block dataset — every block has its own pair — so a
cond_dim-wide vector can memorise the whole training set.  KL peaked at 3.6
nats/sample around epoch 10 (while beta was still ramping) and then decayed to
zero as beta reached 1.0, so the information is recoverable, not absent.

Three mitigations live here:
  1. `cond_dropout_p` — randomly blanks the conditioning vector during training
     so the decoder cannot rely on the unique key.  Doubles as the groundwork
     for classifier-free guidance (the blank embedding is the null condition).
  2. `free_bits` — per-latent-dim KL floor.  Dimensions below the floor incur no
     KL penalty, so beta cannot squeeze them to exactly zero.
  3. Global-scalar code normalization instead of per-dimension.  PCA components
     are unit-norm and orthogonal, so MSE in raw code space is (up to a constant)
     MSE in weight space.  Dividing by a PER-DIM std equalises PC 0 and PC 96 —
     which differ ~3.4x in code std and ~12x in variance — and spends decoder
     capacity on directions that carry almost no weight-space energy.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmzoo.artifacts.io import require_version
from llmzoo.models.registry import N_FAMILIES

# Bumped whenever the on-disk layout under vae_k<k>/ changes.
#   1 : vae_meta.json + vae_weights.pt, with provenance binding the checkpoint to
#       an ensemble fingerprint and a code-stats fingerprint. Anything older has
#       only vae_config.json + a bare vae_best.pt and no provenance at all.
VAE_LAYOUT_VERSION = 1


# ---------------------------------------------------------------------------
# KL warmup scheduler
# ---------------------------------------------------------------------------

class BetaScheduler:
    """
    Linear warmup for the β coefficient in the VAE ELBO.

    Prevents posterior collapse when training on small datasets by starting
    with pure reconstruction loss and gradually introducing KL regularization.
    """

    def __init__(self, beta_max: float = 1.0, warmup_epochs: int = 50):
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs

    def get(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.beta_max
        return min(self.beta_max, self.beta_max * epoch / self.warmup_epochs)


# ---------------------------------------------------------------------------
# StackVAE — the whole-stack (DeepWeightFlow-aligned) variant
# ---------------------------------------------------------------------------

class StackVAE(nn.Module):
    """
    beta-VAE over PCA codes of WHOLE decoder stacks, conditioned on family only.

    Difference from the removed block-era ConditionedBlockVAE, and why
    -------------------------------------------
    That model assumed one sample = one transformer block, so it conditioned
    on `(block_idx, family_idx)`. Under the whole-stack framing a sample IS the entire
    decoder stack, so there is no block index to condition on — `family_idx` is the
    only label.

    That is not merely a simplification, it removes the mechanism behind the measured
    posterior collapse. `(family_idx, block_idx)` was a UNIQUE KEY over the block
    dataset: every block had its own pair, so a cond_dim-wide vector could memorise
    the whole training set and the decoder never needed z (measured total KL:
    0.00106 nats/sample). Here N samples share one `family_idx`, so the conditioning
    cannot identify a sample and z has to carry the difference.

    Per-family code normalization
    -----------------------------
    Each family has its OWN PCA basis, so code dimension j of family A and code
    dimension j of family B are coefficients on unrelated basis vectors with
    unrelated magnitudes. A single pooled mean/std over the stacked code matrix is
    therefore meaningless on every dimension, not just some of them.

    So `_code_mean` is (n_families, code_dim) and `_code_std` is (n_families, 1),
    both indexed by `family_idx`. Keeping them as buffers rather than normalising
    upstream means evaluation cannot forget to apply them.

    `_code_std` is a single scalar PER FAMILY (that family's RMS over dimensions),
    not per dimension: PCA components are unit-norm and orthogonal, so MSE in raw
    code space is MSE in weight space up to a constant, and per-dimension whitening
    would equalise PC 0 and PC k-1 and spend decoder capacity on directions carrying
    almost no weight-space energy.
    """

    def __init__(
        self,
        code_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        cond_dim: int = 64,
        n_families: int = N_FAMILIES,
        cond_dropout_p: float = 0.15,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_families = n_families
        self.cond_dropout_p = cond_dropout_p

        # Per-family normalization stats; identity until set_code_norm is called.
        self.register_buffer("_code_mean", torch.zeros(n_families, code_dim))
        self.register_buffer("_code_std", torch.ones(n_families, 1))

        self.family_emb = nn.Embedding(n_families, cond_dim)
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        self.encoder = nn.Sequential(
            nn.Linear(code_dim + cond_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, code_dim),
        )

    # ---------------- normalization ----------------

    def set_code_norm(self, means: torch.Tensor, stds: torch.Tensor) -> None:
        """
        means : (n_families, code_dim) per-family per-dimension code mean
        stds  : (n_families,) or (n_families, 1) per-family scalar code scale
        """
        if means.shape != self._code_mean.shape:
            raise ValueError(f"means must be {tuple(self._code_mean.shape)}, "
                             f"got {tuple(means.shape)}")
        self._code_mean.copy_(means.to(self._code_mean.device))
        s = stds.reshape(-1, 1).to(self._code_std.device).clamp(min=1e-8)
        if s.shape[0] != self.n_families:
            raise ValueError(f"stds must have {self.n_families} rows, got {s.shape[0]}")
        self._code_std.copy_(s)

    def _norm(self, codes: torch.Tensor, family_idx: torch.Tensor) -> torch.Tensor:
        return (codes - self._code_mean[family_idx]) / self._code_std[family_idx]

    def _denorm(self, codes_norm: torch.Tensor, family_idx: torch.Tensor) -> torch.Tensor:
        return codes_norm * self._code_std[family_idx] + self._code_mean[family_idx]

    # ---------------- conditioning ----------------

    def _condition(
        self,
        family_idx: torch.Tensor,
        force_null: bool = False,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        cond = self.cond_proj(self.family_emb(family_idx))
        null = self.null_cond.unsqueeze(0).expand_as(cond)
        if force_null:
            return null
        if apply_dropout and self.training and self.cond_dropout_p > 0.0:
            drop = torch.rand(cond.shape[0], 1, device=cond.device) < self.cond_dropout_p
            cond = torch.where(drop, null, cond)
        return cond

    # ---------------- forward ----------------

    def encode(self, codes: torch.Tensor, family_idx: torch.Tensor):
        # The encoder always sees the true conditioning; dropout is decoder-side, so
        # q(z|x,c) stays well defined and z is what must carry identity when the
        # decoder's conditioning is blanked.
        cond = self._condition(family_idx, apply_dropout=False)
        h = self.encoder(torch.cat([self._norm(codes, family_idx), cond], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu, logvar, sample: bool = True):
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z, family_idx, force_null: bool = False):
        cond = self._condition(family_idx, force_null=force_null)
        return self._denorm(self.decoder(torch.cat([z, cond], dim=-1)), family_idx)

    def decode_cfg(self, z, family_idx, guidance_scale: float = 1.0):
        out = self.decode(z, family_idx)
        if guidance_scale == 1.0:
            return out
        null_out = self.decode(z, family_idx, force_null=True)
        return null_out + guidance_scale * (out - null_out)

    def forward(self, codes, family_idx, sample: bool = True):
        mu, logvar = self.encode(codes, family_idx)
        z = self.reparameterize(mu, logvar, sample=sample)
        return self.decode(z, family_idx), mu, logvar

    # ---------------- loss ----------------

    @staticmethod
    def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-latent-dim KL in nats, averaged over the batch. The scalar mean cannot
        distinguish a healthy latent from a collapsed one; this can."""
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)

    def elbo_loss(self, recon, target, mu, logvar, family_idx,
                  beta: float = 1.0, free_bits: float = 0.0):
        """
        Returns (total, recon_loss, true_kl). The PENALISED kl uses the free-bits
        floor; the REPORTED kl is the true one, so logs stay comparable across
        settings.
        """
        recon_loss = F.mse_loss(self._norm(recon, family_idx),
                                self._norm(target, family_idx))
        kl_dims = self.kl_per_dim(mu, logvar)
        kl_loss = kl_dims.mean()
        kl_penalty = kl_dims.clamp(min=free_bits).mean() if free_bits > 0.0 else kl_loss
        return recon_loss + beta * kl_penalty, recon_loss, kl_loss

    # ---------------- persistence ----------------
    #
    # `vae_best.pt` (a bare state_dict, rewritten on every improving epoch) stays
    # exactly as it was: turning that into a full save() would rewrite the metadata
    # JSON up to `epochs` times for no benefit. It is scratch. `vae_weights.pt` +
    # `vae_meta.json` are the SEALED artifact, written once at the end of training
    # after the best weights are reloaded.

    CTOR_KEYS = ("code_dim", "latent_dim", "hidden_dim", "cond_dim",
                 "n_families", "cond_dropout_p")

    def config(self) -> dict:
        """The ctor kwargs, read straight off self."""
        return {key: getattr(self, key) for key in self.CTOR_KEYS}

    def save(self, save_dir: str, *, provenance: Optional[dict] = None,
             code_stats_fingerprint: Optional[str] = None,
             code_stats_dir: Optional[str] = None,
             train: Optional[dict] = None) -> None:
        """Write vae_meta.json + vae_weights.pt into `save_dir`."""
        os.makedirs(save_dir, exist_ok=True)
        meta = {
            "layout_version": VAE_LAYOUT_VERSION,
            "class": "StackVAE",
            "config": self.config(),
            "code_norm": {
                "means_shape": list(self._code_mean.shape),
                "stds_shape": list(self._code_std.shape),
                "code_stats_fingerprint": code_stats_fingerprint,
                "code_stats_dir": code_stats_dir,
            },
            "provenance": provenance or {},
            "train": train or {},
        }
        torch.save(self.state_dict(), os.path.join(save_dir, "vae_weights.pt"))
        with open(os.path.join(save_dir, "vae_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[StackVAE] sealed → {save_dir} "
              f"(code_dim={self.code_dim}, latent_dim={self.latent_dim}, "
              f"trust={meta['provenance'].get('trust', 'unknown')})")

    @classmethod
    def load(cls, save_dir: str, *, device=None, allow_legacy: bool = False,
             expect_code_dim: Optional[int] = None,
             code_stats=None) -> "StackVAE":
        """
        Load a sealed StackVAE, refusing anything it cannot verify.

        The legacy fallback (vae_config.json + vae_best.pt) is OFF by default. Those
        directories carry no provenance at all -- nothing records which ensemble or
        which code statistics produced them -- and RESEARCH_NOTES misstep 17 is
        precisely that failure. A full re-run is ~10 minutes on one V100, which is
        cheaper than debugging a mis-paired checkpoint. Pass allow_legacy=True to
        adopt one anyway; it gets stamped trust="unverified-legacy", and that label
        propagates into any flow trained on it and into the rendered report.

        code_stats
            A run_bundle.CodeStats. When given, its fingerprint must match the
            sealed one AND the loaded _code_mean/_code_std buffers must match its
            arrays. That second check is the one that catches the real failure: a
            vae_k50/ directory left behind by a run whose stage-3 statistics
            differed.
        """
        meta_path = os.path.join(save_dir, "vae_meta.json")
        legacy_cfg = os.path.join(save_dir, "vae_config.json")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(meta_path):
            if not (allow_legacy and os.path.exists(legacy_cfg)):
                raise RuntimeError(
                    f"Refusing to load {save_dir}: no vae_meta.json. This directory "
                    f"was written before VAE_LAYOUT_VERSION existed, so nothing "
                    f"binds it to an ensemble or to a code-stats artifact. A full "
                    f"re-run is ~10 minutes:\n"
                    f"    python train_stack.py --run_name <run> --k <k> ...\n"
                    f"To evaluate it anyway, unprovenanced, pass --allow_legacy_vae.")
            with open(legacy_cfg) as f:
                cfg = json.load(f)
            meta = {"layout_version": VAE_LAYOUT_VERSION, "class": "StackVAE",
                    "config": cfg, "code_norm": {},
                    "provenance": {"trust": "unverified-legacy"}, "train": {}}
            weights = os.path.join(save_dir, "vae_best.pt")
            print(f"[StackVAE] WARNING adopting LEGACY checkpoint {save_dir}\n"
                  f"           no provenance: the ensemble and code stats behind "
                  f"these weights are unknown.\n"
                  f"           every metric derived from it is marked "
                  f"trust=unverified-legacy.")
        else:
            with open(meta_path) as f:
                meta = json.load(f)
            require_version(meta, "layout_version", VAE_LAYOUT_VERSION, save_dir)
            if meta.get("class") != "StackVAE":
                raise RuntimeError(
                    f"Refusing to load {save_dir}: class={meta.get('class')!r}, not "
                    f"'StackVAE'. ConditionedBlockVAE is the legacy block pipeline "
                    f"and its codes mean something different.")
            weights = os.path.join(save_dir, "vae_weights.pt")

        cfg = meta["config"]
        if expect_code_dim is not None and cfg["code_dim"] != expect_code_dim:
            raise RuntimeError(
                f"Refusing to load {save_dir}: VAE code_dim={cfg['code_dim']} but "
                f"rank k={expect_code_dim} was requested. Train a VAE for this rank "
                f"(train_stack.py --k {expect_code_dim}).")

        model = cls(**{key: cfg[key] for key in cls.CTOR_KEYS}).to(dev)
        model.load_state_dict(torch.load(weights, map_location=dev))
        model.eval()

        if code_stats is not None:
            sealed = meta.get("code_norm", {}).get("code_stats_fingerprint")
            got = code_stats.fingerprint()
            if sealed is not None and sealed != got:
                raise RuntimeError(
                    f"Refusing to load {save_dir}: sealed code_stats_fingerprint "
                    f"{sealed} but the run's codes_k{cfg['code_dim']}/ hashes to "
                    f"{got}. The VAE was trained on different code statistics. "
                    f"Re-train, or evaluate the run this VAE belongs to.")
            want_m = code_stats.torch_means(model._code_mean.device)
            want_s = code_stats.torch_stds(model._code_std.device)
            if not torch.allclose(model._code_mean, want_m, rtol=1e-5, atol=1e-8):
                raise RuntimeError(
                    f"Refusing to load {save_dir}: the checkpoint's _code_mean "
                    f"buffer disagrees with codes_k{cfg['code_dim']}/code_stats.npz "
                    f"(max|Δ|={float((model._code_mean - want_m).abs().max()):.3g}). "
                    f"The artifact is canonical; this checkpoint is stale.")
            if not torch.allclose(model._code_std, want_s, rtol=1e-5, atol=1e-8):
                raise RuntimeError(
                    f"Refusing to load {save_dir}: the checkpoint's _code_std "
                    f"buffer disagrees with codes_k{cfg['code_dim']}/code_stats.npz "
                    f"(max|Δ|={float((model._code_std - want_s).abs().max()):.3g}). "
                    f"The artifact is canonical; this checkpoint is stale.")

        model.loaded_meta = meta
        print(f"[StackVAE] loaded {save_dir} (code_dim={cfg['code_dim']}, "
              f"latent_dim={cfg['latent_dim']}, "
              f"trust={meta.get('provenance', {}).get('trust', 'unknown')})")
        return model
