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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import MAX_BLOCKS, N_FAMILIES


class ConditionedBlockVAE(nn.Module):
    """
    β-VAE over PCA codes, conditioned on block position and model family.

    Parameters
    ----------
    code_dim       : PCA code dimension (k ≤ n_training_blocks − 1)
    latent_dim     : VAE bottleneck dimension
    hidden_dim     : MLP hidden layer width
    cond_dim       : embedding dimension for block_idx and family_idx
    max_blocks     : vocabulary size for block_idx embedding (upper bound on L)
    n_families     : number of distinct model families
    cond_dropout_p : probability of blanking the conditioning vector during
                     training (anti-collapse + CFG null condition)
    """

    def __init__(
        self,
        code_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        cond_dim: int = 64,
        max_blocks: int = MAX_BLOCKS,
        n_families: int = N_FAMILIES,
        cond_dropout_p: float = 0.15,
    ):
        super().__init__()

        self.code_dim   = code_dim
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim
        self.cond_dropout_p = cond_dropout_p

        # ---- Code normalization (identity until set_code_norm is called) ----
        # Stored as buffers so they are saved/loaded with the checkpoint and
        # automatically moved with .to(device).
        #
        # _code_mean is per-dimension (a pure shift — harmless, and centring is
        # what PCA already assumes).  _code_std is a SINGLE SCALAR broadcast over
        # all dimensions, so the relative magnitudes of the principal components
        # survive into the loss.  Kept shaped (code_dim,) rather than () so old
        # checkpoints still load; set_code_norm fills it with a constant.
        self.register_buffer('_code_mean', torch.zeros(code_dim))
        self.register_buffer('_code_std',  torch.ones(code_dim))

        # ---- Conditioning embeddings ----
        self.block_idx_emb = nn.Embedding(max_blocks, cond_dim)
        self.family_emb    = nn.Embedding(n_families, cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.SiLU(),
        )
        # Learned null condition, substituted for the real conditioning vector
        # with probability cond_dropout_p during training.  At sampling time it
        # is the unconditional branch of classifier-free guidance.
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        # ---- Encoder ----
        enc_in = code_dim + cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # ---- Decoder ----
        dec_in = latent_dim + cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, code_dim),
        )

    # ------------------------------------------------------------------
    # Code normalization helpers
    # ------------------------------------------------------------------

    def set_code_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Store normalization stats so they travel with the checkpoint. Call once
        after the full code matrix is available, before training begins.

        `mean` is used per-dimension.  `std` is reduced to a SINGLE SCALAR (its
        RMS over dimensions) and broadcast, so the loss keeps the relative scale
        of the principal components instead of whitening it away.  Pass a
        1-element tensor to set the scalar directly.
        """
        self._code_mean.copy_(mean.to(self._code_mean.device))
        std = std.to(self._code_std.device).clamp(min=1e-8)
        # RMS rather than mean: matches the total-variance scale of the code
        # matrix, so the reconstruction term stays O(1).
        global_std = std.pow(2).mean().sqrt().clamp(min=1e-8)
        self._code_std.fill_(float(global_std))

    def _norm(self, codes: torch.Tensor) -> torch.Tensor:
        """Raw codes → zero-mean unit-std codes."""
        return (codes - self._code_mean) / self._code_std

    def _denorm(self, codes_norm: torch.Tensor) -> torch.Tensor:
        """Normalized codes → raw codes."""
        return codes_norm * self._code_std + self._code_mean

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _condition(
        self,
        block_idx: torch.Tensor,   # (B,) int64
        family_idx: torch.Tensor,  # (B,) int64
        force_null: bool = False,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        """
        Produce (B, cond_dim) conditioning vector.

        force_null=True returns the learned null condition for every row — the
        unconditional branch for classifier-free guidance.  Otherwise, when
        apply_dropout and self.training and cond_dropout_p > 0, each row is
        independently replaced by the null condition with that probability.
        """
        bi = self.block_idx_emb(block_idx)    # (B, cond_dim)
        fi = self.family_emb(family_idx)       # (B, cond_dim)
        cond = self.cond_proj(torch.cat([bi, fi], dim=-1))

        null = self.null_cond.unsqueeze(0).expand_as(cond)
        if force_null:
            return null
        if apply_dropout and self.training and self.cond_dropout_p > 0.0:
            drop = torch.rand(cond.shape[0], 1, device=cond.device) < self.cond_dropout_p
            cond = torch.where(drop, null, cond)
        return cond

    def encode(
        self,
        codes: torch.Tensor,       # (B, code_dim)  raw PCA codes
        block_idx: torch.Tensor,
        family_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (mu, logvar), each (B, latent_dim). Input is raw codes.

        The encoder always sees the true conditioning — dropout is applied on the
        decoder side only, so that q(z|x,c) stays well defined and z is what has
        to carry the block identity when the decoder's conditioning is blanked.
        """
        cond = self._condition(block_idx, family_idx, apply_dropout=False)
        h = self.encoder(torch.cat([self._norm(codes), cond], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """
        Draw z ~ q(z|x) when sample=True, else return the posterior mean.

        `sample` is explicit rather than inferred from torch.is_grad_enabled():
        the old behaviour silently returned mu inside any no_grad block, which
        made the val ELBO a different objective from the train ELBO and would
        no-op any sampling code written under torch.no_grad().
        """
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(
        self,
        z: torch.Tensor,           # (B, latent_dim)
        block_idx: torch.Tensor,
        family_idx: torch.Tensor,
        force_null: bool = False,
    ) -> torch.Tensor:
        """Return reconstructed raw codes (B, code_dim). Output is denormalized."""
        cond = self._condition(block_idx, family_idx, force_null=force_null)
        codes_norm = self.decoder(torch.cat([z, cond], dim=-1))
        return self._denorm(codes_norm)

    def decode_cfg(
        self,
        z: torch.Tensor,
        block_idx: torch.Tensor,
        family_idx: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Classifier-free-guided decode.

        guidance_scale=1.0 is plain conditional decoding; >1.0 extrapolates away
        from the null condition, sharpening family/depth identity. Requires the
        model to have been trained with cond_dropout_p > 0.
        """
        cond_out = self.decode(z, block_idx, family_idx)
        if guidance_scale == 1.0:
            return cond_out
        null_out = self.decode(z, block_idx, family_idx, force_null=True)
        return null_out + guidance_scale * (cond_out - null_out)

    def forward(
        self,
        codes: torch.Tensor,
        block_idx: torch.Tensor,
        family_idx: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns
        -------
        recon   : (B, code_dim)  reconstructed PCA codes
        mu      : (B, latent_dim)
        logvar  : (B, latent_dim)
        """
        mu, logvar = self.encode(codes, block_idx, family_idx)
        z = self.reparameterize(mu, logvar, sample=sample)
        recon = self.decode(z, block_idx, family_idx)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Per-latent-dimension KL(q(z|x) || N(0,I)), averaged over the batch.

        Returns (latent_dim,) in nats. This is the diagnostic that tells collapse
        apart from healthy regularization — the scalar mean cannot.
        """
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)

    def elbo_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
        free_bits: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        β-VAE ELBO loss computed in globally-rescaled code space.

        recon and target are raw codes; the loss divides by the scalar _code_std
        so the reconstruction term is O(1) regardless of code magnitude while
        preserving the relative scale of the principal components (per-dimension
        whitening would flatten it — see the module docstring).

        free_bits is a per-dimension KL floor in nats.  Dimensions already below
        the floor contribute a constant, so beta cannot drive them to exactly
        zero.  The *penalised* KL is what enters the loss; the *true* KL is what
        gets reported, so the logged number stays comparable across settings.

        Returns
        -------
        total_loss  : scalar
        recon_loss  : scalar  (MSE in rescaled space)
        kl_loss     : scalar  (true KL, mean over batch and latent dims, nats)
        """
        recon_norm  = self._norm(recon)
        target_norm = self._norm(target)
        recon_loss = F.mse_loss(recon_norm, target_norm)

        kl_dims = self.kl_per_dim(mu, logvar)          # (latent_dim,)
        kl_loss = kl_dims.mean()                        # true KL, for reporting
        if free_bits > 0.0:
            kl_penalty = kl_dims.clamp(min=free_bits).mean()
        else:
            kl_penalty = kl_loss

        return recon_loss + beta * kl_penalty, recon_loss, kl_loss


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

    Difference from ConditionedBlockVAE, and why
    -------------------------------------------
    ConditionedBlockVAE assumes one sample = one transformer block, so it conditions
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
