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
    code_dim     : PCA code dimension (k ≤ n_training_blocks − 1)
    latent_dim   : VAE bottleneck dimension
    hidden_dim   : MLP hidden layer width
    cond_dim     : embedding dimension for block_idx and family_idx
    max_blocks   : vocabulary size for block_idx embedding (upper bound on L)
    n_families   : number of distinct model families
    """

    def __init__(
        self,
        code_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        cond_dim: int = 64,
        max_blocks: int = MAX_BLOCKS,
        n_families: int = N_FAMILIES,
    ):
        super().__init__()

        self.code_dim   = code_dim
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim

        # ---- Code normalization (identity until set_code_norm is called) ----
        # Stored as buffers so they are saved/loaded with the checkpoint and
        # automatically moved with .to(device).
        self.register_buffer('_code_mean', torch.zeros(code_dim))
        self.register_buffer('_code_std',  torch.ones(code_dim))

        # ---- Conditioning embeddings ----
        self.block_idx_emb = nn.Embedding(max_blocks, cond_dim)
        self.family_emb    = nn.Embedding(n_families, cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.SiLU(),
        )

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
        Store per-dimension normalization stats so they travel with the
        checkpoint. Call once after the full code matrix is available,
        before training begins.
        """
        self._code_mean.copy_(mean.to(self._code_mean.device))
        self._code_std.copy_(std.clamp(min=1e-8).to(self._code_std.device))

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
    ) -> torch.Tensor:
        """Produce (B, cond_dim) conditioning vector."""
        bi = self.block_idx_emb(block_idx)    # (B, cond_dim)
        fi = self.family_emb(family_idx)       # (B, cond_dim)
        return self.cond_proj(torch.cat([bi, fi], dim=-1))

    def encode(
        self,
        codes: torch.Tensor,       # (B, code_dim)  raw PCA codes
        block_idx: torch.Tensor,
        family_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar), each (B, latent_dim). Input is raw codes."""
        cond = self._condition(block_idx, family_idx)
        h = self.encoder(torch.cat([self._norm(codes), cond], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(
        self,
        z: torch.Tensor,           # (B, latent_dim)
        block_idx: torch.Tensor,
        family_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return reconstructed raw codes (B, code_dim). Output is denormalized."""
        cond = self._condition(block_idx, family_idx)
        codes_norm = self.decoder(torch.cat([z, cond], dim=-1))
        return self._denorm(codes_norm)

    def forward(
        self,
        codes: torch.Tensor,
        block_idx: torch.Tensor,
        family_idx: torch.Tensor,
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
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, block_idx, family_idx)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def elbo_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        β-VAE ELBO loss computed in normalized code space for scale stability.

        recon and target are raw codes; the loss divides by _code_std so that
        the reconstruction term is always O(1) regardless of code magnitude.

        Returns
        -------
        total_loss  : scalar
        recon_loss  : scalar  (MSE in normalized space)
        kl_loss     : scalar  (KL divergence)
        """
        recon_norm  = self._norm(recon)
        target_norm = self._norm(target)
        recon_loss = F.mse_loss(recon_norm, target_norm)
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


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
