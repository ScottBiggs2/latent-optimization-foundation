"""
Conditional rectified flow over stack PCA codes, or over StackVAE latents.

Two interchangeable target spaces, ONE code path
------------------------------------------------
`FlowSpace` is the only thing that differs between them. The velocity net, the
interpolant, the loss, the sampler, CFG, ODE integration and save/load are all
written once against the protocol:

    codes   dim = k   target = per-family-normalized PCA codes
    latent  dim = 32  target = StackVAE posterior means mu(codes)

`build_space` is the single factory. Nothing downstream branches on the space name
again -- `train_flow.py` and `eval_stack.py` both just hold a `FlowSpace`.

Relationship to DeepWeightFlow
------------------------------
The interpolant and velocity target are DeepWeightFlow's: linear path
`x_t = (1-t)x0 + t*x1`, target `u = x1 - x0`, MSE loss, Euler integration. Two
deliberate departures:

* **CFG exists here.** DeepWeightFlow's multi-class variant concatenates a class
  embedding and has no classifier-free guidance. This one follows `StackVAE` instead
  -- family embedding through a `cond_proj`, a learned `null_cond` parameter, and
  `cond_dropout_p` at train time -- because RESEARCH_NOTES Experiment 4 wants to
  anneal an unseen family in through CFG, and that is impossible without an
  unconditional branch.
* **`source_std` defaults to 1.0, not 0.001.** The existing `generate` arm samples
  `z ~ N(0, I)`, so 1.0 makes `flow_latent` directly comparable to it -- same source
  distribution, different transport. At 0.001 the source is a near-point-mass, the
  map is essentially deterministic, and CFG has almost nothing to interpolate
  between. Pass `--source_std 1e-3` to reproduce DeepWeightFlow.

The honest caveat, restated
---------------------------
The current ensemble is manufactured as `w_0 + s*sigma*eps_i`, so its per-family code
distribution is near-Gaussian BY CONSTRUCTION and a flow fit to it may learn nothing
beyond the prior. Nothing in this file fixes that; the design makes it measurable
instead. The spectrum ratio is sealed into every checkpoint's metadata, and the
`gauss_codes` eval arm is the null model these flows have to beat. Judge samples on
dPPL against `gauss_codes`, never on code-space or weight-space L2 (misstep 13).
"""

from __future__ import annotations

import abc
import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from llmzoo.artifacts.io import require_version
from llmzoo.models.registry import N_FAMILIES

# Bumped whenever the on-disk layout under flow_k<k>_<space>/ changes.
FLOW_LAYOUT_VERSION = 1


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

class FlowSpace(abc.ABC):
    """
    Maps between raw PCA-code space and the space the flow actually lives in.

    Contract
    --------
    to_flow(codes, fidx)                    (B, k)   -> (B, dim)
    from_flow(x, fidx, guidance_scale=1.0)  (B, dim) -> (B, k)

    `from_flow(to_flow(c))` is the identity up to the space's own lossiness: exact
    for `codes` (an affine rescale), a full VAE round trip for `latent`. That
    asymmetry is the point -- it is what the `vae` eval arm already measures.
    """

    name: str = "abstract"
    dim: int = 0

    @abc.abstractmethod
    def to_flow(self, codes: torch.Tensor,
                family_idx: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def from_flow(self, x: torch.Tensor, family_idx: torch.Tensor,
                  guidance_scale: float = 1.0) -> torch.Tensor: ...

    @abc.abstractmethod
    def state(self) -> dict:
        """Anything that must be SEALED into the checkpoint to rebuild this space."""

    def provenance(self) -> dict:
        return {"space": self.name, "dim": int(self.dim)}


class CodeFlowSpace(FlowSpace):
    """
    dim = k. Target is the per-family-normalized PCA codes.

    Normalization is CodeStats', identical to `StackVAE._norm`, so a codes-space flow
    and the VAE agree on scale and their samples are comparable without a conversion.

    `guidance_scale` is ignored in `from_flow`: in this space the only CFG that means
    anything is on the velocity field, applied during integration. There is no
    decoder here to guide.
    """

    name = "codes"

    def __init__(self, code_stats):
        self.code_stats = code_stats
        self.dim = int(code_stats.k)

    def to_flow(self, codes, family_idx):
        return self.code_stats.normalize(codes, family_idx)

    def from_flow(self, x, family_idx, guidance_scale: float = 1.0):
        return self.code_stats.denormalize(x, family_idx)

    def state(self) -> dict:
        return {"code_stats_fingerprint": self.code_stats.fingerprint()}


class LatentFlowSpace(FlowSpace):
    """
    dim = vae.latent_dim. Target is the VAE posterior MEAN mu(codes).

    The posterior mean, not a sample: the flow should learn the aggregate posterior,
    and injecting the encoder's own noise into the target would blur it for no gain.

    Why mu is standardized, and why the statistics are SEALED
    --------------------------------------------------------
    mu is only approximately N(0, I) -- the measured total KL was 11.36 nats over 32
    dimensions, not the 32 x 0.5 an exactly-standard posterior would give -- so a
    flow whose source is N(0, I) would be transporting between two distributions
    that are needlessly mismatched in scale. A per-family affine standardization
    fixes that.

    Those statistics are computed at TRAIN time and written into the checkpoint.
    Recomputing them at eval time from whatever code matrix happens to be around
    would silently shift the flow's target distribution, which is the kind of bug
    that shows up as "the flow got worse" and takes a day to find.
    """

    name = "latent"

    def __init__(self, vae, mu_mean: torch.Tensor, mu_std: torch.Tensor,
                 vae_guidance_scale: float = 1.0):
        self.vae = vae
        self.dim = int(vae.latent_dim)
        dev = next(vae.parameters()).device
        self.mu_mean = mu_mean.to(dev).float()          # (F, dim)
        self.mu_std = mu_std.to(dev).float().clamp(min=1e-6)   # (F, 1)
        self.vae_guidance_scale = float(vae_guidance_scale)

    @classmethod
    def fit(cls, vae, codes: torch.Tensor, family_idxs: torch.Tensor,
            n_families: int = N_FAMILIES,
            vae_guidance_scale: float = 1.0) -> "LatentFlowSpace":
        """Measure mu's per-family statistics on the training code matrix."""
        dev = next(vae.parameters()).device
        with torch.no_grad():
            mu, _ = vae.encode(codes.to(dev), family_idxs.to(dev))
        mm = torch.zeros(n_families, vae.latent_dim, device=dev)
        ms = torch.ones(n_families, 1, device=dev)
        for fi in sorted(set(int(x) for x in family_idxs.tolist())):
            block = mu[family_idxs.to(dev) == fi]
            mm[fi] = block.mean(dim=0)
            # One scalar per family, matching CodeStats' reduction: the latent
            # dimensions are not individually meaningful, so per-dimension whitening
            # would just amplify whichever dimension carries least information.
            ms[fi, 0] = block.std(dim=0).clamp(min=1e-8).pow(2).mean().sqrt()
        return cls(vae, mm, ms, vae_guidance_scale=vae_guidance_scale)

    def to_flow(self, codes, family_idx):
        with torch.no_grad():
            mu, _ = self.vae.encode(codes, family_idx)
        return (mu - self.mu_mean[family_idx]) / self.mu_std[family_idx]

    def from_flow(self, x, family_idx, guidance_scale: float = 1.0):
        z = x * self.mu_std[family_idx] + self.mu_mean[family_idx]
        with torch.no_grad():
            return self.vae.decode_cfg(
                z, family_idx,
                guidance_scale=guidance_scale if guidance_scale != 1.0
                else self.vae_guidance_scale)

    def state(self) -> dict:
        return {
            "mu_mean": self.mu_mean.detach().cpu().tolist(),
            "mu_std": self.mu_std.detach().cpu().tolist(),
            "vae_guidance_scale": self.vae_guidance_scale,
        }


def build_space(name: str, *, code_stats, vae=None, mu_stats=None,
                vae_guidance_scale: float = 1.0,
                codes: Optional[torch.Tensor] = None,
                family_idxs: Optional[torch.Tensor] = None) -> FlowSpace:
    """
    The single factory. Pass `mu_stats` to rebuild a sealed latent space, or
    `codes`/`family_idxs` to fit a fresh one.
    """
    if name == "codes":
        if code_stats is None:
            raise ValueError("the 'codes' space needs code_stats")
        return CodeFlowSpace(code_stats)
    if name == "latent":
        if vae is None:
            raise ValueError(
                "the 'latent' space needs a StackVAE: its target is mu(codes) and "
                "its samples are decoded through vae.decode_cfg. Train a VAE for "
                "this rank first, or use --space codes.")
        if mu_stats is not None:
            return LatentFlowSpace(vae,
                                   torch.as_tensor(mu_stats["mu_mean"]),
                                   torch.as_tensor(mu_stats["mu_std"]),
                                   vae_guidance_scale=vae_guidance_scale)
        if codes is None or family_idxs is None:
            raise ValueError("fitting a latent space needs codes and family_idxs")
        return LatentFlowSpace.fit(vae, codes, family_idxs,
                                   vae_guidance_scale=vae_guidance_scale)
    raise ValueError(f"unknown flow space {name!r} (expected 'codes' or 'latent')")


# ---------------------------------------------------------------------------
# Velocity field
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) in [0,1] -> (B, dim). Log-spaced sin/cos pairs, the standard form."""
    if dim % 2 != 0:
        raise ValueError(f"time_embed_dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                          device=t.device) / max(half - 1, 1))
    ang = t.reshape(-1, 1).float() * freqs.reshape(1, -1)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowVelocityNet(nn.Module):
    """
    v(x, t, family) -> (B, dim).

    Sizing: DeepWeightFlow uses [512, 1024, 2048], but on inputs orders of magnitude
    wider than these. Here `dim` is 99 or 32 and the training set is
    M = N * n_families = 300 rows, so the default is three hidden layers of
    (256, 512, 256) -- RESEARCH_NOTES asks for "a 3 to 4 layer MLP". LayerNorm + SiLU
    to match StackVAE's blocks.

    Conditioning is StackVAE's, not DeepWeightFlow's: a family embedding through
    cond_proj, a learned null_cond, and conditioning dropout at train time. The 12
    duplicated lines are deliberate -- factoring them into a shared mixin would mean
    editing StackVAE's class hierarchy, and that risk is not worth the saving.

    Two conditioning modes
    ----------------------
    `cond_mode="family"` (default) is the original: a discrete `nn.Embedding` row per
    family. `cond_mode="pi"` replaces the lookup with a small MLP over a CONTINUOUS
    vector -- the pretraining data mixture pi on the 5-simplex (RESEARCH_PLAN 6.2).

    Everything after the encoder is byte-identical between the two. `cond_proj`,
    `null_cond`, the conditioning dropout and all of `forward_cfg` act on the
    post-projection (B, cond_dim) tensor and never see the label, which is why
    classifier-free guidance carries over to a continuous condition unchanged
    (Ho & Salimans, https://arxiv.org/abs/2207.12598). The encoder-MLP pattern is
    FiLM's (Perez et al., https://arxiv.org/abs/1709.07871); DiT uses the same shape
    for its timestep/label embedder (https://arxiv.org/abs/2212.09748).

    Why this is not cosmetic: in a single-architecture zoo EVERY member shares one
    `family_idx`, so `cond_mode="family"` is literally unconditional there. The two
    modes are therefore each other's control -- same codes, same net size, same
    epochs, conditioning present or absent.

    What does NOT become pi: `family_idx` stays the discrete key for CodeStats'
    per-family normalization and for the VAE's, so it is still threaded everywhere.
    Only the conditioning input changes.

    The hazard to keep in view (CcGAN, https://arxiv.org/abs/2011.07466): with a
    continuous condition there may be exactly ONE real sample at any given label.
    Phase 2's zoo has 85 distinct pi over 100 members, and each of the 80 singletons
    is seen once, so pi is very nearly a unique key and the net can memorise a point
    mass per mixture. That is what 6.4's retrieval baseline exists to detect.
    """

    COND_MODES = ("family", "pi")

    def __init__(self, dim: int, n_families: int = N_FAMILIES, cond_dim: int = 64,
                 hidden_dims: Sequence[int] = (256, 512, 256),
                 time_embed_dim: int = 64, cond_dropout_p: float = 0.15,
                 cond_mode: str = "family", pi_dim: int = 5):
        super().__init__()
        if cond_mode not in self.COND_MODES:
            raise ValueError(
                f"unknown cond_mode {cond_mode!r} (expected one of {self.COND_MODES})")
        self.dim = int(dim)
        self.n_families = int(n_families)
        self.cond_dim = int(cond_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.time_embed_dim = int(time_embed_dim)
        self.cond_dropout_p = float(cond_dropout_p)
        self.cond_mode = str(cond_mode)
        self.pi_dim = int(pi_dim)

        # Exactly ONE encoder is built. load_state_dict is strict by default, so a
        # meta that disagrees with its weights fails loudly instead of quietly
        # carrying an unused, untrained module.
        if self.cond_mode == "pi":
            if self.pi_dim < 1:
                raise ValueError(f"cond_mode='pi' needs pi_dim >= 1, got {pi_dim}")
            self.pi_mlp = nn.Sequential(
                nn.Linear(self.pi_dim, cond_dim), nn.SiLU(),
                nn.Linear(cond_dim, cond_dim))
        else:
            self.family_emb = nn.Embedding(n_families, cond_dim)
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        layers = []
        prev = self.dim + self.time_embed_dim + cond_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, self.dim)]
        self.net = nn.Sequential(*layers)

    def config(self) -> dict:
        return {"dim": self.dim, "n_families": self.n_families,
                "cond_dim": self.cond_dim, "hidden_dims": list(self.hidden_dims),
                "time_embed_dim": self.time_embed_dim,
                "cond_dropout_p": self.cond_dropout_p,
                "cond_mode": self.cond_mode, "pi_dim": self.pi_dim}

    def _encode_cond(self, family_idx: torch.Tensor,
                     pi: Optional[torch.Tensor]) -> torch.Tensor:
        """The ONLY place the two modes differ. Both return (B, cond_dim)."""
        if self.cond_mode == "pi":
            if pi is None:
                raise ValueError(
                    "this net is pi-conditioned (cond_mode='pi') but no pi= was "
                    "passed. Sampling would silently fall back to a label the net "
                    "never learned.")
            if pi.shape[-1] != self.pi_dim:
                raise ValueError(
                    f"pi has width {pi.shape[-1]} but this net was built for "
                    f"pi_dim={self.pi_dim}. A mixture vector whose domains do not "
                    f"line up is worse than none.")
            return self.pi_mlp(pi.float())
        if pi is not None:
            # Silence here would read as "conditioning does not generalise" when the
            # truth is "the conditioning argument was discarded".
            raise ValueError(
                "this net is family-conditioned (cond_mode='family') but pi= was "
                "passed, and it would be ignored. Train with --cond_mode pi.")
        return self.family_emb(family_idx)

    def _condition(self, family_idx: torch.Tensor, force_null: bool = False,
                   apply_dropout: bool = True,
                   pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self.cond_proj(self._encode_cond(family_idx, pi))
        null = self.null_cond.unsqueeze(0).expand_as(cond)
        if force_null:
            return null
        if apply_dropout and self.training and self.cond_dropout_p > 0.0:
            drop = torch.rand(cond.shape[0], 1, device=cond.device) \
                < self.cond_dropout_p
            cond = torch.where(drop, null, cond)
        return cond

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                family_idx: torch.Tensor, force_null: bool = False,
                pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        # `pi` is LAST in every signature it appears in, here and on RectifiedFlow.
        # Not style: tests/test_flow.py passes t0/t1/n_steps to integrate() and
        # guidance_scale to forward_cfg() POSITIONALLY, so an earlier slot would
        # silently rebind nine existing call sites.
        te = self.time_mlp(sinusoidal_time_embedding(t, self.time_embed_dim))
        if pi is not None and pi.dim() == 2 and pi.shape[0] == 1 \
                and x.shape[0] != 1:
            # One mixture, a batch of draws -- the eval-side `fidx.expand(n)` idiom.
            pi = pi.expand(x.shape[0], -1)
        cond = self._condition(family_idx, force_null=force_null, pi=pi)
        return self.net(torch.cat([x, te, cond], dim=-1))

    def forward_cfg(self, x, t, family_idx,
                    guidance_scale: float = 1.0,
                    pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        v_null + s*(v_cond - v_null). Same algebra as StackVAE.decode_cfg.

        Unchanged by the conditioning mode: the null branch is `null_cond`, which is
        reached through force_null and never touches the label. With a continuous
        pi, s > 1 extrapolates away from the unconditional field in the direction of
        that mixture -- a "more of this mixture" control the discrete table could not
        express.
        """
        out = self.forward(x, t, family_idx, pi=pi)
        if guidance_scale == 1.0:
            return out
        null = self.forward(x, t, family_idx, force_null=True, pi=pi)
        return null + guidance_scale * (out - null)


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------

class RectifiedFlow(nn.Module):
    """
    Conditional rectified flow / conditional-OT path.

        x0  ~ N(0, source_std^2 I)      the source
        x1                              a data point, in FlowSpace coordinates
        t   ~ U(0, 1)                   one draw per sample
        x_t = (1 - t)*x0 + t*x1  [+ path_noise * eps]
        u   = x1 - x0                   the velocity target
        loss = MSE(v(x_t, t, c), u)

    path_noise defaults to 0.0, and that default is load-bearing rather than
    cosmetic -- but note WHERE it acts. It enters `loss` only; `integrate` never
    reads it. So it cannot break the stepper's invertibility directly. What it
    breaks is the round trip's MEANING: it trains the field against a path that is
    not the straight segment from x0 to x1, after which the learned field has no
    reason to be consistent between the forward and reverse passes, and
    `rel_l2_space` stops being attributable to Euler error. `tests_step5_flow.py`
    pins the geometric invariant behind that -- with path_noise=0 the interpolant
    lies exactly ON the segment, and with path_noise>0 it does not -- so the default
    cannot drift silently.
    """

    def __init__(self, net: FlowVelocityNet, source_std: float = 1.0,
                 path_noise: float = 0.0):
        super().__init__()
        self.net = net
        self.source_std = float(source_std)
        self.path_noise = float(path_noise)

    @property
    def dim(self) -> int:
        return self.net.dim

    def config(self) -> dict:
        return {"source_std": self.source_std, "path_noise": self.path_noise}

    def sample_source(self, n: int, device=None, generator=None) -> torch.Tensor:
        return torch.randn(n, self.dim, device=device,
                           generator=generator) * self.source_std

    # -------- training --------

    def loss(self, x1: torch.Tensor, family_idx: torch.Tensor,
             generator=None, pi: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, dict]:
        b = x1.shape[0]
        dev = x1.device
        x0 = self.sample_source(b, device=dev, generator=generator)
        t = torch.rand(b, device=dev, generator=generator)
        xt = (1.0 - t).reshape(-1, 1) * x0 + t.reshape(-1, 1) * x1
        if self.path_noise > 0.0:
            xt = xt + self.path_noise * torch.randn(
                x1.shape, device=dev, generator=generator)
        u = x1 - x0
        pred = self.net(xt, t, family_idx, pi=pi)
        loss = torch.mean((pred - u) ** 2)
        return loss, {
            "loss": float(loss.detach()),
            "t_mean": float(t.mean()),
            "target_rms": float(u.pow(2).mean().sqrt()),
            "pred_rms": float(pred.detach().pow(2).mean().sqrt()),
        }

    # -------- integration --------

    @torch.no_grad()
    def integrate(self, x: torch.Tensor, family_idx: torch.Tensor,
                  t0: float, t1: float, n_steps: int,
                  guidance_scale: float = 1.0,
                  pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Explicit Euler from t0 to t1 in n_steps uniform steps.

        dt = (t1 - t0)/n_steps, so t1 < t0 gives dt < 0 and this SAME code path
        integrates backwards. That is the whole reason `round_trip` needs no second
        implementation and cannot drift out of sync with the forward sampler.

        RK4 is deliberately absent. A rectified flow with path_noise = 0 has
        (ideally) straight paths, so Euler's local error IS the model's deviation
        from straightness -- which is exactly what the round-trip arm is trying to
        measure. A higher-order solver would hide it. Add one behind a --solver flag
        if the round-trip error turns out to be solver-dominated rather than
        model-dominated; sweep --flow_rt_steps first to find out which.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        dt = (t1 - t0) / n_steps
        cur = x
        for i in range(n_steps):
            t = torch.full((cur.shape[0],), t0 + i * dt, device=cur.device)
            cur = cur + dt * self.net.forward_cfg(cur, t, family_idx,
                                                  guidance_scale=guidance_scale,
                                                  pi=pi)
        return cur

    @torch.no_grad()
    def sample(self, n: int, family_idx: torch.Tensor, n_steps: int = 50,
               guidance_scale: float = 1.0, generator=None,
               x0: Optional[torch.Tensor] = None,
               pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Draw from the source (or take x0) and integrate 0 -> 1."""
        if x0 is None:
            x0 = self.sample_source(
                n, device=next(self.net.parameters()).device, generator=generator)
        return self.integrate(x0, family_idx, 0.0, 1.0, n_steps,
                              guidance_scale=guidance_scale, pi=pi)

    @torch.no_grad()
    def round_trip(self, x1: torch.Tensor, family_idx: torch.Tensor,
                   n_steps: int = 50, guidance_scale: float = 1.0,
                   pi: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        The only sense in which a flow has a round trip.

            x0_hat = integrate(x1,     1 -> 0)     reverse
            x1_hat = integrate(x0_hat, 0 -> 1)     forward again

        `rel_l2_space` measures TWO things at once and cannot separate them: Euler
        discretisation error, and the model's failure to be a consistent vector
        field. Sweep n_steps to tell them apart -- discretisation error falls like
        1/n_steps, model inconsistency does not.

        `x0_hat_rms` is reported against source_std because a reverse integration
        that lands far from the source distribution is diagnostic on its own, even
        when the forward pass happens to come back to the right place.

        CFG is pinned to 1.0 by default and warns if overridden: with guidance on,
        the reverse pass integrates a DIFFERENT vector field than the forward pass,
        so the residual is uninterpretable even in principle.
        """
        if guidance_scale != 1.0:
            print(f"[RectifiedFlow] WARNING round_trip with guidance_scale="
                  f"{guidance_scale} integrates a different field forwards than "
                  f"backwards; the residual is not interpretable as ODE error.")
        x0_hat = self.integrate(x1, family_idx, 1.0, 0.0, n_steps,
                                guidance_scale=guidance_scale, pi=pi)
        x1_hat = self.integrate(x0_hat, family_idx, 0.0, 1.0, n_steps,
                                guidance_scale=guidance_scale, pi=pi)
        num = float(torch.linalg.vector_norm(x1_hat - x1))
        den = float(torch.linalg.vector_norm(x1)) + 1e-30
        return x1_hat, x0_hat, {
            "rel_l2_space": num / den,
            "x0_hat_rms": float(x0_hat.pow(2).mean().sqrt()),
            "source_std": self.source_std,
            "n_steps": int(n_steps),
        }


# ---------------------------------------------------------------------------
# What eval_stack holds
# ---------------------------------------------------------------------------

@dataclass
class FlowModel:
    """A trained flow plus the space it decodes through, plus its sealed metadata."""
    flow: RectifiedFlow
    space: FlowSpace
    meta: dict = field(default_factory=dict)

    @property
    def default_steps(self) -> int:
        return int(self.meta.get("defaults", {}).get("n_steps", 50))

    @property
    def default_guidance(self) -> float:
        return float(self.meta.get("defaults", {}).get("guidance_scale", 1.0))

    @property
    def trust(self) -> str:
        return self.meta.get("provenance", {}).get("trust", "verified")

    @property
    def spectrum(self) -> dict:
        return self.meta.get("train", {}).get("spectrum", {}) or {}

    @property
    def cond(self) -> dict:
        """
        The sealed conditioning block. Defaults to family mode, which is what makes
        every flow sealed before pi-conditioning existed load unchanged.
        """
        return self.meta.get("cond", {}) or {"mode": "family"}

    @property
    def cond_mode(self) -> str:
        return str(self.cond.get("mode", "family"))

    @torch.no_grad()
    def sample_codes(self, family_idx: torch.Tensor, n_steps: Optional[int] = None,
                     guidance_scale: Optional[float] = None,
                     vae_guidance_scale: float = 1.0,
                     generator=None,
                     pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate raw PCA codes, ready for DualGramPCA.inverse_transform.

        `pi` is the requested mixture for a pi-conditioned flow, and may be one the
        zoo never trained on -- that is the zero-shot arm. `family_idx` still selects
        the basis and the per-family code normalization, so it is required either way.
        """
        x = self.flow.sample(
            family_idx.shape[0], family_idx,
            n_steps=n_steps or self.default_steps,
            guidance_scale=(self.default_guidance if guidance_scale is None
                            else guidance_scale),
            generator=generator, pi=pi)
        return self.space.from_flow(x, family_idx,
                                    guidance_scale=vae_guidance_scale)

    @torch.no_grad()
    def round_trip_codes(self, codes: torch.Tensor, family_idx: torch.Tensor,
                         n_steps: Optional[int] = None,
                         vae_guidance_scale: float = 1.0,
                         pi: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, dict]:
        """
        Encode real codes into the space, reverse then re-integrate, decode back.

        For the latent space the returned residual folds in the VAE round trip as
        well as the ODE error, which is why the diagnostic dict reports the
        space-level `rel_l2_space` separately: that one is ODE-only.
        """
        x1 = self.space.to_flow(codes, family_idx)
        x1_hat, _x0, diag = self.flow.round_trip(
            x1, family_idx, n_steps=n_steps or self.default_steps, pi=pi)
        out = self.space.from_flow(x1_hat, family_idx,
                                    guidance_scale=vae_guidance_scale)
        return out, diag


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_flow(flow: RectifiedFlow, space: FlowSpace, save_dir: str, *,
              provenance: dict, train_meta: Optional[dict] = None,
              defaults: Optional[dict] = None,
              cond_meta: Optional[dict] = None) -> dict:
    """
    Write flow_meta.json + flow_weights.pt. Same shape as vae_meta.json.

    FLOW_LAYOUT_VERSION deliberately does NOT move when pi-conditioning is added.
    The asymmetry is what we want: new code reading an old meta gets cond_mode's
    "family" default and behaves identically, while old code reading a new meta dies
    on an unexpected keyword at FlowVelocityNet(**meta["net"]) -- loud, not silent.
    Bumping instead would make every already-sealed flow unloadable, including
    runs/emb3/flow_k*_collapsed/, which CLAUDE.md forbids pruning because it is the
    only surviving evidence for misstep 21.

    `cond_meta` carries what pi-conditioning has to bind to and the net config
    cannot express: the DOMAIN ORDER the mixtures were written in, and a fingerprint
    of the pi matrix. Both are load-time refusals rather than comments.
    """
    os.makedirs(save_dir, exist_ok=True)
    meta = {
        "layout_version": FLOW_LAYOUT_VERSION,
        "class": "RectifiedFlow",
        "space": space.name,
        "dim": int(space.dim),
        "net": flow.net.config(),
        "flow": flow.config(),
        "space_state": space.state(),
        "defaults": defaults or {"n_steps": 50, "guidance_scale": 1.0},
        # Derived from the net itself, never from a caller argument, so meta["cond"]
        # and meta["net"] cannot disagree about which mode was trained.
        "cond": {
            "mode": flow.net.cond_mode,
            "pi_dim": flow.net.pi_dim,
            # The VAE decoder stays family-conditioned (RESEARCH_PLAN 6.6 makes it an
            # ablation, not the mechanism), so a latent-space pi flow has a
            # pi-conditioned velocity field and a pi-blind decoder. Sealed rather
            # than remembered.
            "decoder_cond_mode": "family",
            **(cond_meta or {}),
        },
        "provenance": provenance,
        "train": train_meta or {},
    }
    torch.save(flow.state_dict(), os.path.join(save_dir, "flow_weights.pt"))
    with open(os.path.join(save_dir, "flow_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[flow] sealed → {save_dir} (space={space.name}, dim={space.dim}, "
          f"trust={provenance.get('trust', 'unknown')})")
    return meta


def load_flow(save_dir: str, *, code_stats=None, vae=None, device=None,
              vae_guidance_scale: float = 1.0) -> FlowModel:
    """
    Rebuild a sealed flow, including its FlowSpace.

    The space is reconstructed from the SEALED state, not re-fitted: a latent space
    takes its mu statistics from the checkpoint, and a codes space checks the
    passed CodeStats against the fingerprint it was trained with. Re-fitting either
    at load time would silently move the target distribution.
    """
    meta_path = os.path.join(save_dir, "flow_meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Refusing to load {save_dir}: no flow_meta.json. Train a flow for this "
            f"rank and space (train_flow.py --k <k> --space <space>).")
    with open(meta_path) as f:
        meta = json.load(f)
    require_version(meta, "layout_version", FLOW_LAYOUT_VERSION, save_dir)
    if meta.get("class") != "RectifiedFlow":
        raise RuntimeError(f"Refusing to load {save_dir}: class="
                           f"{meta.get('class')!r}, not 'RectifiedFlow'.")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    space_name = meta["space"]

    if space_name == "codes":
        if code_stats is None:
            raise RuntimeError(
                f"Refusing to load {save_dir}: a codes-space flow needs the run's "
                f"CodeStats to denormalize its samples. Include 'codes' in "
                f"load_run's `want`.")
        sealed = meta.get("space_state", {}).get("code_stats_fingerprint")
        got = code_stats.fingerprint()
        if sealed is not None and sealed != got:
            raise RuntimeError(
                f"Refusing to load {save_dir}: trained against code stats {sealed} "
                f"but this run's codes hash to {got}. The flow's target "
                f"normalization would be wrong. Re-train the flow for this run.")
        space: FlowSpace = build_space("codes", code_stats=code_stats)
    elif space_name == "latent":
        if vae is None:
            raise RuntimeError(
                f"Refusing to load {save_dir}: a latent-space flow decodes through "
                f"vae.decode_cfg, so it cannot be used without the StackVAE it was "
                f"trained against. Include 'vae' in load_run's `want`.")
        space = build_space("latent", code_stats=code_stats, vae=vae,
                            mu_stats=meta["space_state"],
                            vae_guidance_scale=vae_guidance_scale)
    else:
        raise RuntimeError(f"Refusing to load {save_dir}: unknown space "
                           f"{space_name!r}.")

    # cond_mode defaults to "family" inside FlowVelocityNet, so a meta sealed before
    # pi-conditioning existed rebuilds byte-identically.
    net = FlowVelocityNet(**meta["net"]).to(dev)
    flow = RectifiedFlow(net, **meta["flow"]).to(dev)
    sd = torch.load(os.path.join(save_dir, "flow_weights.pt"), map_location=dev)

    # Name the mismatch. load_state_dict is strict, so a meta whose cond_mode has
    # been edited away from its weights already fails -- but it fails with
    # "Missing key(s) in state_dict: net.pi_mlp.0.weight", which reads as a corrupt
    # checkpoint rather than as the one-word disagreement it is.
    has_pi = any(k.startswith("net.pi_mlp.") for k in sd)
    has_fam = any(k.startswith("net.family_emb.") for k in sd)
    want_pi = net.cond_mode == "pi"
    if want_pi != has_pi or want_pi == has_fam:
        raise RuntimeError(
            f"Refusing to load {save_dir}: flow_meta.json says "
            f"cond_mode={net.cond_mode!r} but flow_weights.pt holds "
            f"{'pi_mlp' if has_pi else 'family_emb'} weights. The metadata and the "
            f"checkpoint disagree about how this flow was conditioned.")
    flow.load_state_dict(sd)
    flow.eval()

    if int(space.dim) != int(meta["dim"]):
        raise RuntimeError(
            f"Refusing to load {save_dir}: sealed dim={meta['dim']} but the rebuilt "
            f"{space_name} space has dim={space.dim}. The rank or latent width "
            f"changed under this checkpoint.")

    print(f"[flow] loaded {save_dir} (space={space_name}, dim={space.dim}, "
          f"trust={meta.get('provenance', {}).get('trust', 'unknown')})")
    return FlowModel(flow=flow, space=space, meta=meta)
