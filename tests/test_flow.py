"""
The conditional rectified flow, both spaces, and its persistence.

Everything here runs on CPU with no downloads and no real models. Three groups:

  * ARITHMETIC -- the interpolant, the velocity target, and the Euler stepper are
    checked against in-file brute force, and the stepper is checked against a field
    whose exact solution is known in closed form. `integrate` is the one function in
    flow.py where an off-by-one in `dt` would produce plausible-looking numbers
    forever, so it gets an exactness test at EVERY n_steps rather than a tolerance.
  * DIAGNOSTIC VALIDITY -- the round-trip arm's whole claim is that its residual
    falls like 1/n_steps when the error is Euler discretisation. If that trend does
    not hold on a randomly-initialised field, the diagnostic cannot separate solver
    error from model inconsistency and `--flow_rt_steps` is meaningless.
  * PERSISTENCE -- a sealed flow must reproduce seeded samples exactly, in both
    spaces, or refuse. Same contract as tests/test_bundle.py, same reasoning (misstep 17).

    python tests/test_flow.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import List

import numpy as np
import torch
import torch.nn as nn

from llmzoo.gen.flow import (FLOW_LAYOUT_VERSION, FlowVelocityNet, RectifiedFlow,
                  build_space, load_flow, save_flow,
                  sinusoidal_time_embedding)
from llmzoo.models.registry import N_FAMILIES
from llmzoo.artifacts.bundle import CodeStats
from llmzoo.gen.vae import StackVAE

FAILS: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _raises(fn, needle: str) -> bool:
    """True when fn() raises AND the message mentions `needle`."""
    try:
        fn()
        return False
    except Exception as exc:
        return needle in str(exc)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class ConstantField(nn.Module):
    """
    v(x, t, c) = c_vec, for every x, t and family.

    The point of a constant field is that the ODE dx/dt = c has the exact solution
    x(t1) = x(t0) + c*(t1 - t0), independent of the path. So explicit Euler is
    EXACT at any n_steps, and any deviation is arithmetic in `integrate` -- which is
    exactly the bug class a tolerance-based test would wave through.

    Duck-typed against FlowVelocityNet rather than subclassed: RectifiedFlow only
    ever touches `.dim`, `forward`, `forward_cfg` and `parameters()`.
    """

    def __init__(self, dim: int, value: float = 0.3, dtype=torch.float32):
        super().__init__()
        self.dim = int(dim)
        # dtype is a ctor argument rather than a later .double(): torch.full in
        # float32 stores 0.3 as 0.30000001192092896, and widening afterwards
        # PRESERVES that error instead of removing it -- which shows up as a 1.2e-8
        # residual in a test that claims float64 exactness and looks like a bug in
        # `integrate`.
        self.register_buffer("c", torch.full((dim,), float(value), dtype=dtype))
        # RectifiedFlow.sample reads next(self.net.parameters()).device.
        self.dummy = nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, x, t, family_idx, force_null: bool = False):
        return self.c.unsqueeze(0).expand(x.shape[0], self.dim)

    def forward_cfg(self, x, t, family_idx, guidance_scale: float = 1.0):
        return self.forward(x, t, family_idx)

    def config(self) -> dict:
        return {"dim": self.dim}


class SpyNet(nn.Module):
    """Records the (x, t) it was called with, so the interpolant can be inspected."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.lin = nn.Linear(dim, dim)
        self.seen: List[tuple] = []

    def forward(self, x, t, family_idx, force_null: bool = False):
        self.seen.append((x.detach().clone(), t.detach().clone()))
        return self.lin(x)

    def forward_cfg(self, x, t, family_idx, guidance_scale: float = 1.0):
        return self.forward(x, t, family_idx)


def _code_stats(k: int = 9, n_families: int = 3, seed: int = 0) -> CodeStats:
    """Per-family statistics that are genuinely DIFFERENT per family.

    Identical per-family stats would let a family-indexing bug pass: the whole point
    of CodeFlowSpace is that row f of `means` is applied to family f and no other.
    """
    rng = np.random.default_rng(seed)
    rows, fidx = [], []
    for f in range(n_families):
        # Distinct centre and distinct scale per family.
        rows.append(rng.standard_normal((14, k)).astype(np.float32) * (1.0 + 3.0 * f)
                    + 10.0 * f)
        fidx += [f] * 14
    return CodeStats.from_codes(
        np.concatenate(rows, axis=0), np.array(fidx, dtype=np.int64),
        {f"arch{f}": f for f in range(n_families)}, n_families=n_families)


def _vae(code_dim: int, latent_dim: int = 8, cs: CodeStats = None) -> StackVAE:
    torch.manual_seed(11)
    m = StackVAE(code_dim=code_dim, latent_dim=latent_dim, hidden_dim=32,
                 cond_dim=8, n_families=N_FAMILIES, cond_dropout_p=0.15)
    if cs is not None:
        means = torch.zeros(N_FAMILIES, code_dim)
        stds = torch.ones(N_FAMILIES, 1)
        means[:cs.n_families] = cs.torch_means()
        stds[:cs.n_families] = cs.torch_stds()
        m.set_code_norm(means, stds)
    # eval(): conditioning dropout is stochastic in train mode, and every identity
    # below (space round trips, seeded sample reproduction) assumes a deterministic
    # decoder.
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Time embedding and velocity net
# ---------------------------------------------------------------------------

def test_time_embedding() -> None:
    print("\n--- sinusoidal time embedding ---")
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    e = sinusoidal_time_embedding(t, 16)
    check("shape is (B, dim)", tuple(e.shape) == (4, 16), str(tuple(e.shape)))
    check("finite everywhere", bool(torch.isfinite(e).all()))
    check("t=0 and t=1 embed differently",
          not torch.allclose(e[0], e[3]),
          f"max|diff|={float((e[0] - e[3]).abs().max()):.4g}")
    check("t=0 gives sin=0, cos=1",
          torch.allclose(e[0, :8], torch.zeros(8)) and
          torch.allclose(e[0, 8:], torch.ones(8)))
    # An odd width would silently produce dim-1 columns and misalign the MLP.
    check("odd dim is refused",
          _raises(lambda: sinusoidal_time_embedding(t, 15), "even"))


def test_velocity_net() -> None:
    print("\n--- FlowVelocityNet: shapes and CFG algebra ---")
    for dim in (32, 99):
        for b in (1, 7):
            torch.manual_seed(3)
            net = FlowVelocityNet(dim=dim, n_families=N_FAMILIES, cond_dim=16,
                                  hidden_dims=(24, 24), time_embed_dim=8).eval()
            x = torch.randn(b, dim)
            t = torch.rand(b)
            f = torch.randint(0, N_FAMILIES, (b,))
            out = net(x, t, f)
            check(f"forward shape dim={dim} batch={b}",
                  tuple(out.shape) == (b, dim), str(tuple(out.shape)))
            check(f"forward finite dim={dim} batch={b}",
                  bool(torch.isfinite(out).all()))

    torch.manual_seed(3)
    net = FlowVelocityNet(dim=16, n_families=N_FAMILIES, cond_dim=16,
                          hidden_dims=(24, 24), time_embed_dim=8).eval()
    x, t = torch.randn(5, 16), torch.rand(5)
    f = torch.tensor([0, 1, 2, 1, 0])

    # guidance_scale == 1.0 must take the cheap path and skip the null forward
    # entirely -- bit-identical, not merely close.
    check("forward_cfg(1.0) is bit-identical to forward",
          torch.equal(net.forward_cfg(x, t, f, 1.0), net(x, t, f)))

    v_cond = net(x, t, f)
    v_null = net(x, t, f, force_null=True)
    for s in (0.0, 2.0, 3.5):
        check(f"forward_cfg({s}) == v_null + s*(v_cond - v_null)",
              torch.allclose(net.forward_cfg(x, t, f, s),
                             v_null + s * (v_cond - v_null), atol=1e-6))
    check("force_null ignores the family label",
          torch.equal(net(x, t, torch.zeros_like(f), force_null=True),
                      net(x, t, torch.full_like(f, 2), force_null=True)))
    # cond_dropout is train-time only; an eval-mode forward that varied run to run
    # would make every seeded-sample check below meaningless.
    check("eval-mode forward is deterministic",
          torch.equal(net(x, t, f), net(x, t, f)))


# ---------------------------------------------------------------------------
# Interpolant and target
# ---------------------------------------------------------------------------

def test_interpolant() -> None:
    print("\n--- interpolant and velocity target vs brute force ---")
    dim = 6
    spy = SpyNet(dim)
    flow = RectifiedFlow(spy, source_std=1.0, path_noise=0.0)
    x1 = torch.randn(9, dim)
    f = torch.randint(0, N_FAMILIES, (9,))

    g = torch.Generator().manual_seed(4242)
    loss, stats = flow.loss(x1, f, generator=g)

    # Replay the exact same draws: sample_source then rand, in that order.
    g2 = torch.Generator().manual_seed(4242)
    x0_ref = torch.randn(9, dim, generator=g2) * 1.0
    t_ref = torch.rand(9, generator=g2)
    xt_ref = torch.stack([(1.0 - t_ref[i]) * x0_ref[i] + t_ref[i] * x1[i]
                          for i in range(9)])
    u_ref = x1 - x0_ref

    xt_seen, t_seen = spy.seen[-1]
    check("x_t matches an elementwise brute force",
          torch.allclose(xt_seen, xt_ref, atol=1e-6),
          f"max|diff|={float((xt_seen - xt_ref).abs().max()):.3g}")
    check("t matches the replayed draw", torch.allclose(t_seen, t_ref))
    check("reported target_rms matches x1 - x0",
          abs(stats["target_rms"] - float(u_ref.pow(2).mean().sqrt())) < 1e-5)
    loss_ref = float(((spy.lin(xt_ref) - u_ref) ** 2).mean())
    check("loss is MSE(v(x_t,t,c), x1-x0)",
          abs(float(loss) - loss_ref) < 1e-6,
          f"{float(loss):.6g} vs {loss_ref:.6g}")

    check("t is drawn in [0,1)",
          bool((t_seen >= 0).all() and (t_seen < 1).all()))
    check("source_std scales the source",
          abs(float(RectifiedFlow(SpyNet(dim), source_std=4.0)
                    .sample_source(4096, generator=torch.Generator()
                                   .manual_seed(1)).std()) - 4.0) < 0.2)

    # path_noise: what it actually does, and where.
    #
    # path_noise enters `loss` ONLY -- `integrate` never reads it. So it cannot break
    # round-trip exactness directly; it breaks the round trip's MEANING by training
    # the field against a path that is not the straight segment, after which the
    # learned field has no reason to be consistent between the forward and reverse
    # passes. The invariant worth pinning is therefore the geometric one: with
    # path_noise=0 the interpolant lies exactly ON the segment from x0 to x1, and
    # with path_noise>0 it does not.
    for pn, want_on_segment in ((0.0, True), (0.05, False)):
        spy2 = SpyNet(dim)
        f2 = RectifiedFlow(spy2, source_std=1.0, path_noise=pn)
        gg = torch.Generator().manual_seed(77)
        f2.loss(x1, f, generator=gg)
        gg2 = torch.Generator().manual_seed(77)
        x0r = torch.randn(9, dim, generator=gg2)
        tr = torch.rand(9, generator=gg2)
        seg = (1.0 - tr).reshape(-1, 1) * x0r + tr.reshape(-1, 1) * x1
        on_segment = torch.allclose(spy2.seen[-1][0], seg, atol=1e-6)
        check(f"path_noise={pn}: interpolant "
              f"{'lies on' if want_on_segment else 'leaves'} the x0->x1 segment",
              on_segment == want_on_segment,
              f"max|diff|={float((spy2.seen[-1][0] - seg).abs().max()):.3g}")
    check("path_noise defaults to 0.0 (round-trip invertibility depends on it)",
          RectifiedFlow(SpyNet(dim)).path_noise == 0.0)
    check("source_std defaults to 1.0 (comparable to the `generate` arm)",
          RectifiedFlow(SpyNet(dim)).source_std == 1.0)


# ---------------------------------------------------------------------------
# The Euler stepper
# ---------------------------------------------------------------------------

def test_euler_exactness() -> None:
    print("\n--- Euler on a constant field ---")
    # dx/dt = c has the exact solution x(t1) = x(t0) + c*(t1 - t0), independent of
    # the path, so explicit Euler is EXACT at any n_steps and any deviation is
    # arithmetic in `integrate`.
    #
    # Two passes, because "exact" and "exact in float32" are different claims:
    #
    #   float64 -- tolerance-free. Pins dt = (t1-t0)/n_steps to 1e-12.
    #   float32 -- the production dtype. Sequential adds accumulate ~1e-6 over 100
    #              steps, so an absolute tolerance here would either be too loose to
    #              catch anything or too tight to ever pass. Instead compare against
    #              the signature of the bug being hunted: an off-by-one in the step
    #              count gives err ~ |c|/n_steps exactly, so require the measured
    #              error to sit at least 50x below that.
    dim, c = 5, 0.3
    steps = (1, 2, 3, 5, 7, 13, 50, 97)

    net64 = ConstantField(dim, c, dtype=torch.float64)
    flow64 = RectifiedFlow(net64, source_std=1.0)
    # Compare against the buffer's OWN value, so the test measures `integrate` and
    # not the representation of the constant.
    c = float(net64.c[0])
    x64 = torch.randn(4, dim, dtype=torch.float64)
    f = torch.randint(0, N_FAMILIES, (4,))

    worst_f = worst_r = 0.0
    for n in steps:
        worst_f = max(worst_f, float((flow64.integrate(x64, f, 0.0, 1.0, n)
                                      - (x64 + c)).abs().max()))
        worst_r = max(worst_r, float((flow64.integrate(x64, f, 1.0, 0.0, n)
                                      - (x64 - c)).abs().max()))
    check("float64: integrate(0->1) == x + c for every n_steps", worst_f < 1e-12,
          f"worst |err| = {worst_f:.3g}")
    # dt < 0 runs the SAME loop backwards, which is why round_trip needs no second
    # implementation. If the sign handling were wrong this is where it shows.
    check("float64: integrate(1->0) == x - c for every n_steps (dt < 0)",
          worst_r < 1e-12, f"worst |err| = {worst_r:.3g}")

    flow32 = RectifiedFlow(ConstantField(dim, c), source_std=1.0)
    x32 = torch.randn(4, dim)
    ok32, margins = True, []
    for n in steps:
        err = float((flow32.integrate(x32, f, 0.0, 1.0, n) - (x32 + c)).abs().max())
        off_by_one = abs(c) / n          # what a wrong step count would cost
        margins.append(off_by_one / max(err, 1e-30))
        ok32 &= err < max(0.02 * off_by_one, 2e-5)
    check("float32: error is >=50x below an off-by-one in the step count", ok32,
          f"min margin {min(margins):.0f}x at n={steps[margins.index(min(margins))]}")

    # Partial intervals: the exact solution scales with (t1 - t0).
    check("integrate over a partial interval scales with (t1-t0)",
          torch.allclose(flow64.integrate(x64, f, 0.25, 0.75, 8), x64 + c * 0.5,
                         atol=1e-12))
    check("n_steps < 1 is refused",
          _raises(lambda: flow64.integrate(x64, f, 0.0, 1.0, 0), "n_steps"))

    # Sub-interval composition: one integrate over [0,1] must equal two chained
    # halves. This is what makes round_trip legitimate as a composition.
    check("integration composes across a split interval",
          torch.allclose(flow64.integrate(x64, f, 0.0, 1.0, 20),
                         flow64.integrate(flow64.integrate(x64, f, 0.0, 0.5, 10),
                                          f, 0.5, 1.0, 10), atol=1e-12))

    print("\n--- round trip on a constant field: exact ---")
    ok_rt, worst_rt = True, 0.0
    for n in (1, 5, 50):
        x1h, x0h, diag = flow64.round_trip(x64, f, n_steps=n)
        worst_rt = max(worst_rt, float((x1h - x64).abs().max()),
                       float((x0h - (x64 - c)).abs().max()))
        ok_rt &= worst_rt < 1e-12
    check("round_trip is exact for a constant field at any n_steps", ok_rt,
          f"worst |err| = {worst_rt:.3g}")
    check("round_trip reports n_steps and source_std",
          diag["n_steps"] == 50 and diag["source_std"] == 1.0)
    check("round_trip rel_l2 is ~0 for a constant field",
          diag["rel_l2_space"] < 1e-12, f"{diag['rel_l2_space']:.3g}")


def test_roundtrip_converges() -> None:
    print("\n--- round-trip residual must fall like 1/n_steps ---")
    # The diagnostic's entire claim: discretisation error falls with n_steps, model
    # inconsistency does not. If this trend fails on a smooth field, --flow_rt_steps
    # cannot separate the two and the round-trip arm reports an uninterpretable
    # number.
    torch.manual_seed(9)
    dim = 12
    net = FlowVelocityNet(dim=dim, n_families=N_FAMILIES, cond_dim=16,
                          hidden_dims=(32, 32), time_embed_dim=8).eval()
    # Scale the output layer up so the field is strong enough that Euler error is
    # well above float32 noise; at default init the residual is ~1e-4 and the trend
    # is measured against rounding.
    with torch.no_grad():
        net.net[-1].weight.mul_(6.0)
        net.net[-1].bias.mul_(6.0)
    flow = RectifiedFlow(net, source_std=1.0)
    x1 = torch.randn(16, dim)
    f = torch.randint(0, N_FAMILIES, (16,))

    errs = {}
    for n in (5, 20, 100):
        _, _, diag = flow.round_trip(x1, f, n_steps=n)
        errs[n] = diag["rel_l2_space"]
    print(f"      rel_l2: n=5 {errs[5]:.4g}   n=20 {errs[20]:.4g}   "
          f"n=100 {errs[100]:.4g}")
    check("residual falls from n=5 to n=20", errs[20] < errs[5],
          f"{errs[20]:.4g} < {errs[5]:.4g}")
    check("residual falls from n=20 to n=100", errs[100] < errs[20],
          f"{errs[100]:.4g} < {errs[20]:.4g}")
    # First-order solver: 20x the steps should buy well over one order of magnitude.
    check("n=100 residual is >=5x smaller than n=5 (first-order convergence)",
          errs[5] / max(errs[100], 1e-30) > 5.0,
          f"ratio {errs[5] / max(errs[100], 1e-30):.1f}x")

    check("round_trip warns rather than silently accepting CFG != 1.0",
          flow.round_trip(x1, f, n_steps=3, guidance_scale=2.0)[2]
          ["rel_l2_space"] >= 0.0)


def test_determinism() -> None:
    print("\n--- seeded sampling is reproducible ---")
    torch.manual_seed(5)
    net = FlowVelocityNet(dim=8, n_families=N_FAMILIES, cond_dim=16,
                          hidden_dims=(16, 16), time_embed_dim=8).eval()
    flow = RectifiedFlow(net, source_std=1.0)
    f = torch.tensor([0, 1, 2, 0])
    a = flow.sample(4, f, n_steps=7, generator=torch.Generator().manual_seed(1))
    b = flow.sample(4, f, n_steps=7, generator=torch.Generator().manual_seed(1))
    c = flow.sample(4, f, n_steps=7, generator=torch.Generator().manual_seed(2))
    check("same generator seed gives bit-identical samples", torch.equal(a, b))
    check("a different seed gives different samples", not torch.allclose(a, c))
    x0 = torch.randn(4, 8)
    check("explicit x0 bypasses the source draw",
          torch.equal(flow.sample(4, f, n_steps=7, x0=x0),
                      flow.integrate(x0, f, 0.0, 1.0, 7)))
    check("guidance_scale changes the trajectory",
          not torch.allclose(flow.sample(4, f, n_steps=7, x0=x0),
                             flow.sample(4, f, n_steps=7, x0=x0,
                                         guidance_scale=3.0)))


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

def test_code_space() -> None:
    print("\n--- CodeFlowSpace: exact, and per-family ---")
    cs = _code_stats(k=9, n_families=3)
    sp = build_space("codes", code_stats=cs)
    check("dim == k", sp.dim == 9, str(sp.dim))
    check("name == 'codes'", sp.name == "codes")

    codes = torch.from_numpy(cs.codes)
    fidx = torch.from_numpy(cs.family_idxs)
    back = sp.from_flow(sp.to_flow(codes, fidx), fidx)
    check("from_flow(to_flow(codes)) is the identity",
          torch.allclose(back, codes, atol=1e-4),
          f"max|diff|={float((back - codes).abs().max()):.3g}")

    # Normalized codes must actually be centred and scaled, or the flow's source
    # N(0,I) is transporting to somewhere with the wrong radius.
    xn = sp.to_flow(codes, fidx)
    per_family_mean = torch.stack([xn[fidx == f].mean() for f in range(3)])
    check("normalized codes are centred per family",
          float(per_family_mean.abs().max()) < 1e-3,
          f"max|mean|={float(per_family_mean.abs().max()):.3g}")
    per_family_rms = torch.stack([xn[fidx == f].pow(2).mean().sqrt()
                                  for f in range(3)])
    check("normalized codes have per-family RMS ~1",
          float((per_family_rms - 1.0).abs().max()) < 0.15,
          f"rms={[round(float(v), 3) for v in per_family_rms]}")

    # A family-indexing bug is the failure this catches: applying family 0's stats
    # to family 2's codes.
    wrong = sp.from_flow(sp.to_flow(codes, fidx), torch.zeros_like(fidx))
    check("using the wrong family label does NOT round-trip",
          not torch.allclose(wrong, codes, atol=1e-2))

    check("state() seals the code-stats fingerprint",
          sp.state()["code_stats_fingerprint"] == cs.fingerprint())
    check("the codes space needs code_stats",
          _raises(lambda: build_space("codes", code_stats=None), "code_stats"))
    check("an unknown space name is refused",
          _raises(lambda: build_space("wavelet", code_stats=cs), "unknown flow space"))


def test_latent_space() -> None:
    print("\n--- LatentFlowSpace: lossiness IS the VAE round trip ---")
    cs = _code_stats(k=9, n_families=3, seed=1)
    vae = _vae(code_dim=9, latent_dim=8, cs=cs)
    codes = torch.from_numpy(cs.codes)
    fidx = torch.from_numpy(cs.family_idxs)

    sp = build_space("latent", code_stats=cs, vae=vae, codes=codes,
                     family_idxs=fidx)
    check("dim == vae.latent_dim", sp.dim == 8, str(sp.dim))
    check("name == 'latent'", sp.name == "latent")

    # The key identity: the latent space adds NO lossiness of its own. Its round
    # trip is exactly the VAE's deterministic round trip, so the `flow_latent` arm's
    # cost decomposes into (VAE cost) + (ODE cost) and the `vae` arm already
    # measures the first term.
    back = sp.from_flow(sp.to_flow(codes, fidx), fidx)
    with torch.no_grad():
        vae_rt = vae(codes, fidx, sample=False)[0]
    check("from_flow(to_flow(codes)) == vae(codes, f, sample=False)[0]",
          torch.allclose(back, vae_rt, atol=1e-4),
          f"max|diff|={float((back - vae_rt).abs().max()):.3g}")

    xn = sp.to_flow(codes, fidx)
    check("standardized mu is centred per family",
          float(torch.stack([xn[fidx == f].mean()
                             for f in range(3)]).abs().max()) < 1e-4)
    check("standardized mu has per-family RMS ~1",
          float((torch.stack([xn[fidx == f].pow(2).mean().sqrt()
                              for f in range(3)]) - 1.0).abs().max()) < 0.2)

    check("mu_mean is (F, dim) and mu_std is (F, 1)",
          tuple(sp.mu_mean.shape) == (N_FAMILIES, 8) and
          tuple(sp.mu_std.shape) == (N_FAMILIES, 1))

    # Named, actionable refusal -- 'latent' without a VAE is the mistake a user
    # makes by running train_flow --space latent before train_stack has built one.
    check("the latent space refuses to build without a VAE",
          _raises(lambda: build_space("latent", code_stats=cs, vae=None),
                  "needs a StackVAE"))
    check("fitting a latent space refuses without codes",
          _raises(lambda: build_space("latent", code_stats=cs, vae=vae),
                  "needs codes and family_idxs"))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _prov() -> dict:
    return {"run_name": "t", "k": 9, "trust": "verified",
            "ensemble_fingerprint": "abc123"}


def test_save_load_codes() -> None:
    print("\n--- save_flow / load_flow: codes space ---")
    tmp = tempfile.mkdtemp()
    try:
        cs = _code_stats(k=9, n_families=3, seed=2)
        sp = build_space("codes", code_stats=cs)
        torch.manual_seed(7)
        net = FlowVelocityNet(dim=9, n_families=N_FAMILIES, cond_dim=16,
                              hidden_dims=(16, 16), time_embed_dim=8)
        flow = RectifiedFlow(net, source_std=1.0).eval()
        d = os.path.join(tmp, "flow_k9_codes")
        meta = save_flow(flow, sp, d, provenance=_prov(),
                         train_meta={"spectrum": {"spectrum_ev0_over_median": 1.01}},
                         defaults={"n_steps": 13, "guidance_scale": 1.0})
        check("flow_meta.json and flow_weights.pt both written",
              os.path.exists(os.path.join(d, "flow_meta.json")) and
              os.path.exists(os.path.join(d, "flow_weights.pt")))
        check("sealed layout_version is current",
              meta["layout_version"] == FLOW_LAYOUT_VERSION)

        fm = load_flow(d, code_stats=cs, device="cpu")
        check("defaults survive the round trip",
              fm.default_steps == 13 and fm.default_guidance == 1.0)
        check("trust survives", fm.trust == "verified")
        # The receipt that makes a flat-spectrum flow self-documenting forever.
        check("the training spectrum is sealed into the checkpoint",
              fm.spectrum.get("spectrum_ev0_over_median") == 1.01)

        f = torch.tensor([0, 1, 2, 0])
        a = flow.sample(4, f, n_steps=5, generator=torch.Generator().manual_seed(3))
        b = fm.flow.sample(4, f, n_steps=5,
                           generator=torch.Generator().manual_seed(3))
        check("reloaded flow reproduces seeded samples bit-identically",
              torch.equal(a, b))

        c1 = fm.sample_codes(f, n_steps=5,
                             generator=torch.Generator().manual_seed(3))
        check("sample_codes returns raw codes of width k",
              tuple(c1.shape) == (4, 9), str(tuple(c1.shape)))
        check("sample_codes == denormalize(sample)",
              torch.allclose(c1, cs.denormalize(a, f), atol=1e-5))

        codes = torch.from_numpy(cs.codes[:4])
        rt, diag = fm.round_trip_codes(codes, torch.from_numpy(cs.family_idxs[:4]))
        check("round_trip_codes returns codes and an ODE diagnostic",
              tuple(rt.shape) == (4, 9) and "rel_l2_space" in diag)
        check("round_trip_codes uses the sealed default n_steps",
              diag["n_steps"] == 13, str(diag["n_steps"]))

        # --- refusals -------------------------------------------------------
        check("a codes-space flow refuses to load without CodeStats",
              _raises(lambda: load_flow(d, code_stats=None, device="cpu"),
                      "needs the run's CodeStats"))

        # The failure this exists for: a vae_k9/ or flow_k9_codes/ left behind by a
        # run whose stage-3 statistics differed. Silently using it would put the
        # flow's samples on the wrong scale.
        other = _code_stats(k=9, n_families=3, seed=999)
        check("a codes-space flow refuses a different CodeStats fingerprint",
              _raises(lambda: load_flow(d, code_stats=other, device="cpu"),
                      "trained against code stats"))

        bumped = os.path.join(tmp, "bumped")
        shutil.copytree(d, bumped)
        p = os.path.join(bumped, "flow_meta.json")
        m = json.load(open(p))
        m["layout_version"] = 99
        json.dump(m, open(p, "w"))
        check("a bumped layout_version is refused",
              _raises(lambda: load_flow(bumped, code_stats=cs, device="cpu"),
                      "layout_version=99"))

        wrongclass = os.path.join(tmp, "wrongclass")
        shutil.copytree(d, wrongclass)
        p = os.path.join(wrongclass, "flow_meta.json")
        m = json.load(open(p))
        m["class"] = "SomethingElse"
        json.dump(m, open(p, "w"))
        check("a wrong class is refused",
              _raises(lambda: load_flow(wrongclass, code_stats=cs, device="cpu"),
                      "not 'RectifiedFlow'"))

        # k changed under the checkpoint: sealed dim != rebuilt space dim.
        mismatched = os.path.join(tmp, "mismatched")
        shutil.copytree(d, mismatched)
        p = os.path.join(mismatched, "flow_meta.json")
        m = json.load(open(p))
        m["dim"] = 7
        m["space_state"]["code_stats_fingerprint"] = cs.fingerprint()
        json.dump(m, open(p, "w"))
        check("a sealed dim that disagrees with the rebuilt space is refused",
              _raises(lambda: load_flow(mismatched, code_stats=cs, device="cpu"),
                      "sealed dim=7"))

        check("a directory with no flow_meta.json is refused",
              _raises(lambda: load_flow(os.path.join(tmp, "nope"),
                                        code_stats=cs, device="cpu"),
                      "no flow_meta.json"))
    finally:
        shutil.rmtree(tmp)


def test_save_load_latent() -> None:
    print("\n--- save_flow / load_flow: latent space, sealed mu statistics ---")
    tmp = tempfile.mkdtemp()
    try:
        cs = _code_stats(k=9, n_families=3, seed=3)
        vae = _vae(code_dim=9, latent_dim=8, cs=cs)
        codes = torch.from_numpy(cs.codes)
        fidx = torch.from_numpy(cs.family_idxs)
        sp = build_space("latent", code_stats=cs, vae=vae, codes=codes,
                         family_idxs=fidx)

        torch.manual_seed(8)
        net = FlowVelocityNet(dim=8, n_families=N_FAMILIES, cond_dim=16,
                              hidden_dims=(16, 16), time_embed_dim=8)
        flow = RectifiedFlow(net, source_std=1.0).eval()
        d = os.path.join(tmp, "flow_k9_latent")
        save_flow(flow, sp, d, provenance=_prov(),
                  defaults={"n_steps": 9, "guidance_scale": 1.0})

        fm = load_flow(d, code_stats=cs, vae=vae, device="cpu")
        check("reloaded latent space keeps dim = latent_dim", fm.space.dim == 8)
        # SEALED, not recomputed. Recomputing mu statistics at eval time from a
        # different code matrix would silently move the flow's target distribution
        # -- a bug that presents as "the flow got worse" and takes a day to find.
        check("mu_mean is restored from the checkpoint, not refitted",
              torch.allclose(fm.space.mu_mean, sp.mu_mean, atol=1e-6),
              f"max|diff|={float((fm.space.mu_mean - sp.mu_mean).abs().max()):.3g}")
        check("mu_std is restored from the checkpoint, not refitted",
              torch.allclose(fm.space.mu_std, sp.mu_std, atol=1e-6))

        # Prove it is the sealed value and not a coincidence: a space fitted on a
        # DIFFERENT code matrix has different statistics, and load must ignore them.
        other_cs = _code_stats(k=9, n_families=3, seed=1234)
        refit = build_space("latent", code_stats=other_cs, vae=vae,
                            codes=torch.from_numpy(other_cs.codes),
                            family_idxs=torch.from_numpy(other_cs.family_idxs))
        check("a refit on different codes really does differ (test is not vacuous)",
              not torch.allclose(refit.mu_mean, sp.mu_mean, atol=1e-3))
        fm2 = load_flow(d, code_stats=other_cs, vae=vae, device="cpu")
        check("load ignores the passed CodeStats for a latent flow's mu stats",
              torch.allclose(fm2.space.mu_mean, sp.mu_mean, atol=1e-6))

        f = torch.tensor([0, 1, 2, 0])
        a = flow.sample(4, f, n_steps=6, generator=torch.Generator().manual_seed(3))
        b = fm.flow.sample(4, f, n_steps=6,
                           generator=torch.Generator().manual_seed(3))
        check("reloaded latent flow reproduces seeded samples bit-identically",
              torch.equal(a, b))
        cd = fm.sample_codes(f, n_steps=6,
                             generator=torch.Generator().manual_seed(3))
        check("sample_codes decodes to raw codes of width k",
              tuple(cd.shape) == (4, 9), str(tuple(cd.shape)))
        check("sample_codes finite", bool(torch.isfinite(cd).all()))

        # Two DISTINCT guidance mechanisms: CFG on the velocity field and CFG on the
        # VAE decoder. Conflating them is the mistake --flow_guidance_scale vs
        # --guidance_scale exists to prevent.
        g_vae = fm.sample_codes(f, n_steps=6, vae_guidance_scale=3.0,
                                generator=torch.Generator().manual_seed(3))
        g_vel = fm.sample_codes(f, n_steps=6, guidance_scale=3.0,
                                generator=torch.Generator().manual_seed(3))
        check("VAE-decoder CFG changes the decoded codes",
              not torch.allclose(cd, g_vae, atol=1e-5))
        check("velocity CFG changes the decoded codes",
              not torch.allclose(cd, g_vel, atol=1e-5))
        check("the two CFG scales are not the same mechanism",
              not torch.allclose(g_vae, g_vel, atol=1e-5))

        check("a latent-space flow refuses to load without its VAE",
              _raises(lambda: load_flow(d, code_stats=cs, vae=None, device="cpu"),
                      "cannot be used without the StackVAE"))
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    torch.manual_seed(0)
    test_time_embedding()
    test_velocity_net()
    test_interpolant()
    test_euler_exactness()
    test_roundtrip_converges()
    test_determinism()
    test_code_space()
    test_latent_space()
    test_save_load_codes()
    test_save_load_latent()

    print("\n" + "=" * 64)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
