"""
Is the flow's contraction UNDERFITTING or finite-sample regression?

The question this settles
------------------------
`flow_codes` emits codes at 0.25x the correct RMS at k=99 and 1.26x at k=6
(RESEARCH_NOTES misstep 19). Those two points came from different runs, which also
differed in net width and training-row count, so "collapse grows with k" was
confounded. Two hypotheses:

  UNDERFITTING          -> more capacity or more epochs recovers the spread.
  FINITE-SAMPLE regress -> it does not, and more capacity may make it WORSE
                           (the velocity target u = x1 - x0 is irreducibly noisy at
                           M = 300 rows, so a bigger net fits the noise, not the map).

Both are fixable, but by opposite actions, so guessing is expensive.

Design: hold k, the code matrix, the seed and the space FIXED; sweep only net width
and epochs. Trains in-process and writes NO artifacts, so the sealed flows of the run
being studied are untouched.

    python scripts/diag_flow_capacity.py --run_name emb3 --k 99
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import List, Tuple

import torch

from llmzoo.gen.flow import FlowVelocityNet, RectifiedFlow, build_space
from llmzoo.models.registry import N_FAMILIES
from llmzoo.artifacts.bundle import load_run


def train_one(codes: torch.Tensor, fidx: torch.Tensor, space, dim: int,
              hidden: Tuple[int, ...], epochs: int, batch: int, lr: float,
              seed: int, device) -> Tuple[RectifiedFlow, float]:
    torch.manual_seed(seed)
    net = FlowVelocityNet(dim=dim, n_families=N_FAMILIES, cond_dim=64,
                          hidden_dims=hidden, time_embed_dim=64,
                          cond_dropout_p=0.15).to(device)
    flow = RectifiedFlow(net, source_std=1.0, path_noise=0.0).to(device)
    # Every detail below is copied from train_flow.py's loop on purpose: gradient
    # clipping at 1.0, eta_min=1e-6, batch-mean loss, and BEST-loss weights rather
    # than final-epoch weights. A sweep that trains differently from the real
    # trainer answers a question about the sweep, not about the sealed flows.
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                      eta_min=1e-6)
    x1 = space.to_flow(codes, fidx).detach()
    n = x1.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = float("inf"), None
    for _ep in range(epochs):
        flow.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss, _ = flow.loss(x1[idx], fidx[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        tot /= max(nb, 1)
        sched.step()
        if tot < best:
            best = tot
            best_state = copy.deepcopy(flow.state_dict())
    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()
    return flow, best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--run_name", default="emb3")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--space", default="codes", choices=["codes", "latent"])
    p.add_argument("--widths", nargs="+", default=["64,128,64", "256,512,256",
                                                   "512,1024,512"])
    p.add_argument("--epochs", nargs="+", type=int, default=[500, 2000, 8000])
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    want = ("codes", "vae") if args.space == "latent" else ("codes",)
    bundle = load_run(run_root, args.k, device=str(device), want=want)
    cs = bundle.code_stats
    codes = torch.from_numpy(cs.codes).float().to(device)
    fidx = torch.from_numpy(cs.family_idxs).long().to(device)
    space = build_space(args.space, code_stats=cs, vae=bundle.vae,
                        codes=codes, family_idxs=fidx)

    real = space.to_flow(codes, fidx).detach()
    real_rms = float(real.pow(2).mean().sqrt())

    print("=" * 84)
    print(f"Flow capacity sweep — run={args.run_name} k={args.k} "
          f"space={args.space} dim={space.dim}")
    print(f"rows M = {codes.shape[0]}   real target RMS = {real_rms:.4f}")
    print("Everything except net width and epochs is held fixed: same codes, same")
    print("seed, same space, same lr/batch. rms_ratio 1.0 is correct.")
    print("=" * 84)
    print(f"{'width':16s} {'params':>10s} {'epochs':>7s} {'loss':>10s} "
          f"{'rms_ratio':>10s}   verdict")

    rows: List[dict] = []
    for wstr in args.widths:
        hidden = tuple(int(v) for v in wstr.split(","))
        for ep in args.epochs:
            flow, loss = train_one(codes, fidx, space, space.dim, hidden, ep,
                                   args.batch_size, args.lr, args.seed, device)
            nparam = sum(q.numel() for q in flow.net.parameters())
            g = torch.Generator(device="cpu").manual_seed(0)
            fv = fidx[:1].expand(args.n_eval).contiguous()
            with torch.no_grad():
                x0 = (torch.randn(args.n_eval, space.dim, generator=g)
                      .to(device) * flow.source_std)
                x = flow.sample(args.n_eval, fv, n_steps=50, x0=x0)
            ratio = float(x.pow(2).mean().sqrt()) / max(real_rms, 1e-12)
            verdict = ("COLLAPSED" if ratio < 0.5 else "shrunken" if ratio < 0.8
                       else "over-disp" if ratio > 1.25 else "ok")
            print(f"{wstr:16s} {nparam:10,d} {ep:7d} {loss:10.5f} "
                  f"{ratio:10.3f}   {verdict}")
            rows.append({"width": wstr, "hidden_dims": list(hidden),
                         "n_params": nparam, "epochs": ep, "final_loss": loss,
                         "rms_ratio": ratio, "verdict": verdict})

    print("\n" + "=" * 84)
    best = max(rows, key=lambda r: r["rms_ratio"])
    by_width = {}
    for r in rows:
        by_width.setdefault(r["width"], []).append(r["rms_ratio"])
    print("HOW TO READ THIS.")
    print("  If rms_ratio rises toward 1.0 with width or epochs -> UNDERFITTING;")
    print("    the fix is capacity/training.")
    print("  If it is flat or FALLS as capacity grows -> finite-sample regression;")
    print("    the fix is more rows or a narrower code space, not a bigger net.")
    print(f"\n  best: width={best['width']} epochs={best['epochs']} "
          f"rms_ratio={best['rms_ratio']:.3f}")
    for w, rs in by_width.items():
        print(f"  width {w:16s} rms_ratio range {min(rs):.3f} … {max(rs):.3f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"run": args.run_name, "k": args.k, "space": args.space,
                       "dim": space.dim, "n_rows": int(codes.shape[0]),
                       "real_rms": real_rms, "sweep": rows}, fh, indent=2)
        print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
