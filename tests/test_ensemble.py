"""
EnsembleDataset + DualGramPCA.

The DualGramPCA checks use a small stand-in ensemble whose full data matrix fits in
memory, so every result can be compared against an explicit brute-force PCA. That is
the whole point: DualGramPCA never builds the (k, D) basis, so its correctness has to
be pinned against a reference that does.

    python tests/test_ensemble.py            # math only, no model downloads
    python tests/test_ensemble.py --tiny     # also exercise real EnsembleDataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from llmzoo.pca.gram import DualGramPCA

FAILS: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# A minimal stand-in with EnsembleDataset's loader interface
# ---------------------------------------------------------------------------

@dataclass
class _Stack:
    n_params: int
    n_layers: int = 2
    block_size: int = 0
    family_idx: int = 0
    weight_std: float = 1.0
    arch: str = "fake"


class _FakeEnsemble:
    """Holds an explicit (N, D) matrix and serves it through the chunk interface."""

    def __init__(self, X: np.ndarray, chunk: int, tmpdir: str, arch: str = "fake"):
        self.X = X.astype(np.float32)
        self.n_samples = X.shape[0]
        self._chunk = chunk
        self.arch_list = [arch]
        st = _Stack(n_params=X.shape[1], arch=arch)
        st.block_size = X.shape[1] // st.n_layers
        self.stacks = {arch: st}
        self._w0_path = os.path.join(tmpdir, f"{arch}_w0.npy")
        np.save(self._w0_path, self.X[0])

    def chunk_size(self, arch: str) -> int:
        return self._chunk

    def chunk_bounds(self, arch: str) -> List[Tuple[int, int]]:
        D = self.stacks[arch].n_params
        return [(r0, min(r0 + self._chunk, D)) for r0 in range(0, D, self._chunk)]

    def w0(self, arch: str) -> np.ndarray:
        return np.load(self._w0_path, mmap_mode="r")

    def load_chunk(self, arch, chunk_idx, device="cpu", w0_mm=None) -> torch.Tensor:
        r0, r1 = self.chunk_bounds(arch)[chunk_idx]
        return torch.from_numpy(np.ascontiguousarray(self.X[:, r0:r1])).to(device)


def brute_force(X: np.ndarray, k: int):
    """Explicit reference PCA. Returns (mean, components (k,D), codes (N,k))."""
    mean = X.mean(axis=0)
    Xc = X - mean
    C = Xc @ Xc.T
    ev, U = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    ev, U = ev[order], U[:, order]
    Uk, Sk = U[:, :k], ev[:k]
    comps = (Uk.T @ Xc) / np.sqrt(np.clip(Sk, 1e-300, None))[:, None]
    codes = Xc @ comps.T
    return mean, comps, codes


# ---------------------------------------------------------------------------
# DualGramPCA math
# ---------------------------------------------------------------------------

def test_gram_math(label: str, X: np.ndarray, k: int, chunk: int) -> None:
    print(f"\n--- {label}  (N={X.shape[0]}, D={X.shape[1]}, k={k}, chunk={chunk}) ---")
    N, D = X.shape
    tmp = tempfile.mkdtemp()
    try:
        ds = _FakeEnsemble(X, chunk, tmp)
        pca = DualGramPCA(n_components=k, device="cpu").fit(ds, "fake")
        kk = pca.n_components

        ref_mean, ref_comps, ref_codes = brute_force(X, kk)

        check(f"{label}: mean matches brute force",
              np.allclose(pca.mean_, ref_mean, rtol=1e-4, atol=1e-6),
              f"max|Δ|={np.max(np.abs(pca.mean_ - ref_mean)):.3g}")

        # Codes are defined up to a per-component sign.
        got = pca.codes(kk)
        cc = np.array([
            abs(np.dot(got[:, j], ref_codes[:, j]) /
                (np.linalg.norm(got[:, j]) * np.linalg.norm(ref_codes[:, j]) + 1e-30))
            for j in range(kk)
        ])
        check(f"{label}: codes == U*sqrt(S) matches explicit Xc @ V",
              np.all(cc > 0.9999), f"min|cos|={cc.min():.8f}")
        # Compare relative to the code SCALE. Codes are O(sqrt(S)), which is large
        # for raw weight data, so an absolute tolerance is meaningless here.
        scale = float(np.abs(ref_codes).max()) + 1e-30
        mag_err = float(np.max(np.abs(np.abs(got) - np.abs(ref_codes))) / scale)
        check(f"{label}: code magnitudes match", mag_err < 1e-4,
              f"max|Δ|/scale={mag_err:.3g} (scale={scale:.4g})")

        # Reconstruction of an ensemble member.
        i = min(3, N - 1)
        recon = pca.inverse_transform(pca.codes(kk)[i], ds, "fake", k=kk)
        ref_recon = ref_codes[i] @ ref_comps + ref_mean
        check(f"{label}: inverse_transform matches explicit basis reconstruction",
              np.allclose(recon, ref_recon, rtol=1e-3, atol=1e-5),
              f"max|Δ|={np.max(np.abs(recon - ref_recon)):.3g}")

        rel = np.linalg.norm(recon - X[i]) / (np.linalg.norm(X[i]) + 1e-30)
        if kk >= N - 1:
            check(f"{label}: exact round-trip at k=N-1", rel < 1e-5,
                  f"relative error={rel:.3g}")
        else:
            check(f"{label}: round-trip finite at k<N-1", np.isfinite(rel),
                  f"relative error={rel:.3g}")

        # Sample 0 is the real model, so its reconstruction is the one that matters.
        recon0 = pca.inverse_transform(pca.codes(kk)[0], ds, "fake", k=kk)
        rel0 = np.linalg.norm(recon0 - X[0]) / (np.linalg.norm(X[0]) + 1e-30)
        if kk >= N - 1:
            check(f"{label}: sample 0 (real model) exact at k=N-1", rel0 < 1e-5,
                  f"relative error={rel0:.3g}")

        # transform_vector on an ensemble member must recover that member's code.
        z0 = pca.transform_vector(X[0], ds, "fake", k=kk)
        check(f"{label}: transform_vector(X[0]) == codes[0]",
              np.allclose(z0, pca.codes(kk)[0], rtol=1e-2, atol=1e-3 * np.abs(pca.codes(kk)[0]).max()),
              f"max|Δ|={np.max(np.abs(z0 - pca.codes(kk)[0])):.3g}")

        # Chunk size must not change any result.
        ds2 = _FakeEnsemble(X, max(1, chunk // 3), tmp, arch="fake")
        pca2 = DualGramPCA(n_components=k, device="cpu").fit(ds2, "fake")
        check(f"{label}: result independent of chunk size",
              pca2.n_components == kk and
              np.allclose(pca2.gram_evals_, pca.gram_evals_, rtol=1e-6),
              f"k {pca2.n_components} vs {kk}")
    finally:
        shutil.rmtree(tmp)


def test_save_load() -> None:
    print("\n--- DualGramPCA save/load ---")
    rng = np.random.default_rng(5)
    N, D, k = 14, 900, 13
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    tmp = tempfile.mkdtemp()
    try:
        ds = _FakeEnsemble(X, 250, tmp)
        pca = DualGramPCA(n_components=k, device="cpu").fit(ds, "fake")
        sub = os.path.join(tmp, "saved")
        pca.save(sub)

        loaded = DualGramPCA.load(sub, device="cpu")
        check("k preserved", loaded.n_components == pca.n_components,
              f"{loaded.n_components} vs {pca.n_components}")
        check("codes preserved", np.array_equal(loaded.codes_, pca.codes_))
        check("eigenbasis preserved", np.allclose(loaded.gram_evecs_, pca.gram_evecs_))
        check("mean preserved", np.allclose(np.asarray(loaded.mean_), pca.mean_))
        r1 = pca.inverse_transform(pca.codes()[2], ds, "fake")
        r2 = loaded.inverse_transform(loaded.codes()[2], ds, "fake")
        check("reconstruction identical after reload", np.allclose(r1, r2, atol=1e-6),
              f"max|Δ|={np.max(np.abs(r1 - r2)):.3g}")

        # The whole point: no (k, D) basis on disk.
        check("no components matrix written",
              not os.path.exists(os.path.join(sub, "components.npy")))
    finally:
        shutil.rmtree(tmp)


def test_noise_ensemble_analytic() -> None:
    """The plan's four analytic predictions, on a w0 + noise ensemble."""
    print("\n--- noise ensemble: analytic predictions ---")
    rng = np.random.default_rng(0)
    N, D, s = 40, 30000, 1e-2
    w0 = rng.normal(0, 0.02, D).astype(np.float32)
    sigma = float(w0.std())
    X = np.empty((N, D), dtype=np.float32)
    X[0] = w0
    X[1:] = w0 + s * sigma * rng.normal(0, 1, (N - 1, D)).astype(np.float32)

    tmp = tempfile.mkdtemp()
    try:
        ds = _FakeEnsemble(X, 8000, tmp)
        pca = DualGramPCA(n_components=N - 1, device="cpu").fit(ds, "fake")
        k = pca.n_components
        check("(0) full rank retained", k == N - 1, f"k={k} of {N - 1}")

        # Flatness holds over the WELL-CONDITIONED directions. Sample 0 carries no
        # noise, so only N-1 members contribute noise and centering removes one dof:
        # ~N-2 directions are flat and the last is much weaker. See the
        # "Consequences of making sample 0 noise-free" note in dual_gram_pca.
        ev = pca.explained_variance_
        ratio_wc = float(ev[0] / ev[k - 2])
        check("(1) spectrum flat over well-conditioned directions", ratio_wc < 3.0,
              f"ev[0]/ev[{k - 2}]={ratio_wc:.3f}")
        check("(1b) final direction is the weak one (sample-0 artefact)",
              float(ev[0] / ev[-1]) > ratio_wc,
              f"ev[0]/ev[{k - 1}]={float(ev[0] / ev[-1]):.3f} vs "
              f"ev[0]/ev[{k - 2}]={ratio_wc:.3f}")

        evr = float(np.sum(pca.explained_variance_ratio_))
        check("(2a) EVR ~1.00 at k=N-1", abs(evr - 1.0) < 0.02, f"EVR={evr:.4f}")
        half = N // 2
        evr_h = float(np.sum(pca.explained_variance_ratio_[:half]))
        check("(2b) EVR ~0.50 at k=N/2", abs(evr_h - 0.5) < 0.08, f"EVR={evr_h:.4f}")

        r_full = pca.inverse_transform(pca.codes()[0], ds, "fake", k=k)
        rel = np.linalg.norm(r_full - w0) / np.linalg.norm(w0)
        check("(3) real model exact at k=N-1", rel < 1e-5, f"relative error={rel:.3g}")

        # (4) applies to a TYPICAL member. s*sigma/sqrt(2) is the residual left when
        # half the (isotropic) variance is discarded.
        j = 5
        r_typ = pca.inverse_transform(pca.codes()[j][:half], ds, "fake", k=half)
        resid_typ = float(np.std(r_typ - X[j]))
        predicted = s * sigma / np.sqrt(2.0)
        check("(4) k=N/2 residual std ~ s*sigma/sqrt(2) for a typical member",
              0.5 * predicted < resid_typ < 1.8 * predicted,
              f"measured={resid_typ:.5g}, predicted={predicted:.5g}")

        # Sample 0 sits ~sqrt(N) closer to the mean, so its residual is smaller by
        # about that factor. Checking it separately keeps the two straight.
        r_half0 = pca.inverse_transform(pca.codes()[0][:half], ds, "fake", k=half)
        resid0 = float(np.std(r_half0 - w0))
        predicted0 = s * sigma / np.sqrt(2.0 * N)
        check("(4b) sample 0 residual ~ s*sigma/sqrt(2N) (nearer the mean)",
              0.4 * predicted0 < resid0 < 2.5 * predicted0,
              f"measured={resid0:.5g}, predicted={predicted0:.5g}")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# The real EnsembleDataset (needs transformers)
# ---------------------------------------------------------------------------

def test_real_ensemble_dataset() -> None:
    print("\n--- EnsembleDataset (tiny mode) ---")
    from llmzoo.data.ensemble import EnsembleDataset

    tmp = tempfile.mkdtemp()
    try:
        ds = EnsembleDataset(
            arch_list=["gpt2_medium", "smollm2_360m", "pythia_410m"],
            n_samples=8, noise_scale=1e-2, exclude_1d=True,
            mode="tiny", artifact_dir=tmp, seed=7,
            chunk_budget_bytes=1 << 20,
        )
        print(ds.summary())

        for arch in ds.arch_list:
            st = ds.stacks[arch]
            check(f"{arch}: D == extra + L * block_size",
                  st.n_params == st.extra_size + st.n_layers * st.block_size,
                  f"{st.n_params} vs {st.extra_size}+{st.n_layers}*{st.block_size}")
            check(f"{arch}: extra segment is non-empty",
                  st.extra_size > 0 and len(st.extra_schema) > 0,
                  f"extra_size={st.extra_size}, {len(st.extra_schema)} entries")

            bounds = ds.chunk_bounds(arch)
            covered = sum(r1 - r0 for r0, r1 in bounds)
            contiguous = all(bounds[i][1] == bounds[i + 1][0]
                             for i in range(len(bounds) - 1))
            check(f"{arch}: chunk grid tiles D exactly",
                  covered == st.n_params and contiguous and bounds[0][0] == 0,
                  f"covered {covered} of {st.n_params} in {len(bounds)} chunks")

            # Determinism across passes -- the property the whole no-disk scheme
            # rests on. If this fails, the Gram and mean passes disagree.
            a = ds.load_chunk(arch, 0)
            b = ds.load_chunk(arch, 0)
            check(f"{arch}: load_chunk deterministic across calls",
                  torch.equal(a, b))

            # Sample 0 must be the real model, bit-exactly.
            w0 = ds.w0(arch)
            r0, r1 = bounds[0]
            check(f"{arch}: sample 0 is w0 exactly",
                  np.array_equal(a[0].cpu().numpy(), np.asarray(w0[r0:r1])))

            # Noise magnitude should match the requested scale.
            spread = float((a[1:] - a[0]).std())
            want = 1e-2 * st.weight_std
            check(f"{arch}: noise std matches noise_scale * weight_std",
                  0.7 * want < spread < 1.4 * want,
                  f"measured={spread:.4g}, want={want:.4g}")

            # Block slicing must line up with the concatenation order, which now
            # starts after the extra segment.
            lo, hi = st.block_slice(1)
            check(f"{arch}: block_slice(1) is offset past the extra segment",
                  (lo, hi) == (st.extra_size + st.block_size,
                               st.extra_size + 2 * st.block_size),
                  f"({lo}, {hi}) with extra_size={st.extra_size}")
            check(f"{arch}: extra_slice leads the stack",
                  st.extra_slice() == (0, st.extra_size))
            check(f"{arch}: last block ends exactly at D",
                  st.block_slice(st.n_layers - 1)[1] == st.n_params)

        # Cache-invalidation: reusing the dir with different ensemble params must
        # refuse rather than silently serve a different ensemble.
        raised = False
        try:
            EnsembleDataset(arch_list=["gpt2_medium"], n_samples=8,
                            noise_scale=5e-2, exclude_1d=True, mode="tiny",
                            artifact_dir=tmp, seed=7)
        except RuntimeError as exc:
            raised = "noise_scale" in str(exc)
        check("changing noise_scale invalidates the cached ensemble", raised)

        # include_extra changes D, so it must invalidate too -- otherwise a
        # decoder-only basis gets silently reused against a wider ensemble.
        raised = False
        try:
            EnsembleDataset(arch_list=["gpt2_medium"], n_samples=8,
                            noise_scale=1e-2, exclude_1d=True, mode="tiny",
                            include_extra=False, artifact_dir=tmp, seed=7)
        except RuntimeError as exc:
            raised = "include_extra" in str(exc)
        check("changing include_extra invalidates the cached ensemble", raised)
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# The extra segment: embeddings / final norm / untied LM head
# ---------------------------------------------------------------------------

def test_extra_segment() -> None:
    print("\n--- extra segment (embeddings / final norm / LM head) ---")
    from llmzoo.models.registry import build_tiny_model, get_arch_config
    from llmzoo.models.weight_extractor import (
        build_stack_spec, extract_extra_flat, read_stack_from_model,
        write_stack_to_model,
    )

    for arch in ("gpt2_medium", "smollm2_360m", "pythia_410m"):
        model = build_tiny_model(arch)
        prefix = get_arch_config(arch)["layers_attr"] + "."

        w0, spec = build_stack_spec(model, arch, exclude_1d=True,
                                    include_extra=True)

        # The strongest available check that tying is neither double-counted nor
        # dropped: D must equal the model's own count of UNIQUE >=2-D parameters.
        # named_parameters() deduplicates shared tensors, so a tied lm_head/wte
        # pair contributes once. Double-counting would overshoot, omitting the
        # extra segment would undershoot.
        unique = sum(p.numel() for _, p in model.named_parameters()
                     if p.dim() >= 2)
        check(f"{arch}: D equals the model's unique >=2-D parameter count",
              spec.n_params == unique, f"{spec.n_params:,} vs {unique:,}")

        names = [e.name for e in spec.extra_schema]
        check(f"{arch}: extra schema names are unique",
              len(names) == len(set(names)), f"{len(names)} entries")
        check(f"{arch}: no extra entry lives inside the block subtree",
              not any(n.startswith(prefix) for n in names))
        check(f"{arch}: exclude_1d left no 1-D entry in the extra schema",
              all(len(e.shape) >= 2 for e in spec.extra_schema))

        # extract -> write -> re-extract must be bit-identical, which is what makes
        # restore() trustworthy between evaluation arms.
        rt = read_stack_from_model(model, arch, spec)
        check(f"{arch}: read_stack_from_model reproduces build_stack_spec",
              np.array_equal(rt, w0))

        perturbed = (w0 + 0.1).astype(np.float32)
        write_stack_to_model(perturbed, model, arch, spec)
        after = read_stack_from_model(model, arch, spec)
        check(f"{arch}: write -> read round trip is bit-identical",
              np.array_equal(after, perturbed),
              f"max|Δ|={float(np.abs(after - perturbed).max()):.3g}")

        # And the extra segment specifically must have been written, not skipped.
        e_after, _ = extract_extra_flat(model, arch, exclude_1d=True)
        check(f"{arch}: the extra segment was actually written back",
              np.array_equal(e_after, perturbed[:spec.extra_size]))

        write_stack_to_model(w0, model, arch, spec)
        check(f"{arch}: restoring w0 undoes the perturbation",
              np.array_equal(read_stack_from_model(model, arch, spec), w0))

        # include_extra=False must reproduce the layout_version 1 geometry exactly.
        wb, sb = build_stack_spec(model, arch, exclude_1d=True,
                                  include_extra=False)
        check(f"{arch}: include_extra=False gives extra_size 0",
              sb.extra_size == 0 and sb.block_slice(0) == (0, sb.block_size))
        check(f"{arch}: the two layouts agree on the block region",
              np.array_equal(wb, w0[spec.extra_size:]),
              f"{wb.size:,} vs {w0.size - spec.extra_size:,}")
        check(f"{arch}: blocks-only D == L * block_size",
              sb.n_params == sb.n_layers * sb.block_size)

        del model


def test_tied_lm_head() -> None:
    """
    build_tiny_model forces tie_word_embeddings=False, so the tied path is never
    exercised by the other tests -- yet gpt2_medium and smollm2_360m are BOTH tied
    in their real checkpoints. Build a tied model explicitly.
    """
    print("\n--- tied LM head ---")
    from transformers import AutoConfig, AutoModelForCausalLM
    from llmzoo.models.registry import get_arch_config
    from llmzoo.models.weight_extractor import build_stack_spec

    arch = "gpt2_medium"
    cfg = get_arch_config(arch)
    tiny = cfg["tiny_config"].copy()
    tiny["model_type"] = cfg["hf_model_type"]
    hf_cfg = AutoConfig.for_model(**tiny)
    hf_cfg.tie_word_embeddings = True
    model = AutoModelForCausalLM.from_config(hf_cfg)
    model.eval()

    tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    check("the test model really is tied", tied)

    w0, spec = build_stack_spec(model, arch, exclude_1d=True, include_extra=True)
    names = [e.name for e in spec.extra_schema]

    check("wte is in the extra schema", "transformer.wte.weight" in names)
    # named_parameters() yields the shared tensor once, under whichever name it
    # reaches first. Either name is fine; BOTH would mean double-counting.
    check("the tied tensor appears exactly once",
          ("transformer.wte.weight" in names) != ("lm_head.weight" in names)
          or names.count("transformer.wte.weight") + names.count("lm_head.weight") == 1,
          f"names={names}")

    unique = sum(p.numel() for _, p in model.named_parameters() if p.dim() >= 2)
    check("D equals the unique parameter count under tying",
          spec.n_params == unique, f"{spec.n_params:,} vs {unique:,}")
    check("D counts the embedding once, not twice",
          spec.n_params < unique + tiny["vocab_size"] * tiny["n_embd"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true",
                    help="also run the real EnsembleDataset (imports transformers)")
    args = ap.parse_args()

    rng = np.random.default_rng(42)

    # Chunk count > 1 in every case, so the streaming paths are actually exercised.
    X = rng.normal(0, 1, (16, 2000)).astype(np.float32)
    test_gram_math("generic, k=N-1", X, 15, 512)
    test_gram_math("generic, k=N/2", X, 8, 512)
    test_gram_math("generic, D not a multiple of chunk", X, 15, 700)

    test_noise_ensemble_analytic()
    test_save_load()

    if args.tiny:
        test_extra_segment()
        test_tied_lm_head()
        test_real_ensemble_dataset()
    else:
        print("\n(skipping real EnsembleDataset — pass --tiny to include it)")

    print("\n" + "=" * 64)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
