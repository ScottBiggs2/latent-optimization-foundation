"""
Step 4 validation: artifact_io fingerprints, CodeStats, StackVAE save/load, manifest.

Everything here runs on CPU with no downloads and no real models. The subject is the
PERSISTENCE contract, not numerics: a checkpoint must either reload into a state that
is bit-identical to what was saved, or refuse. Every refusal below corresponds to a
way this repo has already been bitten (RESEARCH_NOTES missteps 3, 10, 16, 17).

    python tests_step4_run_bundle.py
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

import artifact_io as aio
from run_bundle import CodeStats, rebuild_manifest, update_manifest_section
from vae import VAE_LAYOUT_VERSION, StackVAE

FAILS: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _raises(fn, needle: str) -> bool:
    """
    True when fn() raises and the message mentions `needle`.

    The needle matters as much as the raise: a refusal whose message does not name
    the offending field is a refusal the next person has to reverse-engineer.
    """
    try:
        fn()
        return False
    except Exception as exc:
        return needle in str(exc)


# ---------------------------------------------------------------------------
# artifact_io
# ---------------------------------------------------------------------------

def _ens_meta(**over) -> dict:
    meta = {
        "layout_version": 2, "arch_list": ["gpt2_medium", "pythia_410m"],
        "n_samples": 12, "noise_scale": 1e-2, "exclude_1d": True,
        "include_extra": True, "mode": "tiny", "seed": 7,
        "chunk_budget_bytes": 1 << 20, "source": "noise",
        "stacks": {
            "gpt2_medium": {"n_params": 1000, "n_layers": 2, "block_size": 400,
                            "extra_size": 200, "weight_std": 0.018},
            "pythia_410m": {"n_params": 900, "n_layers": 2, "block_size": 350,
                            "extra_size": 200, "weight_std": 0.020},
        },
    }
    meta.update(over)
    return meta


def test_artifact_io() -> None:
    print("\n--- artifact_io: fingerprints and atomic writes ---")

    check("fingerprint ignores key order",
          aio.fingerprint({"a": 1, "b": 2}) == aio.fingerprint({"b": 2, "a": 1}))
    check("fingerprint tracks values",
          aio.fingerprint({"a": 1}) != aio.fingerprint({"a": 2}))

    base = aio.ensemble_fingerprint(_ens_meta())

    # chunk_budget_bytes sets the chunk grid, and the noise is keyed to
    # (seed, chunk_idx) -- misstep 10. One seed with two grids is two different
    # ensembles, and the fingerprint has to say so.
    check("ensemble fp changes with chunk_budget_bytes",
          aio.ensemble_fingerprint(_ens_meta(chunk_budget_bytes=1 << 21)) != base)
    check("ensemble fp changes with include_extra",
          aio.ensemble_fingerprint(_ens_meta(include_extra=False)) != base)
    # The one that matters for Experiment 1: the moment EnsembleDataset gains
    # source="revisions", every noise-ensemble artifact stops matching automatically,
    # with no code change here.
    check("ensemble fp changes with source (noise -> revisions)",
          aio.ensemble_fingerprint(_ens_meta(source="revisions")) != base)
    check("ensemble fp changes when a stack's D changes",
          aio.ensemble_fingerprint(
              _ens_meta(stacks={**_ens_meta()["stacks"],
                                "gpt2_medium": {"n_params": 1001, "n_layers": 2,
                                                "block_size": 400,
                                                "extra_size": 201,
                                                "weight_std": 0.018}})) != base)
    check("ensemble fp ignores arch_list order",
          aio.ensemble_fingerprint(
              _ens_meta(arch_list=["pythia_410m", "gpt2_medium"])) == base)
    # A fingerprint that flips whenever a cosmetic field is added is one nobody can
    # keep green.
    check("ensemble fp ignores a cosmetic new field",
          aio.ensemble_fingerprint(_ens_meta(note="hello")) == base)
    # float32 reductions are not bit-reproducible across BLAS versions, so weight_std
    # is rounded before hashing.
    check("ensemble fp tolerates weight_std jitter below 1e-9",
          aio.ensemble_fingerprint(
              _ens_meta(stacks={**_ens_meta()["stacks"],
                                "gpt2_medium": {"n_params": 1000, "n_layers": 2,
                                                "block_size": 400, "extra_size": 200,
                                                "weight_std": 0.018 + 1e-12}})) == base)

    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "x.json")
        aio.atomic_write_json(p, {"a": 1})
        check("atomic_write_json round-trips", aio.read_json(p) == {"a": 1})
        check("atomic_write_json leaves no .tmp behind",
              not [f for f in os.listdir(tmp) if ".tmp" in f],
              f"dir={os.listdir(tmp)}")

        # Simulate a crash between writing the temp file and replacing it: the
        # previous file must still be intact and parseable. This is why the manifest
        # is allowed concurrent writers at all.
        with open(p + ".tmp.99999", "w") as f:
            f.write("{not json")
        check("a stranded temp file does not corrupt the real one",
              aio.read_json(p) == {"a": 1})

        check("read_json returns None for a missing file",
              aio.read_json(os.path.join(tmp, "nope.json")) is None)
        check("require_version accepts a match",
              aio.require_version({"layout_version": 3}, "layout_version", 3,
                                  "somewhere") is None)
        check("require_version refuses a mismatch and names the key",
              _raises(lambda: aio.require_version({"layout_version": 1},
                                                 "layout_version", 3, "somewhere"),
                      "layout_version"))
        prov = aio.provenance_block("myrun", 11, "abc", {"a": "def"})
        check("provenance_block records run/k/fingerprints and trust",
              prov["run_name"] == "myrun" and prov["k"] == 11
              and prov["ensemble_fingerprint"] == "abc"
              and prov["trust"] == "verified")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# CodeStats
# ---------------------------------------------------------------------------

def _synthetic_codes(n_per=10, k=7, families=(0, 1, 5), seed=3):
    """
    Deliberately different scale AND offset per family.

    Each family has its OWN PCA basis, so pooled statistics would be meaningless --
    a bug that pools them has to be visible here, not subtle.
    """
    rng = np.random.default_rng(seed)
    blocks, fidx = [], []
    for i, f in enumerate(families):
        scale = np.geomspace(3.0, 0.01, k).astype(np.float32) * (1.0 + i)
        blocks.append((rng.normal(0, 1, (n_per, k)) * scale).astype(np.float32)
                      + float(i))
        fidx.append(np.full(n_per, f, dtype=np.int64))
    return np.concatenate(blocks), np.concatenate(fidx)


def test_code_stats() -> None:
    print("\n--- CodeStats: the reduction and its persistence ---")
    codes, fidx = _synthetic_codes()
    cs = CodeStats.from_codes(codes, fidx, {"a": 0, "b": 1, "c": 5}, 6)

    check("means/stds have the per-family shapes",
          cs.means.shape == (6, 7) and cs.stds.shape == (6, 1),
          f"{cs.means.shape}, {cs.stds.shape}")

    # Brute force, transcribed from train_stack.stage_codes' original two lines:
    #   means[fi] = c.mean(dim=0)
    #   stds[fi]  = c.std(dim=0).clamp(min=1e-8).pow(2).mean().sqrt()
    # torch's std defaults to unbiased, hence ddof=1.
    for f in (0, 1, 5):
        b = codes[fidx == f]
        want_mean = b.mean(axis=0)
        per_dim = np.clip(b.std(axis=0, ddof=1), 1e-8, None)
        want_std = float(np.sqrt(np.mean(per_dim ** 2)))
        check(f"family {f}: mean matches the brute force",
              np.allclose(cs.means[f], want_mean, rtol=1e-6, atol=1e-8),
              f"max|Δ|={float(np.abs(cs.means[f] - want_mean).max()):.3g}")
        check(f"family {f}: scale matches the brute force",
              np.isclose(cs.stds[f, 0], want_std, rtol=1e-6),
              f"got {cs.stds[f, 0]:.6g}, want {want_std:.6g}")

    # A family with no rows must stay at the identity, or normalization would
    # silently shift codes for an arch that was not in this run.
    check("an unused family stays at mean 0 / scale 1",
          np.all(cs.means[2] == 0.0) and cs.stds[2, 0] == 1.0)

    # One scalar per family, NOT per dimension: per-dimension whitening would
    # equalize PC 0 and PC k-1 and spend capacity on directions carrying almost no
    # weight-space energy.
    check("the scale is one scalar per family, not per dimension",
          cs.stds.shape[1] == 1 and cs.STD_REDUCTION == "per_family_rms_over_dims")

    # normalize/denormalize must be exact inverses, or a codes-space flow and the
    # VAE would disagree on scale and their samples would not be comparable.
    ct = torch.from_numpy(codes).float()
    ft = torch.from_numpy(fidx).long()
    back = cs.denormalize(cs.normalize(ct, ft), ft)
    check("normalize/denormalize round-trips",
          torch.allclose(back, ct, rtol=1e-5, atol=1e-6),
          f"max|Δ|={float((back - ct).abs().max()):.3g}")

    check("from_codes refuses a row/label length mismatch",
          _raises(lambda: CodeStats.from_codes(codes, fidx[:-1], {}, 6),
                  "family_idxs"))
    check("from_codes refuses an out-of-range family_idx",
          _raises(lambda: CodeStats.from_codes(
              codes, np.full(len(fidx), 9, dtype=np.int64), {}, 6), "outside"))
    check("from_codes refuses k wider than the code matrix",
          _raises(lambda: CodeStats.from_codes(codes, fidx, {}, 6, k=99), "exceeds"))

    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "codes_k7")
        cs.save(d, provenance=aio.provenance_block("r", 7, "ENSFP"))
        back = CodeStats.load(d, expect_k=7)
        check("save/load is bit-exact on means", np.array_equal(back.means, cs.means))
        check("save/load is bit-exact on stds", np.array_equal(back.stds, cs.stds))
        check("save/load preserves the code matrix",
              back.codes is not None and np.array_equal(back.codes, cs.codes))
        check("save/load preserves the labels",
              np.array_equal(back.family_idxs, cs.family_idxs))
        check("fingerprint survives the round trip",
              back.fingerprint() == cs.fingerprint(), back.fingerprint())
        check("load refuses the wrong k",
              _raises(lambda: CodeStats.load(d, expect_k=5), "k=7"))

        # The npz and the meta must agree. If one is replaced without the other --
        # a partially re-run stage 3 -- the numbers would silently change while the
        # sealed fingerprint claimed otherwise.
        z = np.load(os.path.join(d, "code_stats.npz"))
        bad = z["means"].copy()
        bad[0, 0] += 1.0
        np.savez(os.path.join(d, "code_stats.npz"), means=bad, stds=z["stds"])
        check("load detects npz/meta disagreement",
              _raises(lambda: CodeStats.load(d), "hashes to"))

        mp = os.path.join(d, "code_stats_meta.json")
        meta = json.load(open(mp))
        meta["layout_version"] += 99
        json.dump(meta, open(mp, "w"))
        check("load refuses a bumped layout_version",
              _raises(lambda: CodeStats.load(d), "layout_version"))
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# StackVAE persistence
# ---------------------------------------------------------------------------

def test_stack_vae_save_load() -> None:
    print("\n--- StackVAE: save/load and every refusal ---")
    torch.manual_seed(0)
    codes, fidx = _synthetic_codes(n_per=8, k=6, families=(0, 1))
    cs = CodeStats.from_codes(codes, fidx, {"a": 0, "b": 1}, 6, k=6)

    model = StackVAE(code_dim=6, latent_dim=4, hidden_dim=32, cond_dim=8,
                     n_families=6, cond_dropout_p=0.15)
    model.set_code_norm(cs.torch_means(), cs.torch_stds())
    model.eval()

    ct = torch.from_numpy(codes[:3]).float()
    ft = torch.from_numpy(fidx[:3]).long()
    z = torch.randn(3, 4)
    with torch.no_grad():
        rec0 = model(ct, ft, sample=False)[0]
        cfg0 = model.decode_cfg(z, ft, guidance_scale=2.0)

    check("config() reports every ctor key",
          set(model.config()) == set(StackVAE.CTOR_KEYS)
          and model.config()["hidden_dim"] == 32)

    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "vae_k6")
        model.save(d,
                   provenance=aio.provenance_block("r", 6, "ENSFP",
                                                   code_stats_fp=cs.fingerprint()),
                   code_stats_fingerprint=cs.fingerprint(),
                   code_stats_dir="codes_k6")
        check("save writes the sealed pair",
              os.path.exists(os.path.join(d, "vae_meta.json"))
              and os.path.exists(os.path.join(d, "vae_weights.pt")))

        back = StackVAE.load(d, device="cpu", expect_code_dim=6, code_stats=cs)
        with torch.no_grad():
            rec1 = back(ct, ft, sample=False)[0]
            cfg1 = back.decode_cfg(z, ft, guidance_scale=2.0)
        # torch.equal, not allclose: a reload that is merely close means some tensor
        # took a dtype or device round trip it should not have.
        check("reload reproduces forward(sample=False) bit-exactly",
              torch.equal(rec0, rec1),
              f"max|Δ|={float((rec0 - rec1).abs().max()):.3g}")
        check("reload reproduces decode_cfg(s=2.0) bit-exactly",
              torch.equal(cfg0, cfg1))
        check("_code_mean survives the round trip",
              torch.equal(model._code_mean, back._code_mean))
        check("_code_std survives the round trip",
              torch.equal(model._code_std, back._code_std))
        check("loaded checkpoint is marked verified",
              back.loaded_meta["provenance"]["trust"] == "verified")

        check("load refuses a code_dim / rank mismatch",
              _raises(lambda: StackVAE.load(d, device="cpu", expect_code_dim=99),
                      "code_dim"))

        # The buffer check catches the real failure: a vae_k*/ left behind by a run
        # whose stage 3 produced different statistics.
        drift = CodeStats.from_codes(codes + 5.0, fidx, {"a": 0, "b": 1}, 6, k=6)
        check("load refuses a code-stats fingerprint mismatch",
              _raises(lambda: StackVAE.load(d, device="cpu", code_stats=drift),
                      "code_stats_fingerprint"))

        mp = os.path.join(d, "vae_meta.json")
        meta = json.load(open(mp))
        meta["layout_version"] = VAE_LAYOUT_VERSION + 99
        json.dump(meta, open(mp, "w"))
        check("load refuses a bumped layout_version",
              _raises(lambda: StackVAE.load(d, device="cpu"), "layout_version"))

        meta["layout_version"] = VAE_LAYOUT_VERSION
        meta["class"] = "ConditionedBlockVAE"
        json.dump(meta, open(mp, "w"))
        check("load refuses a legacy-pipeline class",
              _raises(lambda: StackVAE.load(d, device="cpu"),
                      "ConditionedBlockVAE"))
    finally:
        shutil.rmtree(tmp)


def test_legacy_vae_refusal() -> None:
    """
    A pre-provenance vae_k*/ has vae_config.json plus a bare vae_best.pt and records
    NOTHING about which ensemble or code statistics produced it. Misstep 17 is exactly
    this, so the default is refusal and the escape hatch has to be explicit.
    """
    print("\n--- StackVAE: the legacy escape hatch ---")
    torch.manual_seed(1)
    model = StackVAE(code_dim=5, latent_dim=3, hidden_dim=16, cond_dim=8,
                     n_families=6, cond_dropout_p=0.1)
    model.eval()

    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "vae_k5")
        os.makedirs(d)
        json.dump(model.config(), open(os.path.join(d, "vae_config.json"), "w"))
        torch.save(model.state_dict(), os.path.join(d, "vae_best.pt"))

        check("legacy dir is refused by default, naming the override",
              _raises(lambda: StackVAE.load(d, device="cpu"), "--allow_legacy_vae"))

        back = StackVAE.load(d, device="cpu", allow_legacy=True)
        check("allow_legacy adopts it", back.code_dim == 5 and back.latent_dim == 3)
        check("adopted checkpoint is stamped unverified-legacy",
              back.loaded_meta["provenance"]["trust"] == "unverified-legacy")
        # No silent migration: adopting must not manufacture provenance that was
        # never recorded.
        check("adopting does NOT write a vae_meta.json",
              not os.path.exists(os.path.join(d, "vae_meta.json")))

        ct = torch.randn(2, 5)
        ft = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            check("adopted weights still evaluate identically",
                  torch.equal(model(ct, ft, sample=False)[0],
                              back(ct, ft, sample=False)[0]))
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

def test_manifest() -> None:
    print("\n--- run_manifest: derived, idempotent, race-tolerant ---")
    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, "runs", "t")
        os.makedirs(os.path.join(root, "ensemble"))
        aio.atomic_write_json(
            os.path.join(root, "ensemble", "ensemble_meta.json"), _ens_meta())

        for arch in ("gpt2_medium", "pythia_410m"):
            d = os.path.join(root, "pca", arch)
            os.makedirs(d)
            aio.atomic_write_json(os.path.join(d, "gram_pca_meta.json"), {
                "layout_version": 1, "arch": arch, "n_components": 11,
                "n_samples": 12,
                "n_params": _ens_meta()["stacks"][arch]["n_params"],
                "rank_rtol": 1e-7, "total_variance": 1.0})

        man = rebuild_manifest(root)
        check("manifest finds the ensemble and its fingerprint",
              man["ensemble"]["fingerprint"] == aio.ensemble_fingerprint(_ens_meta()))
        check("manifest finds both PCA fits",
              set(man["pca"]["per_arch"]) == {"gpt2_medium", "pythia_410m"})
        # Absent sections are normal: --arms pca_only runs before any VAE exists.
        check("manifest tolerates missing codes/vae/flow sections",
              man["codes"] == {} and man["vae"] == {} and man["flow"] == {})

        again = rebuild_manifest(root)
        a, b = dict(man), dict(again)
        a.pop("rebuilt")
        b.pop("rebuilt")
        check("rebuild_manifest is idempotent", a == b)

        # One Slurm job runs train_flow once per (rank, space) as separate processes,
        # so sequential section updates must not clobber each other.
        update_manifest_section(root, "flow", "11",
                                {"codes": {"dir": "flow_k11_codes"}})
        update_manifest_section(root, "flow", "6",
                                {"codes": {"dir": "flow_k6_codes"}})
        after = aio.read_json(os.path.join(root, "run_manifest.json"))
        check("two sequential section updates do not lose the first",
              set(after["flow"]) == {"11", "6"}, f"{list(after['flow'])}")

        # And a rebuild re-derives from disk, so a LOST update costs nothing -- which
        # is the whole reason concurrent writers are tolerable here.
        rebuilt = rebuild_manifest(root)
        check("a rebuild drops index entries with no artifact on disk",
              rebuilt["flow"] == {})
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    test_artifact_io()
    test_code_stats()
    test_stack_vae_save_load()
    test_legacy_vae_refusal()
    test_manifest()

    print("\n" + "=" * 64)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
