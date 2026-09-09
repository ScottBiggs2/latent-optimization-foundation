#!/usr/bin/env python
"""
tests/test_zoo.py -- the Phase 0.2 / 0.3 changes and the zoo plumbing.

    python tests/test_zoo.py            # needs numpy + torch
    python tests/test_zoo.py --tiny     # skips anything that builds a real GPT-2

Bare script, not pytest, to match the other modules in this directory.

The load-bearing check is `noise_source_fingerprint_unchanged`: Phase 0.3 was only
allowed to land if source="noise" still fingerprints bit-for-bit identically to a
pre-refactor ensemble, because every existing artifact is keyed on that.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

PASS = FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def section(t):
    print(f"\n--- {t} ---")


# ---------------------------------------------------------------------------

def test_cond_slots_decoupled():
    section("Phase 0.2 -- conditioning slots are not the registry size")
    from llmzoo.models import registry as R

    check("N_COND_SLOTS is a fixed constant",
          R.N_COND_SLOTS == 32, f"got {R.N_COND_SLOTS}")
    check("N_FAMILIES aliases the fixed slot count, not len(ARCH_CONFIGS)",
          R.N_FAMILIES == R.N_COND_SLOTS,
          f"N_FAMILIES={R.N_FAMILIES} N_ARCHS={R.N_ARCHS}")
    check("registering the 3 zoo archs did NOT move the table size",
          R.N_ARCHS != R.N_COND_SLOTS or R.N_ARCHS == 32,
          "the two happen to be equal, which defeats the test -- change one")
    check("every family_idx is inside the table",
          all(0 <= c["family_idx"] < R.N_COND_SLOTS
              for c in R.ARCH_CONFIGS.values()))
    idxs = [c["family_idx"] for c in R.ARCH_CONFIGS.values()]
    check("family_idx values are unique", len(idxs) == len(set(idxs)),
          f"{sorted(idxs)}")


def test_gpt2_configs():
    section("GPT-2 zoo configs match the published architecture")
    from llmzoo.models.registry import (
        GPT2_ZOO_N_POSITIONS, GPT2_ZOO_VOCAB_SIZE, zoo_config, zoo_param_count,
    )

    # RESEARCH_PLAN §4.5: Small and Medium must be exactly the published configs.
    for arch, L, d, h, total in [
        ("gpt2_zoo_mini", 8, 512, 8, 51_475_968),
        ("gpt2_zoo_small", 12, 768, 12, 124_439_808),
        ("gpt2_zoo_medium", 24, 1024, 16, 354_823_168),
    ]:
        c = zoo_config(arch)
        check(f"{arch}: n_layer/n_embd/n_head",
              (c.n_layer, c.n_embd, c.n_head) == (L, d, h),
              f"got {(c.n_layer, c.n_embd, c.n_head)}")
        check(f"{arch}: vocab 50257, n_positions 1024",
              (c.vocab_size, c.n_positions) == (GPT2_ZOO_VOCAB_SIZE,
                                                GPT2_ZOO_N_POSITIONS))
        check(f"{arch}: n_inner == 4*n_embd", c.n_inner == 4 * d)
        check(f"{arch}: embeddings tied", c.tie_word_embeddings is True)
        check(f"{arch}: analytic param count == {total:,}",
              zoo_param_count(arch)["total"] == total,
              f"got {zoo_param_count(arch)['total']:,}")

    check("vocab is identical at every scale (else the scaling axis is confounded)",
          len({zoo_config(a).vocab_size for a in
               ("gpt2_zoo_mini", "gpt2_zoo_small", "gpt2_zoo_medium")}) == 1)


def test_load_model_refuses_zoo_archs():
    section("load_model refuses a from-scratch arch instead of 404ing on the hub")
    from llmzoo.models.registry import load_model
    try:
        load_model("gpt2_zoo_small")
        check("raises ValueError", False, "no exception")
    except ValueError as e:
        check("raises ValueError naming build_zoo_model",
              "build_zoo_model" in str(e), str(e)[:120])
    except Exception as e:                                    # noqa: BLE001
        check("raises ValueError", False, f"got {type(e).__name__}: {e}")


def test_mixtures():
    section("mixture plan (RESEARCH_PLAN §6.3)")
    from llmzoo.data.mixtures import (
        DOMAINS, anchor_mixtures, build_zoo_plan, holdout_split, sample_simplex,
    )

    check("5 domains", len(DOMAINS) == 5, str(DOMAINS))
    anchors = anchor_mixtures()
    check("anchors are the 5 one-hot vertices",
          all(abs(a.sum() - 1) < 1e-12 and a.max() == 1.0 for a in anchors))

    plan = build_zoo_plan(100, 4, seed=0)
    check("plan has 100 members", len(plan) == 100, str(len(plan)))
    n_anchor = sum(1 for m in plan if m["kind"] == "anchor")
    n_single = sum(1 for m in plan if m["kind"] == "singleton")
    check("20 anchors + 80 singletons", (n_anchor, n_single) == (20, 80),
          f"{n_anchor}+{n_single}")
    check("indices are 0..99 in order",
          [m["idx"] for m in plan] == list(range(100)))
    check("every pi is on the simplex",
          all(abs(sum(m["pi"]) - 1) < 1e-9 and min(m["pi"]) >= 0 for m in plan))

    pis = {tuple(np.round(m["pi"], 6)) for m in plan}
    check("85 distinct mixtures", len(pis) == 85, f"got {len(pis)}")

    # A truncated run must still contain WHOLE anchor groups, or the beta
    # calibration at N=12 has no within-mixture noise floor.
    small = build_zoo_plan(12, 4, seed=0)
    check("N=12 plan is all anchors, 3 whole groups",
          len(small) == 12 and all(m["kind"] == "anchor" for m in small)
          and len({m["mixture_id"] for m in small}) == 3,
          f"{len(small)} members, "
          f"{len({m['mixture_id'] for m in small})} groups")

    sp = holdout_split(plan, n_interior=4, holdout_vertex="math")
    check("holdout: 4 interior", len(sp["interior"]) == 4, str(sp["interior"]))
    check("holdout: a whole anchor group as the vertex",
          len(sp["vertex"]) == 4, str(sp["vertex"]))
    check("train + holdout partitions the plan",
          len(sp["train"]) + len(sp["interior"]) + len(sp["vertex"]) == 100
          and not (set(sp["train"]) & set(sp["interior"] + sp["vertex"])))

    check("sample_simplex is deterministic in seed",
          np.allclose(sample_simplex(5, seed=3)[0], sample_simplex(5, seed=3)[0]))
    pts = sample_simplex(30, seed=1, avoid=anchors, min_l1_gap=0.15)
    gaps = [np.abs(a - b).sum() for i, a in enumerate(pts) for b in pts[i + 1:]]
    check("rejection sampling honours min_l1_gap",
          min(gaps) >= 0.15 - 1e-9, f"min gap {min(gaps):.4f}")


def test_source_abstraction(tiny: bool):
    section("Phase 0.3 -- MemberSource abstraction")
    from llmzoo.data.ensemble import (
        ENSEMBLE_SOURCES, NoiseSource, ZooSource, build_source,
    )

    check("two sources registered", set(ENSEMBLE_SOURCES) == {"noise", "zoo"})
    check("build_source('noise') -> NoiseSource",
          isinstance(build_source("noise"), NoiseSource))
    check("build_source('zoo') -> ZooSource",
          isinstance(build_source("zoo", zoo_dir="/tmp/x"), ZooSource))
    try:
        build_source("zoo")
        check("zoo without zoo_dir raises", False, "no exception")
    except ValueError:
        check("zoo without zoo_dir raises", True)
    try:
        build_source("revisions")
        check("unknown source raises", False, "no exception")
    except ValueError:
        check("unknown source raises", True)


def test_noise_source_fingerprint_unchanged():
    section("Phase 0.3 acceptance -- source='noise' fingerprints unchanged")
    from llmzoo.artifacts.io import ensemble_fingerprint

    # A meta dict as written BEFORE Phase 0.3 (no `source` key at all) must hash
    # identically to one written after, because _save_meta omits the default.
    before = {
        "layout_version": 2, "arch_list": ["gpt2_medium"], "n_samples": 100,
        "noise_scale": 0.003, "exclude_1d": True, "include_extra": True,
        "mode": "full", "seed": 42, "chunk_budget_bytes": 536870912,
        "stacks": {"gpt2_medium": {"n_params": 354501632, "n_layers": 24,
                                   "block_size": 12582912, "extra_size": 52511744,
                                   "weight_std": 0.108312}},
    }
    after = dict(before)                       # what _save_meta writes now
    check("no-source meta == explicit-noise meta",
          ensemble_fingerprint(before) == ensemble_fingerprint(
              {**before, "source": "noise"}),
          f"{ensemble_fingerprint(before)} vs "
          f"{ensemble_fingerprint({**before, 'source': 'noise'})}")
    check("source='zoo' produces a DIFFERENT fingerprint",
          ensemble_fingerprint(after) != ensemble_fingerprint(
              {**after, "source": "zoo"}))


def test_zoo_roundtrip():
    section("EnsembleDataset(source='zoo') end to end on synthetic members")
    import torch

    from llmzoo.data.ensemble import ENSEMBLE_LAYOUT_VERSION, EnsembleDataset

    tmp = tempfile.mkdtemp(prefix="zootest_")
    try:
        # D MUST EXCEED MIN_CHUNK_ELEMS OR THE GRID CANNOT SPLIT.
        #
        # chunk_size() is np.clip(raw, MIN_CHUNK_ELEMS, min(MAX_CHUNK_ELEMS,
        # n_params)), and with n_params below MIN_CHUNK_ELEMS (262,144) the upper
        # bound wins, so chunk_size == n_params and there is always exactly ONE
        # chunk. This test used D = 92 and asserted more than one chunk, which no
        # value of chunk_budget_bytes could ever satisfy -- it had never been run.
        #
        # Sizing above the floor is what makes the two assertions below mean
        # anything: reassembling a member from a single chunk is not a test of
        # the streaming path, and misstep 10 (the chunk grid must stay a fixed
        # grid) is only observable across a boundary. 6 x 450k float32 = 10.8 MB.
        arch, N, L, blk, extra = "gpt2_zoo_mini", 6, 4, 100_000, 50_000
        D = extra + L * blk                       # 450,000 > 262,144
        adir = os.path.join(tmp, "zoo", arch)
        os.makedirs(adir)
        rng = np.random.default_rng(0)
        members = [rng.standard_normal(D).astype(np.float32) for _ in range(N)]
        for i, w in enumerate(members):
            np.save(os.path.join(adir, f"w_{i}.npy"), w)
        json.dump({
            "layout_version": ENSEMBLE_LAYOUT_VERSION, "arch": arch,
            "n_members": N, "exclude_1d": False, "include_extra": True,
            "n_layers": L, "block_size": blk, "extra_size": extra,
            "n_params": D, "weight_std": 1.0,
            "schema": [["w", [blk]]], "extra_schema": [["e", [extra]]],
        }, open(os.path.join(adir, "zoo_meta.json"), "w"))

        ds = EnsembleDataset(
            arch_list=[arch], n_samples=N, noise_scale=0.0, exclude_1d=False,
            include_extra=True, mode="full",
            artifact_dir=os.path.join(tmp, "art"),
            source="zoo", zoo_dir=os.path.join(tmp, "zoo"),
            chunk_budget_bytes=1 << 14)

        check("D adopted from zoo_meta", ds.stacks[arch].n_params == D)
        check("more than one chunk (exercises the grid)",
              len(ds.chunk_bounds(arch)) > 1,
              f"{len(ds.chunk_bounds(arch))} chunks for D={D}, "
              f"chunk_size={ds.chunk_size(arch)}")

        # The real test: streaming chunk-by-chunk must reassemble every member.
        rebuilt = np.zeros((N, D), dtype=np.float32)
        for ci, (r0, r1) in enumerate(ds.chunk_bounds(arch)):
            B = ds.load_chunk(arch, ci)
            check_shape = tuple(B.shape) == (N, r1 - r0)
            if not check_shape:
                check(f"chunk {ci} shape", False, str(tuple(B.shape)))
            rebuilt[:, r0:r1] = B.numpy()
        check("every member reassembles exactly from chunks",
              all(np.array_equal(rebuilt[i], members[i]) for i in range(N)))
        check("materialize_sample matches the file",
              np.array_equal(ds.materialize_sample(arch, 3), members[3]))
        check("chunk grid is stable across calls",
              ds.chunk_bounds(arch) == ds.chunk_bounds(arch))

        # A zoo is not mean-centred on any member, unlike a noise ensemble.
        mean = rebuilt.mean(0)
        d0 = np.linalg.norm(members[0] - mean)
        dk = np.median([np.linalg.norm(members[i] - mean) for i in range(1, N)])
        check("member 0 is NOT at a privileged centre (misstep 9 does not apply)",
              0.5 < d0 / dk < 2.0, f"ratio {d0/dk:.3f}")

        # Guards
        try:
            EnsembleDataset(arch_list=[arch], n_samples=N, noise_scale=1e-3,
                            exclude_1d=False, artifact_dir=os.path.join(tmp, "a2"),
                            source="zoo", zoo_dir=os.path.join(tmp, "zoo"))
            check("zoo + nonzero noise_scale is refused", False, "no exception")
        except ValueError as e:
            check("zoo + nonzero noise_scale is refused", "noise_scale" in str(e))
        try:
            EnsembleDataset(arch_list=[arch], n_samples=N, noise_scale=0.0,
                            exclude_1d=True,          # zoo was written with False
                            artifact_dir=os.path.join(tmp, "a3"),
                            source="zoo", zoo_dir=os.path.join(tmp, "zoo"))
            check("exclude_1d mismatch is refused", False, "no exception")
        except RuntimeError as e:
            check("exclude_1d mismatch is refused", "exclude_1d" in str(e))
        try:
            EnsembleDataset(arch_list=[arch], n_samples=N + 3, noise_scale=0.0,
                            exclude_1d=False, artifact_dir=os.path.join(tmp, "a4"),
                            source="zoo", zoo_dir=os.path.join(tmp, "zoo"))
            check("asking for more members than exist is refused", False, "none")
        except RuntimeError as e:
            check("asking for more members than exist is refused",
                  "members" in str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()

    print("=" * 72)
    print("tests/test_zoo.py -- Phase 0.2, 0.3 and the zoo plumbing")
    print("=" * 72)

    test_cond_slots_decoupled()
    test_gpt2_configs()
    test_load_model_refuses_zoo_archs()
    test_mixtures()
    test_source_abstraction(a.tiny)
    test_noise_source_fingerprint_unchanged()
    test_zoo_roundtrip()

    print("\n" + "=" * 72)
    print(f"{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
