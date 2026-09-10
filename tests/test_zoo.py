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
import glob
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


def test_retry_classification():
    """
    open_domain_stream must retry a rate limit and NOT retry a dead id.

    The first singleton-bearing job this repo ran died on a real HF 429
    (2026-09-08). The beta calibration could not have found it: a one-hot anchor
    takes mixture_stream's len(streams)==1 fast path, so Phase 1's jobs resolved
    ONE dataset each while every singleton resolves five.

    The negative cases are the point. Retrying a renamed-or-gated dataset id six
    times and then reporting a rate limit would hide the single most fragile
    thing in this repo, and would invite the corpus edit RESEARCH_PLAN §6.3
    forbids -- which silently invalidates the beta calibration.
    """
    section("retry classification (mixtures.open_domain_stream)")
    from llmzoo.data.mixtures import _is_retryable, open_domain_stream

    # verbatim text of the failure on 2026-09-08
    real429 = Exception(
        "429 Client Error: Too Many Requests for url: "
        "https://huggingface.co/api/datasets/manu/project_gutenberg/revision/"
        "164853d2 (Request ID: Root=1-6aa0bb15) We had to rate limit your IP "
        "(192.69.103.196). To continue using our service, create a HF account")
    for exc, want, label in (
        (real429,                                  True,  "real 429"),
        (TimeoutError("connection timed out"),     True,  "timeout"),
        (Exception("503 Service Unavailable"),     True,  "5xx"),
        (KeyError("text_column 'text' not in"),    False, "schema break"),
        (FileNotFoundError("dataset not found"),   False, "dead dataset id"),
        (ValueError("mixture is all zeros"),       False, "own validation"),
    ):
        check(f"retryable({label}) == {want}", _is_retryable(exc) is want,
              f"got {_is_retryable(exc)}")

    # And the loop itself: a transient failure must be survived, a permanent one
    # must surface on the FIRST attempt rather than after six sleeps.
    import llmzoo.data.mixtures as mx
    calls = {"n": 0}

    class _FakeDatasets:
        @staticmethod
        def load_dataset(*_a, **_k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise real429
            return "STREAM"

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def fake_import(name, *a, **k):
        if name == "datasets":
            return _FakeDatasets
        return real_import(name, *a, **k)

    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = fake_import
    else:
        __builtins__.__import__ = fake_import
    try:
        got = open_domain_stream("web", base_delay=0.001, log=lambda *_a: None)
        check("transient 429 survived after retries",
              got[0] == "STREAM" and got[1] == "text" and calls["n"] == 3,
              f"got {got!r} after {calls['n']} calls")

        calls["n"] = 0

        class _DeadId:
            @staticmethod
            def load_dataset(*_a, **_k):
                calls["n"] += 1
                raise FileNotFoundError("Dataset 'foo/bar' doesn't exist")

        def fake_import2(name, *a, **k):
            if name == "datasets":
                return _DeadId
            return real_import(name, *a, **k)

        if isinstance(__builtins__, dict):
            __builtins__["__import__"] = fake_import2
        else:
            __builtins__.__import__ = fake_import2
        raised = False
        try:
            open_domain_stream("web", base_delay=0.001, log=lambda *_a: None)
        except FileNotFoundError:
            raised = True
        check("dead dataset id raises immediately, no retry loop",
              raised and calls["n"] == 1,
              f"raised={raised} after {calls['n']} calls")
    finally:
        if isinstance(__builtins__, dict):
            __builtins__["__import__"] = real_import
        else:
            __builtins__.__import__ = real_import



# ---------------------------------------------------------------------------
# Mid-run checkpoint / resume  (scripts/train_zoo.py)
# ---------------------------------------------------------------------------
#
# Runs on the `cpu` partition with NO GPU and NO corpus download. Two facts make
# that possible: train_zoo.py's `from transformers import AutoTokenizer` is inside
# main(), so importing the module touches no network; and mixture_stream is
# monkeypatched here with a seeded fake, so tokenizer=None is never dereferenced.

def _tz():
    """Import scripts/train_zoo.py as a module, the way tests/test_report.py does."""
    here = os.path.dirname(os.path.abspath(__file__))
    scripts = os.path.join(here, os.pardir, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import train_zoo
    return train_zoo


class _FakeLM:
    """Stands in for GPT2LMHeadModel: returns .logits from an nn.Embedding(V, V)."""

    def __init__(self, V=64, seed=0):
        import torch
        import torch.nn as nn
        g = torch.Generator().manual_seed(seed)
        self.V = V
        self.emb = nn.Embedding(V, V)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(V, V, generator=g) * 0.02)

    def __call__(self, input_ids=None, labels=None):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.emb(input_ids))

    def parameters(self):
        return self.emb.parameters()

    def train(self):
        return self

    def state_dict(self):
        return self.emb.state_dict()

    def load_state_dict(self, sd):
        return self.emb.load_state_dict(sd)


# Absolute position in the fake corpus, controlled by the tests rather than by the
# generator, so a resumed span can be placed exactly where the killed one stopped.
_FAKE_POS = [0]


def _fake_stream_factory(V=64):
    """
    A deterministic corpus indexed by ABSOLUTE POSITION, not by seed.

    The resume test drives _FAKE_POS by hand so the resumed segment reads the very
    documents the killed segment would have read next. That is deliberately NOT what
    the real path does -- HF streaming cannot be fast-forwarded cheaply, so a real
    resume re-seeds at base_seed + SEGMENT_SEED_STRIDE * segment and re-reads some
    documents (which is why data_seeds and resume_steps are sealed per member). The
    point of pinning the data here is to isolate the model / optimizer / LR-schedule
    arithmetic, so that a failure means the resume machinery is wrong rather than
    merely that the corpus moved.

    batches() pulls exactly batch_size items per step with no prefetch, so after
    `n` steps the position is n * batch_size * grad_accum.
    """
    import torch

    def fake(pi, tokenizer, n_ctx, seed):
        while True:
            i = _FAKE_POS[0]
            _FAKE_POS[0] = i + 1
            g = torch.Generator().manual_seed(1234 + i)
            yield torch.randint(0, V, (n_ctx + 1,), generator=g)
    return fake


def _run_span(TZ, *, start_step, n_steps, total_steps, seed=7, ckpt_every=0,
              save_ckpt=None, prior=None, model=None, opt=None, V=64,
              lrs=None):
    import torch
    if model is None:
        model = _FakeLM(V=V, seed=0)
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4,
                                betas=(0.9, 0.95), weight_decay=0.1)
    if lrs is not None:
        real_lr_at = TZ.lr_at

        def spy(step, total, peak, warmup, **kw):
            v = real_lr_at(step, total, peak, warmup, **kw)
            lrs.append((step, v))
            return v
        TZ.lr_at = spy
    try:
        thr = TZ.train_span(
            model, opt, np.full(5, 0.2), tokenizer=None,
            device=torch.device("cpu"), n_ctx=15, batch_size=2, grad_accum=1,
            start_step=start_step, n_steps=n_steps, total_steps=total_steps,
            peak_lr=6e-4, warmup=2, seed=seed, log_every=5,
            amp_dtype=torch.float32, ckpt_every=ckpt_every,
            save_ckpt=save_ckpt, prior=prior)
    finally:
        if lrs is not None:
            TZ.lr_at = real_lr_at
    return model, opt, thr


def test_ckpt_paths_and_refusals():
    section("checkpoint: destination and key refusals")
    TZ = _tz()

    # -- the /work refusal, a pure string predicate: no cluster, no filesystem
    art = "/work/neu/p2026_0038_neu/u/llm_vae"
    check("ckpt_dir inside /work is refused",
          TZ.refuse_ckpt_dir("/work/neu/p2026_0038_neu/u/ckpt", art) is not None)
    check("ckpt_dir inside the artifact tree is refused",
          TZ.refuse_ckpt_dir(art + "/ckpt", art) is not None)
    check("ckpt_dir on /scratch is allowed",
          TZ.refuse_ckpt_dir("/scratch/u/zoo_ckpt", art) is None)
    check("default_ckpt_dir lives on /scratch",
          TZ.default_ckpt_dir().startswith("/scratch/"))

    # -- SEGMENT_SEED_STRIDE must exceed train_span's max_restarts, or a resumed
    #    segment could re-read exactly the prefix a restart already read.
    check("SEGMENT_SEED_STRIDE > max_restarts (8)", TZ.SEGMENT_SEED_STRIDE > 8,
          f"stride={TZ.SEGMENT_SEED_STRIDE}")

    d = tempfile.mkdtemp(prefix="zoockpt_")
    try:
        base = dict(arch="gpt2_zoo_small", beta=0.15, n_members=100,
                    span="member_7", member_idx=7, total_steps=18988,
                    trunk_steps=16140, branch_steps=2848, tokens_per_step=131072,
                    n_ctx=1024, warmup=189, peak_lr=6e-4, base_seed=10007,
                    zoo_dir=d, trunk_seed=1234, trunk_step=16140,
                    device_name="NVIDIA B200")
        key = TZ.ckpt_key(**base)
        path = TZ.ckpt_path(d, "gpt2_zoo_small", 0.15, 100, "member_0007")
        check("ckpt_path carries arch, beta and N in the directory",
              "gpt2_zoo_small_b015_n100" in path and path.endswith("member_0007.pt"),
              path)

        # -- missing file is not an error and not a refusal
        ck, why = TZ.load_ckpt(path, key)
        check("a missing checkpoint is (None, []) not a refusal",
              ck is None and why == [])

        TZ.save_ckpt_atomic(path, {"format": TZ.ZOO_CKPT_VERSION, "key": key,
                                   "step": 500, "acc": {"segments": 1}})
        check("no .tmp survives an atomic write",
              os.path.exists(path) and not glob.glob(path + ".tmp*"))

        ck, why = TZ.load_ckpt(path, key)
        check("a matching key resumes", ck is not None and ck["step"] == 500, str(why))

        # -- every key field must refuse, and must NAME the field it refused on
        for field, bad in [("arch", "gpt2_zoo_medium"), ("beta", 0.30),
                           ("n_members", 60), ("member_idx", 8),
                           ("total_steps", 54141), ("trunk_steps", 46020),
                           ("device_name", "NVIDIA GeForce RTX 4090")]:
            want = dict(key)
            want[field] = bad
            ck2, why2 = TZ.load_ckpt(path, want)
            check(f"key mismatch on {field!r} is refused and named",
                  ck2 is None and any(repr(field) in r for r in why2),
                  f"{why2}")

        # -- allow_device_change is expressed by dropping the field from the
        #    comparison, so the rest of the key still has to match
        want = {k: v for k, v in key.items() if k != "device_name"}
        ck3, why3 = TZ.load_ckpt(path, want)
        check("dropping device_name from the key permits a cross-device resume",
              ck3 is not None, str(why3))

        # -- a garbage FINAL file is a refusal, never a silent restart
        with open(path, "wb") as f:
            f.write(b"not a torch file")
        ck4, why4 = TZ.load_ckpt(path, key)
        check("an undeserializable checkpoint refuses rather than restarting",
              ck4 is None and why4 and "will not deserialize" in why4[0], str(why4))

        # -- a stale .tmp is invisible: the loader gates on the final name only
        os.remove(path)
        with open(path + ".tmp.somehost.999", "wb") as f:
            f.write(b"truncated")
        ck5, why5 = TZ.load_ckpt(path, key)
        check("a leftover .tmp is never mistaken for a checkpoint",
              ck5 is None and why5 == [])

        # -- drop_ckpt sweeps the orphan tmp too, and tolerates a missing file
        other = TZ.ckpt_path(d, "gpt2_zoo_small", 0.15, 100, "member_0008")
        TZ.save_ckpt_atomic(other, {"format": TZ.ZOO_CKPT_VERSION, "key": key,
                                    "step": 1, "acc": {}})
        TZ.drop_ckpt(path)
        check("drop_ckpt removes orphaned .tmp files",
              not glob.glob(path + ".tmp*"))
        check("drop_ckpt leaves a sibling member's checkpoint alone",
              os.path.exists(other))
        TZ.drop_ckpt(path)
        check("drop_ckpt on a missing path does not raise", True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ckpt_inert_when_off():
    section("checkpoint: inert when --ckpt_every 0")
    TZ = _tz()
    import torch
    TZ.mixture_stream = _fake_stream_factory()

    d = tempfile.mkdtemp(prefix="zooinert_")
    try:
        _FAKE_POS[0] = 0
        m1, o1, thr1 = _run_span(TZ, start_step=0, n_steps=8, total_steps=40,
                                 ckpt_every=0, save_ckpt=None)
        # save_ckpt is None, so nothing can be written even with a huge interval
        _FAKE_POS[0] = 0
        m2, o2, thr2 = _run_span(TZ, start_step=0, n_steps=8, total_steps=40,
                                 ckpt_every=10 ** 9, save_ckpt=None)
        check("no files are written when checkpointing is off",
              os.listdir(d) == [], str(os.listdir(d)))
        same = torch.equal(m1.state_dict()["weight"], m2.state_dict()["weight"])
        check("ckpt_every=0 and ckpt_every=1e9 give bitwise-equal weights", same)

        for k in ("tokens", "seconds", "steady_seconds", "startup_seconds",
                  "tokens_per_sec", "tokens_per_sec_avg", "n_windows",
                  "stream_restarts", "mfu", "peak_tflops", "final_loss"):
            if k not in thr1:
                check(f"legacy throughput key {k!r} survives", False)
                break
        else:
            check("every legacy throughput key survives", True)
        check("an un-resumed span reports segments=1, resumed=False",
              thr1["segments"] == 1 and thr1["resumed"] is False,
              f"{thr1['segments']} / {thr1['resumed']}")
        check("an un-resumed span records no resume steps",
              thr1["resume_steps"] == [] and len(thr1["data_seeds"]) == 1)
        check("stream_restarts equals the sum of its per-segment list",
              thr1["stream_restarts"] == sum(thr1["stream_restarts_per_segment"]))
        check("startup_seconds equals the sum of its per-segment list",
              abs(thr1["startup_seconds"]
                  - sum(thr1["startup_seconds_per_segment"])) < 1e-6)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ckpt_resume_arithmetic():
    section("checkpoint: a resumed span equals an uninterrupted one")
    TZ = _tz()
    import torch
    TZ.mixture_stream = _fake_stream_factory()

    d = tempfile.mkdtemp(prefix="zooresume_")
    try:
        # -- Run A: 20 steps straight through, recording every LR
        _FAKE_POS[0] = 0
        lrs_a = []
        mA, oA, thrA = _run_span(TZ, start_step=0, n_steps=20, total_steps=40,
                                 lrs=lrs_a)

        # -- Run B: 10 steps, checkpoint, restore into FRESH objects, 10 more
        path = TZ.ckpt_path(d, "gpt2_zoo_small", 0.15, 100, "member_0000")
        saved = {}

        def save(next_step, acc, loss):
            saved["step"], saved["acc"] = next_step, acc
            TZ.save_ckpt_atomic(path, {"format": TZ.ZOO_CKPT_VERSION, "key": {},
                                       "step": next_step, "acc": acc,
                                       "model": mB.state_dict(),
                                       "optimizer": oB.state_dict()})

        _FAKE_POS[0] = 0
        lrs_b = []
        mB = _FakeLM(V=64, seed=0)
        oB = torch.optim.AdamW(mB.parameters(), lr=6e-4, betas=(0.9, 0.95),
                               weight_decay=0.1)
        _run_span(TZ, start_step=0, n_steps=20, total_steps=40, ckpt_every=10,
                  save_ckpt=save, model=mB, opt=oB, lrs=lrs_b)
        check("a checkpoint was written at the interval",
              saved.get("step") == 10, str(saved.get("step")))

        ck = torch.load(path, map_location="cpu", weights_only=False)
        mC = _FakeLM(V=64, seed=99)                 # deliberately a DIFFERENT init
        oC = torch.optim.AdamW(mC.parameters(), lr=6e-4, betas=(0.9, 0.95),
                               weight_decay=0.1)
        mC.load_state_dict(ck["model"])
        oC.load_state_dict(ck["optimizer"])
        # Place the resumed segment where the killed one stopped: 10 steps at
        # batch_size 2 and grad_accum 1 is 20 documents.
        _FAKE_POS[0] = ck["step"] * 2
        lrs_c = []
        _, oC, thrC = _run_span(TZ, start_step=ck["step"], n_steps=10,
                                total_steps=40, model=mC, opt=oC,
                                prior=ck["acc"], lrs=lrs_c)

        check("resumed weights are bitwise equal to the uninterrupted run",
              torch.equal(mA.state_dict()["weight"], mC.state_dict()["weight"]))
        stepA = int(list(oA.state.values())[0]["step"])
        stepC = int(list(oC.state.values())[0]["step"])
        check("Adam's own step counter reaches 20 in both", stepA == stepC == 20,
              f"{stepA} vs {stepC}")

        # THE load-bearing one. The trunk/branch design exists so the global cosine
        # continues across the branch point rather than restarting; a resume must
        # not reintroduce the transient that would cause.
        tail = lrs_a[len(lrs_a) - len(lrs_c):]
        check("the per-step LR sequence is identical across the resume",
              lrs_c == tail, f"{lrs_c[:3]} vs {tail[:3]}")

        check("segments accumulates to 2", thrC["segments"] == 2)
        check("resumed is True after a resume", thrC["resumed"] is True)
        check("the resume step is recorded", thrC["resume_steps"] == [10],
              str(thrC["resume_steps"]))
        check("both segments' data seeds are sealed",
              len(thrC["data_seeds"]) == 2, str(thrC["data_seeds"]))
        check("tokens accumulate across segments",
              thrC["tokens"] == thrA["tokens"],
              f"{thrC['tokens']} vs {thrA['tokens']}")
        check("startup is summed over segments, not overwritten",
              len(thrC["startup_seconds_per_segment"]) == 2)
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
    test_retry_classification()
    test_source_abstraction(a.tiny)
    test_noise_source_fingerprint_unchanged()
    test_zoo_roundtrip()
    test_ckpt_paths_and_refusals()
    test_ckpt_inert_when_off()
    test_ckpt_resume_arithmetic()

    print("\n" + "=" * 72)
    print(f"{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
