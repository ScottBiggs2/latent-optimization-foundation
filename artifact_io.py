"""
Shared artifact primitives: version gates, fingerprints, atomic JSON, provenance.

This is a LEAF module. It imports nothing from this repo, deliberately: `vae.py`,
`flow.py` and `run_bundle.py` all need these helpers, and `run_bundle.py` needs
`vae.py` and `flow.py`, so putting the helpers in `run_bundle.py` would be an
import cycle.

Why any of this exists
----------------------
`DualGramPCA` and `EnsembleDataset` already carry a `layout_version` and refuse a
mismatch. `StackVAE` carried nothing: persistence was a bare `state_dict` plus a
six-key config JSON, with no record of which ensemble or which code statistics
produced it. RESEARCH_NOTES misstep 17 is exactly that failure mode -- "do not gate
a pipeline stage on os.path.exists alone" -- so the fingerprints below give every
downstream artifact a way to say which upstream artifact it belongs to, and to
refuse when the answer is "not this one".

The fingerprints are content hashes over the fields that CHANGE THE DATA, not over
whole metadata blobs. Hashing a whole blob would make a checkpoint stale whenever a
cosmetic field is added; hashing too little would let a genuinely different ensemble
pass as the same one.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, Optional

# Bumped whenever run_manifest.json's shape changes.
MANIFEST_VERSION = 1

# Bumped whenever codes_k<k>/ changes shape.
CODE_STATS_VERSION = 1


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def atomic_write_json(path: str, payload: dict) -> None:
    """
    Write JSON via a temp file plus os.replace, so a reader never sees a partial
    file.

    This matters here specifically because one Slurm job invokes train_stack.py
    once per rank and train_flow.py once per (rank, space). Those are separate
    processes writing different sections of the same run manifest, so a torn file
    is a real possibility and is far worse than a lost update -- a lost update is
    recovered by rebuild_manifest(), a torn file is not recovered by anything.
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)


def read_json(path: str) -> Optional[dict]:
    """Parse `path`, or return None if it does not exist. Malformed JSON raises."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Version gates
# ---------------------------------------------------------------------------

def require_version(meta: dict, key: str, expected: int, where: str,
                    override_flag: Optional[str] = None) -> None:
    """
    Raise unless meta[key] == expected.

    Message shape mirrors DualGramPCA.load and EnsembleDataset._load_meta: name the
    directory, the value found, the value expected, and the exact flag that
    overrides. Anything vaguer and the next person guesses.
    """
    found = meta.get(key)
    if found != expected:
        extra = (f" To proceed anyway, pass {override_flag}."
                 if override_flag else "")
        raise RuntimeError(
            f"Refusing to load {where}: {key}={found!r} but this code expects "
            f"{expected}. Re-generate the artifact.{extra}")


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def fingerprint(payload: dict) -> str:
    """
    Stable short content hash. Key order does not matter; values do.

    Truncated to 16 hex chars: this identifies artifacts inside one run directory,
    it is not a security boundary, and a full sha1 makes the metadata unreadable.
    """
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def ensemble_fingerprint(ens_meta: dict) -> str:
    """
    Fingerprint the fields that determine an ensemble's CONTENTS.

    Notes on two entries that look optional and are not:

    * `chunk_budget_bytes` sets the chunk grid, and the augmentation noise is keyed
      to (seed, chunk_idx) -- RESEARCH_NOTES misstep 10. Two ensembles with the same
      seed but different grids are different ensembles.
    * `source` defaults to "noise". When EnsembleDataset gains source="revisions"
      (Experiment 1), every artifact fingerprinted before that flips automatically,
      with no code change here and no chance of pairing a revisions PCA with a
      noise-ensemble VAE.
    """
    stacks = ens_meta.get("stacks", {})
    payload = {
        "layout_version": ens_meta.get("layout_version"),
        "arch_list": sorted(ens_meta.get("arch_list", [])),
        "n_samples": ens_meta.get("n_samples"),
        "noise_scale": ens_meta.get("noise_scale"),
        "exclude_1d": ens_meta.get("exclude_1d"),
        "include_extra": ens_meta.get("include_extra"),
        "mode": ens_meta.get("mode"),
        "seed": ens_meta.get("seed"),
        "chunk_budget_bytes": ens_meta.get("chunk_budget_bytes"),
        "source": ens_meta.get("source", "noise"),
        "per_arch": {
            a: {
                "n_params": d.get("n_params"),
                "n_layers": d.get("n_layers"),
                "block_size": d.get("block_size"),
                "extra_size": d.get("extra_size", 0),
                # Rounded: weight_std is a float32 reduction, so its last digits
                # are not reproducible across BLAS versions, and a fingerprint
                # that flips on a library upgrade is worse than no fingerprint.
                "weight_std": round(float(d.get("weight_std", 0.0)), 9),
            }
            for a, d in sorted(stacks.items())
        },
    }
    return fingerprint(payload)


def pca_fingerprint(pca_meta: dict) -> str:
    """Fingerprint one family's fitted basis."""
    return fingerprint({
        "layout_version": pca_meta.get("layout_version"),
        "arch": pca_meta.get("arch"),
        "n_components": pca_meta.get("n_components"),
        "n_samples": pca_meta.get("n_samples"),
        "n_params": pca_meta.get("n_params"),
        "rank_rtol": pca_meta.get("rank_rtol"),
        "total_variance": (None if pca_meta.get("total_variance") is None
                           else round(float(pca_meta["total_variance"]), 9)),
    })


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def git_commit() -> Optional[str]:
    """Short HEAD, or None. Never raises -- provenance must not break a training run."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def provenance_block(
    run_name: str,
    k: int,
    ensemble_fp: Optional[str] = None,
    pca_fps: Optional[Dict[str, str]] = None,
    code_stats_fp: Optional[str] = None,
    trust: str = "verified",
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    The block every sealed artifact carries, so it can say what it came from.

    `trust` is "verified" for anything written by this code path and
    "unverified-legacy" for anything adopted from a pre-provenance directory. It
    propagates: a flow trained on an unverified VAE inherits the label, and
    report_stack renders every affected row with a footnote.
    """
    block = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "run_name": run_name,
        "k": int(k),
        "ensemble_fingerprint": ensemble_fp,
        "pca_fingerprints": dict(pca_fps or {}),
        "code_stats_fingerprint": code_stats_fp,
        "trust": trust,
    }
    if extra:
        block.update(extra)
    return block
