"""
Thin Weights & Biases integration, shared by train.py / run_hpc.py /
eval_baseline.py / eval_mc.py.

Degrades to a no-op if wandb isn't installed or logging is disabled
(--no_wandb), so nothing in the pipeline ever depends on it to run.

Usage
-----
Each standalone entrypoint's main() calls init_run() once; shared stage
functions (train_vae, evaluate_family, evaluate_family_mc, ...) just call
log() unconditionally — it no-ops if no run is active, so those functions
work the same whether invoked standalone or from run_hpc.py's orchestration.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

PROJECT = "llm-vae"

# wandb.init() silently falls back to the account's server-side "default
# team" when entity isn't passed explicitly — for this account that's a
# shared team entity, not the personal one runs should actually land in.
# Pin it so that ambiguity can't bite again. Override with WANDB_ENTITY if
# this ever needs to change.
ENTITY = os.environ.get("WANDB_ENTITY", "scottbiggs2001-northeastern-university")


def init_run(
    job_type: str,
    config: dict,
    tags: Optional[list] = None,
    enabled: bool = True,
    artifact_dir: Optional[str] = None,
    name_suffix: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """
    Start a W&B run. No-ops (returns False) if wandb is missing, disabled,
    or a run is already active in this process (e.g. run_hpc.py already
    opened one before calling into train.py's stage functions).

    name_suffix
        Appended to the run name. One Slurm job routinely invokes the same
        script several times -- slurm_stack_run.sh trains a VAE at k=N-1 and
        again at k=N/2, and the flow scripts add two spaces on top of that --
        and each invocation is a separate process, so each opens its own W&B
        run. Without a suffix they all land under the identical name
        `train_stack_<jobid>` and become impossible to tell apart in the UI.
        Pass something like "k99" or "k99_codes".

    group
        W&B run group, so every run from one pipeline invocation collapses into
        a single expandable row. Defaults to the Slurm job id, which is exactly
        the right granularity: one sbatch, one group.
    """
    if not enabled:
        return False
    if _wandb is None:
        print("[wandb] not installed — skipping W&B logging (pip install wandb to enable)")
        return False
    if _wandb.run is not None:
        return True

    # All wandb state on scratch, never $HOME (Explorer policy — see train.py).
    scratch = artifact_dir or os.environ.get("ARTIFACT_DIR", "/scratch/biggs.s/llm_vae")
    os.environ.setdefault("WANDB_DIR", scratch)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    name = f"{job_type}_{job_id}"
    if name_suffix:
        name = f"{name}_{name_suffix}"
    _wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        job_type=job_type,
        group=group or f"job_{job_id}",
        config=config,
        tags=tags or [],
    )
    return True


def log(data: dict, step: Optional[int] = None) -> None:
    if _wandb is not None and _wandb.run is not None:
        _wandb.log(data, step=step)


def summary(data: dict) -> None:
    """
    Write final scalars into the run summary rather than the history.

    Summary values are what the W&B runs TABLE shows as columns, so this is how a
    sweep over ranks and flow spaces becomes scannable at a glance instead of
    requiring each run to be opened. Use it for terminal values (final KL, best
    loss, fitted rank), and log() for anything that varies per epoch.
    """
    if _wandb is not None and _wandb.run is not None:
        _wandb.run.summary.update(data)


def finish() -> None:
    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()
