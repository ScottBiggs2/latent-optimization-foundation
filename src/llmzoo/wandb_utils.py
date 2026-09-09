"""
Thin Weights & Biases integration, shared by every training and evaluation
entry point.

Degrades to a no-op if wandb isn't installed or logging is disabled
(--no_wandb), so nothing in the pipeline ever depends on it to run.

Usage
-----
Each standalone entrypoint's main() calls init_run() once; shared stage
functions (train_vae, evaluate_family, evaluate_family_mc, ...) just call
log() unconditionally — it no-ops if no run is active, so those functions
work the same whether invoked standalone or from a multi-stage sbatch chain.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

# The historical project. Everything landed here until 2026-09-09, so it stays the
# default and no existing run moves.
DEFAULT_PROJECT = "llm-vae"

# `project` is PASSED to wandb.init() below, and an explicit argument beats wandb's own
# WANDB_PROJECT env handling — so the env var has to be read here or it silently does
# nothing. Precedence: the caller's project= argument, then $WANDB_PROJECT, then the
# historical default.
PROJECT = os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)

# wandb.init() silently falls back to the account's server-side "default
# team" when entity isn't passed explicitly — for this account that's a
# shared team entity, not the personal one runs should actually land in.
# Pin it so that ambiguity can't bite again. Override with WANDB_ENTITY if
# this ever needs to change.
ENTITY = os.environ.get("WANDB_ENTITY", "scottbiggs2001-northeastern-university")


def resolve_project(project: Optional[str] = None) -> str:
    """The one place the precedence rule lives, so no caller can reimplement it."""
    return project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT


def add_wandb_args(p) -> None:
    """
    The two routing flags, defined once so every entry point spells them the same.

    Both default to None, which means "keep the historical behaviour": the project
    falls back to $WANDB_PROJECT then "llm-vae", and the group falls back to whatever
    the call site already passed (usually the Slurm job id).
    """
    p.add_argument("--wandb_project", default=None,
                   help="W&B project. Default: $WANDB_PROJECT, then 'llm-vae'. "
                        "Split by workstream, e.g. llmzoo-zoo / llmzoo-genverify.")
    p.add_argument("--wandb_group", default=None,
                   help="W&B run group. Default: the Slurm job id. Pass the "
                        "EXPERIMENT CELL when one experiment spans several jobs, "
                        "e.g. b015_k99 or zoo_gpt2_zoo_mini_b015_n100.")


def init_run(
    job_type: str,
    config: dict,
    tags: Optional[list] = None,
    enabled: bool = True,
    artifact_dir: Optional[str] = None,
    name_suffix: Optional[str] = None,
    group: Optional[str] = None,
    project: Optional[str] = None,
) -> bool:
    """
    Start a W&B run. No-ops (returns False) if wandb is missing, disabled,
    or a run is already active in this process (e.g. an orchestrator already
    opened one before calling into train.py's stage functions).

    name_suffix
        Appended to the run name. One Slurm job routinely invokes the same
        script several times -- slurm/stack_run.sbatch trains a VAE at k=N-1 and
        again at k=N/2, and the flow scripts add two spaces on top of that --
        and each invocation is a separate process, so each opens its own W&B
        run. Without a suffix they all land under the identical name
        `train_stack_<jobid>` and become impossible to tell apart in the UI.
        Pass something like "k99" or "k99_codes".

    group
        W&B run group, so every run from one pipeline invocation collapses into
        a single expandable row. Defaults to the Slurm job id, which is exactly
        the right granularity: one sbatch, one group.

        Pass an EXPERIMENT CELL instead when one logical experiment spans several
        sbatch jobs -- `zoo_<arch>_<tag>_n<N>` for a zoo, `b015_k99` for a
        generative-verification cell. The job-id default cannot express that, which
        is why the spectrum stage historically escaped its zoo's group.

    project
        W&B project. Defaults to $WANDB_PROJECT, then the historical "llm-vae".
        Split by workstream rather than by run: one project per phase keeps the runs
        table scannable, where a single project mixes zoo training, spectra and
        generative evaluation into one undifferentiated list.
    """
    if not enabled:
        return False
    if _wandb is None:
        print("[wandb] not installed — skipping W&B logging (pip install wandb to enable)")
        return False
    if _wandb.run is not None:
        return True

    # All wandb state off $HOME: the home quota is 100 GiB on AICR and a full
    # home fails jobs in about a second, with an opaque exit code.
    scratch = artifact_dir or os.environ.get("ARTIFACT_DIR", "./artifacts")
    os.environ.setdefault("WANDB_DIR", scratch)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    name = f"{job_type}_{job_id}"
    if name_suffix:
        name = f"{name}_{name_suffix}"
    _wandb.init(
        entity=ENTITY,
        project=resolve_project(project),
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
