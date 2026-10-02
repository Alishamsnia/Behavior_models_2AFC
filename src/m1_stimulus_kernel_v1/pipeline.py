"""Data preparation for M1_stimulus_kernel_v1.

Delegates raw-file discovery, canonical trial preparation, the legacy
five-short/five-long inclusion rule, and the "never open locked test" leak
guard to M0's own registered pipeline (``prepare_registered_fit_data``, via
the proxied ``source_m0_registry`` -- NOT edited by this module), exactly
as ``m1_known_switch_v1/pipeline.py`` delegates to it. Unlike
``m1_known_switch_v1``, this model's session-tensor contract is identical
to M1's own (no oracle field is needed), so sessions are built directly
with M1's own frame adapter (``m1_session_trials_from_frame``) rather than
a model-specific one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame
from m0_sensory_v1.pipeline import FitInputAudit, prepare_registered_fit_data

from .registry import M1StimulusKernelRegistry


@dataclass(frozen=True)
class M1StimulusKernelPreparedFitData:
    """Train/validation tensors, canonical timelines, and input audit."""

    training_sessions: tuple[M1SessionTrials, ...]
    validation_sessions: tuple[M1SessionTrials, ...]
    canonical_trials: pd.DataFrame
    audit: FitInputAudit


def prepare_registered_m1_stimulus_kernel_fit_data(
    raw_data_root: str | Path,
    registry: M1StimulusKernelRegistry,
) -> M1StimulusKernelPreparedFitData:
    r"""Prepare train/validation timelines via M0's own leak-safe pipeline.

    Purpose
    -------
    Reuse M0's audited discovery/inclusion-rule/leak-safety guarantees
    exactly (by calling its registered pipeline function with the real,
    proxied M0 registry), then rebuild this model's session tensors with
    M1's own frame adapter from the returned canonical timeline rather than
    reimplementing discovery or the inclusion rule.

    Mathematical definition
    -----------------------
    Identical timeline contract to M0/M1:

    .. math::

       \mathcal S_d=\{(s_{d,t},y_{d,t},C_{d,t},F_{d,t})\}_{t=1}^{T_d}.

    Parameter interpretation
    ------------------------
    ``registry.source_m0_registry`` is the real, subject-specific M0
    registry (task moments, split, epsilons); this function never
    constructs its own notion of the frozen split.

    Input units and shapes
    ----------------------
    ``raw_data_root`` is a directory tree matched against the frozen split
    by exact session ID.

    Returned quantities and shapes
    ------------------------------
    Returns ordered train/validation session tuples built by M1's own
    adapter, M0's canonical timeline (with ``split_role`` merged in), and
    M0's own input audit (train/validation session IDs, opened file
    checksums, locked-test-files-opened count).

    Constraints
    -----------
    Delegates every constraint (frozen split membership, legacy inclusion
    rule, one-mouse-per-run) to ``prepare_registered_fit_data``.

    Numerical stability
    -------------------
    Not applicable; this function performs I/O and adapter conversion only.

    Causal restrictions
    -------------------
    Locked test MATLAB files are never opened -- guaranteed transitively by
    M0's own pipeline, not re-implemented here.

    Important edge cases
    --------------------
    A session with zero feedback-valid trials is valid and simply keeps a
    constant 0.5 prior throughout.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_pipeline_matches_m0_audit_and_builds_sessions``.
    """

    if registry.model_id != "M1_stimulus_kernel_v1":
        raise ValueError(
            "prepare_registered_m1_stimulus_kernel_fit_data requires an M1_stimulus_kernel_v1 registry"
        )
    base = prepare_registered_fit_data(raw_data_root, registry.source_m0_registry)

    training: list[M1SessionTrials] = []
    validation: list[M1SessionTrials] = []
    for session_id, frame in base.canonical_trials.groupby(
        "session_id", sort=False, observed=False
    ):
        roles = frame["split_role"].dropna().unique().tolist()
        if not roles:
            continue
        if len(roles) != 1:
            raise ValueError(f"Session {session_id!r} has inconsistent split_role values")
        role = str(roles[0])
        if role not in ("train", "validation"):
            continue
        session = m1_session_trials_from_frame(frame, split_role=role)
        (training if role == "train" else validation).append(session)

    return M1StimulusKernelPreparedFitData(
        training_sessions=tuple(training),
        validation_sessions=tuple(validation),
        canonical_trials=base.canonical_trials,
        audit=base.audit,
    )
