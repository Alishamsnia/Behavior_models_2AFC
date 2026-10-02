"""Data preparation for M2_known_switch_v1.

Mirrors ``m1_known_switch_v1/pipeline.py`` exactly (delegates discovery,
canonical preparation, the legacy inclusion rule, and the locked-test leak
guard to M0's own ``prepare_registered_fit_data`` via the proxied
``source_m0_registry``), duplicated rather than imported for the same
no-runtime-cross-import reason documented in ``session.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m0_sensory_v1.pipeline import FitInputAudit, prepare_registered_fit_data

from .registry import M2KnownSwitchRegistry
from .session import M2KnownSwitchSessionTrials, m2_known_switch_session_trials_from_frame


@dataclass(frozen=True)
class M2KnownSwitchPreparedFitData:
    """Train/validation tensors, canonical timelines, and input audit."""

    training_sessions: tuple[M2KnownSwitchSessionTrials, ...]
    validation_sessions: tuple[M2KnownSwitchSessionTrials, ...]
    canonical_trials: pd.DataFrame
    audit: FitInputAudit


def prepare_registered_m2_known_switch_fit_data(
    raw_data_root: str | Path,
    registry: M2KnownSwitchRegistry,
) -> M2KnownSwitchPreparedFitData:
    r"""Prepare train/validation timelines via M0's own leak-safe pipeline.

    Purpose
    -------
    Identical strategy to ``m1_known_switch_v1``'s pipeline: reuse M0's
    audited discovery/inclusion-rule/leak-safety guarantees, then rebuild
    this model's own session tensors from the returned canonical timeline.

    Mathematical definition
    -----------------------
    Identical timeline contract to M0/M1/M1_known_switch_v1.

    Parameter interpretation
    ------------------------
    ``registry.source_m0_registry`` is the real, subject-specific M0
    registry.

    Input units and shapes
    ----------------------
    ``raw_data_root`` is a directory tree matched against the frozen split
    by exact session ID.

    Returned quantities and shapes
    ------------------------------
    Returns ordered train/validation session tuples, M0's canonical
    timeline, and M0's own input audit.

    Constraints
    -----------
    Delegates every constraint to ``prepare_registered_fit_data``.

    Numerical stability
    -------------------
    Not applicable; I/O and adapter conversion only.

    Causal restrictions
    -------------------
    Locked test MATLAB files are never opened.

    Important edge cases
    --------------------
    A session with zero switches is valid and passed through unchanged.

    Related tests
    -------------
    ``test_m2_known_switch_pipeline_matches_m0_audit_and_builds_switch_sessions``.
    """

    if registry.model_id != "M2_known_switch_v1":
        raise ValueError(
            "prepare_registered_m2_known_switch_fit_data requires an M2_known_switch_v1 registry"
        )
    base = prepare_registered_fit_data(raw_data_root, registry.source_m0_registry)

    training: list[M2KnownSwitchSessionTrials] = []
    validation: list[M2KnownSwitchSessionTrials] = []
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
        session = m2_known_switch_session_trials_from_frame(frame, split_role=role)
        (training if role == "train" else validation).append(session)

    return M2KnownSwitchPreparedFitData(
        training_sessions=tuple(training),
        validation_sessions=tuple(validation),
        canonical_trials=base.canonical_trials,
        audit=base.audit,
    )
