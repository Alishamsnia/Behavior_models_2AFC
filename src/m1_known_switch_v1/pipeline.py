"""Data preparation for M1_known_switch_v1.

Delegates raw-file discovery, canonical trial preparation, the legacy
five-short/five-long inclusion rule, and the "never open locked test" leak
guard to M0's own registered pipeline (``prepare_registered_fit_data``, via
the proxied ``source_m0_registry`` -- NOT edited by this module), exactly as
``m1_dynamic_block_length_v1/pipeline.py`` delegates to M1's. Unlike that
model, this model's session-tensor contract is NOT identical to M0's (it
needs the oracle ``is_switch`` field), so the returned canonical timeline is
re-grouped through this model's own frame adapter rather than reusing M0's
session tensors directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m0_sensory_v1.pipeline import FitInputAudit, prepare_registered_fit_data

from .registry import M1KnownSwitchRegistry
from .session import M1KnownSwitchSessionTrials, m1_known_switch_session_trials_from_frame


@dataclass(frozen=True)
class M1KnownSwitchPreparedFitData:
    """Train/validation tensors, canonical timelines, and input audit."""

    training_sessions: tuple[M1KnownSwitchSessionTrials, ...]
    validation_sessions: tuple[M1KnownSwitchSessionTrials, ...]
    canonical_trials: pd.DataFrame
    audit: FitInputAudit


def prepare_registered_m1_known_switch_fit_data(
    raw_data_root: str | Path,
    registry: M1KnownSwitchRegistry,
) -> M1KnownSwitchPreparedFitData:
    r"""Prepare train/validation timelines via M0's own leak-safe pipeline.

    Purpose
    -------
    Reuse M0's audited discovery/inclusion-rule/leak-safety guarantees
    exactly (by calling its registered pipeline function with the real,
    proxied M0 registry), then rebuild this model's own session tensors
    (with the oracle ``is_switch`` field) from the returned canonical
    timeline rather than reimplementing discovery or the inclusion rule.

    Mathematical definition
    -----------------------
    Identical timeline contract to M0/M1:

    .. math::

       \mathcal S_d=\{(s_{d,t},y_{d,t},C_{d,t},F_{d,t},b_{d,t})\}_{t=1}^{T_d},

    where ``b`` is the canonical ``true_block`` label this model's own
    adapter reduces to a switch-only signal (see ``session.py``).

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
    Returns ordered train/validation session tuples built by this model's
    own adapter, M0's canonical timeline (with ``split_role`` merged in),
    and M0's own input audit (train/validation session IDs, opened file
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
    A session with zero switches (never leaves Neutral) is valid and passed
    through unchanged.

    Related tests
    -------------
    ``test_m1_known_switch_pipeline_matches_m0_audit_and_builds_switch_sessions``.
    """

    if registry.model_id != "M1_known_switch_v1":
        raise ValueError(
            "prepare_registered_m1_known_switch_fit_data requires an M1_known_switch_v1 registry"
        )
    base = prepare_registered_fit_data(raw_data_root, registry.source_m0_registry)

    training: list[M1KnownSwitchSessionTrials] = []
    validation: list[M1KnownSwitchSessionTrials] = []
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
        session = m1_known_switch_session_trials_from_frame(frame, split_role=role)
        (training if role == "train" else validation).append(session)

    return M1KnownSwitchPreparedFitData(
        training_sessions=tuple(training),
        validation_sessions=tuple(validation),
        canonical_trials=base.canonical_trials,
        audit=base.audit,
    )
