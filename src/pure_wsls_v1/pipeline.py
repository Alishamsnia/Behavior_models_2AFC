"""Data preparation for Pure_WSLS_v1.

Delegates raw-file discovery, canonical trial preparation, the legacy
five-short/five-long inclusion rule, and the "never open locked test" leak
guard to M0's own registered pipeline (``prepare_registered_fit_data``, via
the proxied ``source_m0_registry`` -- NOT edited by this module), exactly
as ``m1_stimulus_kernel_v1/pipeline.py`` delegates to it. The legacy
duration-based inclusion rule is retained here purely so this model's
fitted cohort stays directly comparable session-for-session with every
other family in this repository, even though duration itself never enters
Pure_WSLS's likelihood (see ``wsls_core_v1.evaluator``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame
from m0_sensory_v1.pipeline import FitInputAudit, prepare_registered_fit_data

from .registry import PureWslsRegistry


@dataclass(frozen=True)
class PureWslsPreparedFitData:
    """Train/validation tensors, canonical timelines, and input audit."""

    training_sessions: tuple[M1SessionTrials, ...]
    validation_sessions: tuple[M1SessionTrials, ...]
    canonical_trials: pd.DataFrame
    audit: FitInputAudit


def prepare_registered_pure_wsls_fit_data(
    raw_data_root: str | Path,
    registry: PureWslsRegistry,
) -> PureWslsPreparedFitData:
    r"""Prepare train/validation timelines via M0's own leak-safe pipeline.

    Purpose
    -------
    Reuse M0's audited discovery/inclusion-rule/leak-safety guarantees
    exactly (by calling its registered pipeline function with the real,
    proxied M0 registry), then rebuild this model's session tensors with
    M1's own frame adapter from the returned canonical timeline rather than
    reimplementing discovery or the inclusion rule.

    Constraints
    -----------
    Delegates every constraint (frozen split membership, legacy inclusion
    rule, one-mouse-per-run) to ``prepare_registered_fit_data``.

    Important edge cases
    --------------------
    A session with zero feedback-valid trials is valid and simply keeps a
    constant 0.5 prior throughout.

    Related tests
    -------------
    ``test_pure_wsls_pipeline_matches_m0_audit_and_builds_sessions``.
    """

    if registry.model_id != "Pure_WSLS_v1":
        raise ValueError(
            "prepare_registered_pure_wsls_fit_data requires a Pure_WSLS_v1 registry"
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

    return PureWslsPreparedFitData(
        training_sessions=tuple(training),
        validation_sessions=tuple(validation),
        canonical_trials=base.canonical_trials,
        audit=base.audit,
    )
