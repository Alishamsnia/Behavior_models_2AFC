"""Leak-safe M4 data preparation built on the unchanged M3 timeline contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m0_sensory_v1.pipeline import FitInputAudit
from m3_hmm_delta_v1.pipeline import prepare_registered_m3_fit_data

from .registry import M4HSMMRegistry
from .session import M4SessionTrials


@dataclass(frozen=True)
class M4PreparedFitData:
    training_sessions: tuple[M4SessionTrials, ...]
    validation_sessions: tuple[M4SessionTrials, ...]
    validation_block_labels: dict[str, tuple[object, ...]]
    canonical_trials: pd.DataFrame
    audit: FitInputAudit


def prepare_registered_m4_fit_data(
    raw_data_root: str | Path,
    registry: M4HSMMRegistry,
) -> M4PreparedFitData:
    """Prepare M4 inputs without introducing block or run-length labels."""

    if registry.model_id != "M4_hsmm_delta_v1":
        raise ValueError("prepare_registered_m4_fit_data requires an M4 registry")
    shared = prepare_registered_m3_fit_data(raw_data_root, registry.source_m3_registry)
    return M4PreparedFitData(
        training_sessions=shared.training_sessions,
        validation_sessions=shared.validation_sessions,
        validation_block_labels=shared.validation_block_labels,
        canonical_trials=shared.canonical_trials,
        audit=shared.audit,
    )
