"""M1-specific contracts for the isolated chronological sliding experiment."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from m0_sliding_window_v1.experiment import (
    BlockPartition,
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    PreparedSlidingData,
    SlidingWindow,
    WORKFLOW_CHOICES,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    partition_session_blocks,
    prediction_performance,
    prepare_sliding_data,
)


M1_PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa", "ah")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid stably for finite NumPy values."""

    return np.exp(-np.logaddexp(0.0, -values))


def aggregate_m1_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
) -> pd.DataFrame:
    """Average M1 winners for session fine-tuning initialization.

    The sensory/lapse coordinates are averaged on their constrained scales.
    The transition coordinate is averaged as the interpretable biased hazard
    ``h_B=sigmoid(ah)`` and transformed back to ``ah`` once for initialization.
    Tail sessions without training-window membership use the last window winner.
    """

    if not windows:
        raise ValueError("At least one completed M1 window is required")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every M1 window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(
        winner_parameters[final_window.window_id], dtype=float
    )
    if final_values.shape != (6,) or not np.all(np.isfinite(final_values)):
        raise ValueError("M1 window winners must contain six finite parameters")

    rows: list[dict[str, object]] = []
    for assignment in assignments.sort_values("chronological_rank").itertuples(
        index=False
    ):
        session_id = str(assignment.session_id)
        contributors = [
            window.window_id
            for window in windows
            if session_id in window.training_session_ids
        ]
        fallback = not contributors
        used = contributors or [final_window.window_id]
        matrix = np.asarray([winner_parameters[item] for item in used], dtype=float)
        if matrix.shape != (len(used), 6) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every M1 winner must contain six finite parameters")
        if np.any(matrix[:, 4] <= 0.0):
            raise ValueError("Every M1 winner requires positive kappa")

        mean = matrix.mean(axis=0)
        mean[4] = np.exp(np.log(matrix[:, 4]).mean())
        hazard_values = _sigmoid(matrix[:, 5])
        mean_hazard = float(hazard_values.mean())
        if not 0.0 < mean_hazard < 1.0:
            raise ValueError("Mean M1 biased hazard must lie strictly inside (0,1)")
        mean[5] = np.log(mean_hazard) - np.log1p(-mean_hazard)
        coordinate_sd = matrix.std(axis=0, ddof=0)
        row: dict[str, object] = {
            "mouse_id": str(assignment.mouse_id),
            "session_id": session_id,
            "chronological_rank": int(assignment.chronological_rank),
            "contributing_window_ids": ";".join(str(item) for item in used),
            "n_contributing_windows": len(used),
            "initialization_policy": (
                "final_window_winner_fallback_no_training_membership"
                if fallback
                else "mean_constrained_parameters_and_probability_scale_hazard"
            ),
            "initial_hazard": mean_hazard,
            "window_sd_hazard": float(hazard_values.std(ddof=0)),
            "window_min_hazard": float(hazard_values.min()),
            "window_max_hazard": float(hazard_values.max()),
        }
        for index, name in enumerate(M1_PARAMETER_NAMES):
            row[f"initial_{name}"] = float(mean[index])
            row[f"window_sd_{name}"] = float(coordinate_sd[index])
            row[f"window_min_{name}"] = float(matrix[:, index].min())
            row[f"window_max_{name}"] = float(matrix[:, index].max())
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BlockPartition",
    "FULL_SESSION_MATCHED_WORKFLOW",
    "LEGACY_PARTITIONED_WORKFLOW",
    "M1_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "WORKFLOW_CHOICES",
    "aggregate_m1_session_initializations",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
