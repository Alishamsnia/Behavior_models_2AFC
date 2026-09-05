"""M2-specific contracts for the isolated chronological sliding experiment."""

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


M2_PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid stably for finite NumPy values."""

    return np.exp(-np.logaddexp(0.0, -values))


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability-scale M2 means must lie strictly inside (0,1)")
    return float(np.log(probability) - np.log1p(-probability))


def aggregate_m2_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
) -> pd.DataFrame:
    """Average M2 winners on interpretable scales for session initialization.

    Common sensory/lapse coordinates are averaged directly. The biased hazard
    and long subjective prototype are averaged as probabilities, then converted
    back to ``ah`` and positive ``delta0`` respectively. Tail sessions without
    training-window membership use the final chronological winner.
    """

    if not windows:
        raise ValueError("At least one completed M2 window is required")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every M2 window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(
        winner_parameters[final_window.window_id], dtype=float
    )
    if final_values.shape != (7,) or not np.all(np.isfinite(final_values)):
        raise ValueError("M2 window winners must contain seven finite parameters")

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
        if matrix.shape != (len(used), 7) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every M2 winner must contain seven finite parameters")
        if np.any(matrix[:, 4] <= 0.0) or np.any(matrix[:, 6] <= 0.0):
            raise ValueError("Every M2 winner requires positive delta0")

        mean = matrix.mean(axis=0)
        mean[4] = np.exp(np.log(matrix[:, 4]).mean())
        hazard_values = _sigmoid(matrix[:, 5])
        theta_long_values = _sigmoid(matrix[:, 6])
        mean_hazard = float(hazard_values.mean())
        mean_theta_long = float(theta_long_values.mean())
        mean[5] = _logit(mean_hazard)
        mean[6] = _logit(mean_theta_long)
        if mean[6] <= 0.0:
            raise ValueError("Mean M2 subjective contrast must remain positive")
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
                else "mean_common_parameters_hazard_and_subjective_prototype"
            ),
            "initial_hazard": mean_hazard,
            "window_sd_hazard": float(hazard_values.std(ddof=0)),
            "window_min_hazard": float(hazard_values.min()),
            "window_max_hazard": float(hazard_values.max()),
            "initial_theta_N": 0.5,
            "initial_theta_S": 1.0 - mean_theta_long,
            "initial_theta_L": mean_theta_long,
            "window_sd_theta_L": float(theta_long_values.std(ddof=0)),
            "window_min_theta_L": float(theta_long_values.min()),
            "window_max_theta_L": float(theta_long_values.max()),
        }
        for index, name in enumerate(M2_PARAMETER_NAMES):
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
    "M2_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "WORKFLOW_CHOICES",
    "aggregate_m2_session_initializations",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
