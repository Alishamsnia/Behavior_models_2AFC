"""M3-specific contracts for the isolated chronological sliding experiment."""

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
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    partition_session_blocks,
    prediction_performance,
    prepare_sliding_data,
)


M3_PARAMETER_NAMES = (
    "w",
    "s0",
    "ell_short",
    "ell_long",
    "kappa",
    "ah",
    "delta0",
    "alpha_theta",
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid stably for finite NumPy values."""

    return np.exp(-np.logaddexp(0.0, -values))


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability-scale M3 means must lie strictly inside (0,1)")
    return float(np.log(probability) - np.log1p(-probability))


def aggregate_m3_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
) -> pd.DataFrame:
    """Average M3 rolling winners on their interpretable constrained scales.

    Common sensory/lapse values and ``alpha_theta`` are averaged directly.
    ``ah`` is averaged through biased-hazard probability and ``delta0`` through
    the initial long-prototype probability. Sessions that never occur in a
    training window use the final chronological window winner.
    """

    if not windows:
        raise ValueError("At least one completed M3 window is required")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every M3 window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(
        winner_parameters[final_window.window_id], dtype=float
    )
    if final_values.shape != (8,) or not np.all(np.isfinite(final_values)):
        raise ValueError("M3 window winners must contain eight finite parameters")

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
        if matrix.shape != (len(used), 8) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every M3 winner must contain eight finite parameters")
        if np.any(matrix[:, 4] <= 0.0) or np.any(matrix[:, 6] <= 0.0):
            raise ValueError("Every M3 winner requires positive delta0")
        # alpha_theta is averaged on its raw constrained scale (never log-odds
        # transformed), so an exact 0.0 from float64 sigmoid underflow of a
        # genuinely converged, weakly-identified learning rate is a valid
        # "no learning" boundary value, not corrupted data. Only negative or
        # >=1 values -- impossible from the sigmoid decoder -- are rejected.
        if np.any((matrix[:, 7] < 0.0) | (matrix[:, 7] >= 1.0)):
            raise ValueError("Every M3 winner requires alpha_theta in [0,1)")

        mean = matrix.mean(axis=0)
        mean[4] = np.exp(np.log(matrix[:, 4]).mean())
        hazard_values = _sigmoid(matrix[:, 5])
        theta_long_values = _sigmoid(matrix[:, 6])
        mean_hazard = float(hazard_values.mean())
        mean_theta_long = float(theta_long_values.mean())
        mean[5] = _logit(mean_hazard)
        mean[6] = _logit(mean_theta_long)
        if mean[6] <= 0.0 or not 0.0 <= mean[7] < 1.0:
            raise ValueError("Mean M3 dynamic parameters violate their constraints")
        coordinate_sd = matrix.std(axis=0, ddof=0)
        alpha_values = matrix[:, 7]
        row: dict[str, object] = {
            "mouse_id": str(assignment.mouse_id),
            "session_id": session_id,
            "chronological_rank": int(assignment.chronological_rank),
            "contributing_window_ids": ";".join(str(item) for item in used),
            "n_contributing_windows": len(used),
            "initialization_policy": (
                "final_window_winner_fallback_no_training_membership"
                if fallback
                else "mean_common_hazard_initial_prototype_and_learning_rate"
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
            "initial_learning_rate": float(mean[7]),
            "window_sd_learning_rate": float(alpha_values.std(ddof=0)),
            "window_min_learning_rate": float(alpha_values.min()),
            "window_max_learning_rate": float(alpha_values.max()),
        }
        for index, name in enumerate(M3_PARAMETER_NAMES):
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
    "M3_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "WORKFLOW_CHOICES",
    "aggregate_m3_session_initializations",
    "build_sliding_windows",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
