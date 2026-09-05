"""M4-specific contracts for the isolated chronological sliding experiment."""

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


M4_PARAMETER_NAMES = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "alpha_theta", "bh"
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -values))


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability-scale M4 means must lie strictly inside (0,1)")
    return float(np.log(probability) - np.log1p(-probability))


def aggregate_m4_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
    *,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
) -> pd.DataFrame:
    """Average M4 winners on interpretable constrained coordinates.

    The center hazard and initial long prototype are averaged as probabilities;
    the positive log-hazard slope ``bh`` and learning rate are averaged on their
    constrained scales. Tail sessions without training-window membership use
    the final chronological winner.
    """

    if not windows:
        raise ValueError("At least one completed M4 window is required")
    if run_length_center <= 0 or maximum_run_length < run_length_center:
        raise ValueError("M4 run-length support is invalid")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every M4 window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(winner_parameters[final_window.window_id], dtype=float)
    if final_values.shape != (9,) or not np.all(np.isfinite(final_values)):
        raise ValueError("M4 window winners must contain nine finite parameters")

    rows: list[dict[str, object]] = []
    for assignment in assignments.sort_values("chronological_rank").itertuples(index=False):
        session_id = str(assignment.session_id)
        contributors = [
            window.window_id for window in windows
            if session_id in window.training_session_ids
        ]
        fallback = not contributors
        used = contributors or [final_window.window_id]
        matrix = np.asarray([winner_parameters[item] for item in used], dtype=float)
        if matrix.shape != (len(used), 9) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every M4 winner must contain nine finite parameters")
        if np.any(matrix[:, 4] <= 0.0) or np.any(matrix[:, 6] <= 0.0) or np.any(matrix[:, 8] <= 0.0):
            raise ValueError("Every M4 winner requires positive delta0 and bh")
        # alpha_theta is averaged on its raw constrained scale (never log-odds
        # transformed), so an exact 0.0 from float64 sigmoid underflow of a
        # genuinely converged, weakly-identified learning rate is a valid
        # "no learning" boundary value, not corrupted data. Only negative or
        # >=1 values -- impossible from the sigmoid decoder -- are rejected.
        if np.any((matrix[:, 7] < 0.0) | (matrix[:, 7] >= 1.0)):
            raise ValueError("Every M4 winner requires alpha_theta in [0,1)")

        mean = matrix.mean(axis=0)
        mean[4] = np.exp(np.log(matrix[:, 4]).mean())
        center_hazards = _sigmoid(matrix[:, 5])
        theta_long_values = _sigmoid(matrix[:, 6])
        mean_center_hazard = float(center_hazards.mean())
        mean_theta_long = float(theta_long_values.mean())
        mean[5] = _logit(mean_center_hazard)
        mean[6] = _logit(mean_theta_long)
        if mean[6] <= 0.0 or not 0.0 <= mean[7] < 1.0 or mean[8] <= 0.0:
            raise ValueError("Mean M4 dynamic parameters violate their constraints")

        run_one_hazards = _sigmoid(
            matrix[:, 5] + matrix[:, 8] * (1.0 - run_length_center)
        )
        tail_hazards = _sigmoid(
            matrix[:, 5]
            + matrix[:, 8] * (maximum_run_length - run_length_center)
        )
        coordinate_sd = matrix.std(axis=0, ddof=0)
        row: dict[str, object] = {
            "mouse_id": str(assignment.mouse_id),
            "session_id": session_id,
            "chronological_rank": int(assignment.chronological_rank),
            "contributing_window_ids": ";".join(str(item) for item in used),
            "n_contributing_windows": len(used),
            "initialization_policy": (
                "final_window_winner_fallback_no_training_membership"
                if fallback else
                "mean_common_center_hazard_initial_prototype_learning_rate_and_slope"
            ),
            "initial_biased_hazard_at_center": mean_center_hazard,
            "window_sd_biased_hazard_at_center": float(center_hazards.std(ddof=0)),
            "window_min_biased_hazard_at_center": float(center_hazards.min()),
            "window_max_biased_hazard_at_center": float(center_hazards.max()),
            "window_mean_biased_hazard_r1": float(run_one_hazards.mean()),
            "window_min_biased_hazard_r1": float(run_one_hazards.min()),
            "window_max_biased_hazard_r1": float(run_one_hazards.max()),
            "window_mean_biased_hazard_at_tail": float(tail_hazards.mean()),
            "window_min_biased_hazard_at_tail": float(tail_hazards.min()),
            "window_max_biased_hazard_at_tail": float(tail_hazards.max()),
            "initial_theta_N": 0.5,
            "initial_theta_S": 1.0 - mean_theta_long,
            "initial_theta_L": mean_theta_long,
            "initial_learning_rate": float(mean[7]),
            "initial_log_hazard_slope": float(mean[8]),
        }
        for index, name in enumerate(M4_PARAMETER_NAMES):
            row[f"initial_{name}"] = float(mean[index])
            row[f"window_sd_{name}"] = float(coordinate_sd[index])
            row[f"window_min_{name}"] = float(matrix[:, index].min())
            row[f"window_max_{name}"] = float(matrix[:, index].max())
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BlockPartition", "FULL_SESSION_MATCHED_WORKFLOW",
    "LEGACY_PARTITIONED_WORKFLOW", "M4_PARAMETER_NAMES", "PreparedSlidingData",
    "SlidingWindow", "aggregate_m4_session_initializations",
    "WORKFLOW_CHOICES", "build_sliding_windows",
    "match_window_winners_to_test_sessions", "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance", "prepare_sliding_data",
]
