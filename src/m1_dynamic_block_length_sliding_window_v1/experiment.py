"""M1_dynamic_block_length-specific contracts for the isolated chronological
sliding experiment. The generic windowing mechanics are unchanged and reused
from ``m0_sliding_window_v1.experiment``."""

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


M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "bh"
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -values))


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability-scale M1_dynamic_block_length means must lie strictly inside (0,1)")
    return float(np.log(probability) - np.log1p(-probability))


def aggregate_m1_dynamic_block_length_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
    *,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
) -> pd.DataFrame:
    """Average M1_dynamic_block_length winners on interpretable constrained
    coordinates (only used by the legacy_partitioned workflow -- the
    full_session_matched workflow uses the generic
    ``match_window_winners_to_test_sessions`` directly instead)."""

    if not windows:
        raise ValueError("At least one completed M1_dynamic_block_length window is required")
    if run_length_center <= 0 or maximum_run_length < run_length_center:
        raise ValueError("M1_dynamic_block_length run-length support is invalid")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every M1_dynamic_block_length window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(winner_parameters[final_window.window_id], dtype=float)
    if final_values.shape != (7,) or not np.all(np.isfinite(final_values)):
        raise ValueError("M1_dynamic_block_length window winners must contain seven finite parameters")

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
        if matrix.shape != (len(used), 7) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every M1_dynamic_block_length winner must contain seven finite parameters")
        if np.any(matrix[:, 4] <= 0.0) or np.any(matrix[:, 6] <= 0.0):
            raise ValueError("Every M1_dynamic_block_length winner requires positive kappa and bh")

        mean = matrix.mean(axis=0)
        mean[4] = np.exp(np.log(matrix[:, 4]).mean())
        center_hazards = _sigmoid(matrix[:, 5])
        mean_center_hazard = float(center_hazards.mean())
        mean[5] = _logit(mean_center_hazard)
        if mean[6] <= 0.0:
            raise ValueError("Mean M1_dynamic_block_length bh violates its constraint")

        run_one_hazards = _sigmoid(matrix[:, 5] + matrix[:, 6] * (1.0 - run_length_center))
        tail_hazards = _sigmoid(matrix[:, 5] + matrix[:, 6] * (maximum_run_length - run_length_center))
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
                "mean_common_center_hazard_and_slope_fixed_true_rate_prototypes"
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
            "initial_theta_S": 0.2,
            "initial_theta_L": 0.8,
        }
        for index, name in enumerate(M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES):
            row[f"initial_{name}"] = float(mean[index])
            row[f"window_sd_{name}"] = float(coordinate_sd[index])
            row[f"window_min_{name}"] = float(matrix[:, index].min())
            row[f"window_max_{name}"] = float(matrix[:, index].max())
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BlockPartition", "FULL_SESSION_MATCHED_WORKFLOW", "LEGACY_PARTITIONED_WORKFLOW",
    "M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES", "PreparedSlidingData", "SlidingWindow",
    "WORKFLOW_CHOICES", "aggregate_m1_dynamic_block_length_session_initializations",
    "build_sliding_windows", "match_window_winners_to_test_sessions",
    "partition_final_two_blocks", "partition_session_blocks",
    "prediction_performance", "prepare_sliding_data",
]
