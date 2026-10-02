"""Pure_WSLS-specific contracts for the isolated chronological sliding
experiment. The generic windowing mechanics are unchanged and reused from
``m0_sliding_window_v1.experiment``, exactly as
``m1_stimulus_kernel_sliding_window_v1.experiment`` reuses them.
"""

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


PURE_WSLS_PARAMETER_NAMES = ("ell_short", "ell_long")


def aggregate_pure_wsls_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
) -> pd.DataFrame:
    """Average Pure_WSLS winners for session fine-tuning initialization.

    Only used by the legacy_partitioned workflow -- the full_session_matched
    workflow uses the generic ``match_window_winners_to_test_sessions``
    directly instead. Both fitted coordinates are plain lapse probabilities
    with no positive-scale or logit-scale special case (unlike
    ``m1_stimulus_kernel_sliding_window_v1``'s ``kappa``/``a``), so a simple
    arithmetic mean across contributing windows is the correct average.
    """

    if not windows:
        raise ValueError("At least one completed Pure_WSLS window is required")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every Pure_WSLS window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(
        winner_parameters[final_window.window_id], dtype=float
    )
    if final_values.shape != (2,) or not np.all(np.isfinite(final_values)):
        raise ValueError("Pure_WSLS window winners must contain two finite parameters")

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
        if matrix.shape != (len(used), 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every Pure_WSLS winner must contain two finite parameters")

        mean = matrix.mean(axis=0)
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
                else "unweighted_mean_of_training_member_window_winners"
            ),
        }
        for index, name in enumerate(PURE_WSLS_PARAMETER_NAMES):
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
    "PURE_WSLS_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "WORKFLOW_CHOICES",
    "aggregate_pure_wsls_session_initializations",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
