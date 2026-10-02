"""M1_action_kernel-specific contracts for the isolated chronological
sliding experiment. The generic windowing mechanics are unchanged and reused
from ``m0_sliding_window_v1.experiment``."""

from __future__ import annotations

from m0_sliding_window_v1.experiment import (
    BlockPartition,
    FULL_SESSION_MATCHED_WORKFLOW,
    PreparedSlidingData,
    SlidingWindow,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    prediction_performance,
    prepare_full_session_matched_data as prepare_sliding_data,
)


M1_ACTION_KERNEL_PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa", "a")


__all__ = [
    "BlockPartition",
    "FULL_SESSION_MATCHED_WORKFLOW",
    "M1_ACTION_KERNEL_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
