"""Chronological sliding-window and single-session fitting glue for
M2_dynamic_previous_action_bias_v1. Full-session-matched workflow only; the
generic windowing mechanics live in ``m0_sliding_window_v1``."""

from .analysis import (
    FIGURE_ANALYSIS_SCHEMA_VERSION,
    GLOBAL_ANALYSIS_SCHEMA_VERSION,
    figure_outputs_are_current,
    figure_ready,
    generate_global_prediction_figures,
    generate_identifiability_outputs,
    generate_m2_dynamic_previous_action_bias_experiment_outputs,
    generate_truncation_outputs,
    identifiability_outputs_are_current,
    truncation_outputs_are_current,
)
from .experiment import (
    BlockPartition,
    FULL_SESSION_MATCHED_WORKFLOW,
    M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_NAMES,
    PreparedSlidingData,
    SlidingWindow,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    prediction_performance,
    prepare_sliding_data,
)

__all__ = [
    "BlockPartition",
    "FULL_SESSION_MATCHED_WORKFLOW",
    "FIGURE_ANALYSIS_SCHEMA_VERSION",
    "GLOBAL_ANALYSIS_SCHEMA_VERSION",
    "M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "figure_outputs_are_current",
    "figure_ready",
    "generate_global_prediction_figures",
    "generate_identifiability_outputs",
    "generate_m2_dynamic_previous_action_bias_experiment_outputs",
    "generate_truncation_outputs",
    "identifiability_outputs_are_current",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "prediction_performance",
    "prepare_sliding_data",
    "truncation_outputs_are_current",
]
