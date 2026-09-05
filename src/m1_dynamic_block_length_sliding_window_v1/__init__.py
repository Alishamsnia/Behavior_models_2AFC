"""Sliding-window and session-fine-tuning experiment for unchanged
M1_dynamic_block_length_v1."""

from .analysis import (
    FIGURE_ANALYSIS_SCHEMA_VERSION,
    figure_outputs_are_current,
    figure_ready,
    generate_global_prediction_figures,
    generate_identifiability_outputs,
    generate_m1_dynamic_block_length_experiment_outputs,
    generate_truncation_outputs,
    identifiability_outputs_are_current,
    truncation_outputs_are_current,
)
from .experiment import (
    BlockPartition,
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES,
    PreparedSlidingData,
    SlidingWindow,
    WORKFLOW_CHOICES,
    aggregate_m1_dynamic_block_length_session_initializations,
    build_sliding_windows,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    partition_session_blocks,
    prediction_performance,
    prepare_sliding_data,
)

__all__ = [
    "FIGURE_ANALYSIS_SCHEMA_VERSION", "BlockPartition",
    "FULL_SESSION_MATCHED_WORKFLOW", "LEGACY_PARTITIONED_WORKFLOW",
    "M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES",
    "PreparedSlidingData", "SlidingWindow", "aggregate_m1_dynamic_block_length_session_initializations",
    "WORKFLOW_CHOICES", "build_sliding_windows", "figure_outputs_are_current", "figure_ready",
    "generate_global_prediction_figures", "generate_identifiability_outputs",
    "generate_m1_dynamic_block_length_experiment_outputs", "generate_truncation_outputs",
    "identifiability_outputs_are_current", "match_window_winners_to_test_sessions",
    "partition_final_two_blocks", "partition_session_blocks",
    "prediction_performance", "prepare_sliding_data", "truncation_outputs_are_current",
]
