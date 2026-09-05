"""Public interface for canonical behavioral-data preparation."""

from .matlab import load_session_mat
from .preparation import (
    DataPreparationResult,
    SessionFileMetadata,
    has_required_isi_bins,
    parse_behavior_file_path,
    prepare_dataset,
    prepare_session,
    select_legacy_model_trials,
    states_labeling,
)
from .validation import validate_prepared_data
from .task_moments import TaskMoments, estimate_task_moments
from .splits import (
    DEFAULT_SPLIT_VERSION,
    create_chronological_splits,
    largest_remainder_counts,
    select_session_sources,
    validate_chronological_splits,
)

__all__ = [
    "DataPreparationResult",
    "SessionFileMetadata",
    "has_required_isi_bins",
    "load_session_mat",
    "parse_behavior_file_path",
    "prepare_dataset",
    "prepare_session",
    "select_legacy_model_trials",
    "states_labeling",
    "TaskMoments",
    "estimate_task_moments",
    "DEFAULT_SPLIT_VERSION",
    "create_chronological_splits",
    "largest_remainder_counts",
    "select_session_sources",
    "validate_chronological_splits",
    "validate_prepared_data",
]
