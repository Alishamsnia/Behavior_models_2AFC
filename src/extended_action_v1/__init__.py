"""Extended_action_v1: a fitted-learning-rate action-history rule (actionHistory_lapse).

No sensory evidence enters this model at all: the belief pi_t is a scalar
updated only from the mouse's own previous valid choices, with a fitted
learning rate alpha=sigmoid(a), then mixed with the same category-dependent
lapse rate used by every other family in this repository. See
``action_core_v1`` for the shared recurrence this model wraps, and
``pure_action_v1`` for the sibling model that fixes alpha at exactly 1.0.
The action-driven sibling of ``extended_wsls_v1``.
"""

from .analysis import evaluate_extended_action_split_metrics
from .fit import (
    ExtendedActionEpochRecord,
    ExtendedActionFitRunResult,
    ExtendedActionMultiStartFitResult,
    fit_extended_action_all_initializations,
    fit_extended_action_single_initialization,
)
from .initialization import (
    ExtendedActionInitializationSpec,
    validate_extended_action_initialization_table,
)
from .parameters import (
    EXTENDED_ACTION_PARAMETER_ORDER,
    ExtendedActionDecodedParameters,
    decode_extended_action_raw_parameters,
    encode_extended_action_constrained_parameters,
)
from .pipeline import (
    ExtendedActionPreparedFitData,
    prepare_registered_extended_action_fit_data,
)
from .registry import ExtendedActionRegistry, load_extended_action_model_registry
from .session import (
    M1SessionTrials,
    ExtendedActionMouseObjectiveResult,
    ExtendedActionSessionEvaluation,
    evaluate_extended_action_mouse_objective,
    evaluate_extended_action_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "EXTENDED_ACTION_PARAMETER_ORDER",
    "decode_extended_action_raw_parameters",
    "encode_extended_action_constrained_parameters",
    "ExtendedActionDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "ExtendedActionMouseObjectiveResult",
    "ExtendedActionSessionEvaluation",
    "evaluate_extended_action_mouse_objective",
    "evaluate_extended_action_session",
    "ExtendedActionRegistry",
    "load_extended_action_model_registry",
    "ExtendedActionPreparedFitData",
    "prepare_registered_extended_action_fit_data",
    "ExtendedActionInitializationSpec",
    "validate_extended_action_initialization_table",
    "ExtendedActionEpochRecord",
    "ExtendedActionFitRunResult",
    "ExtendedActionMultiStartFitResult",
    "fit_extended_action_all_initializations",
    "fit_extended_action_single_initialization",
    "evaluate_extended_action_split_metrics",
]
