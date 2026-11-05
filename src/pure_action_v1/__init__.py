"""Pure_action_v1: a literal repeat-the-last-valid-choice rule (actionSingle_lapse).

No sensory evidence enters this model at all: the state pi_t is a scalar
updated only from the mouse's own last valid choice (alpha fixed at
exactly 1.0, so pi_t = A_{t'} exactly, t' the last valid choice before t),
then mixed with the same category-dependent lapse rate used by every
other family in this repository. The action-driven sibling of
``pure_wsls_v1``. See ``action_core_v1`` for the shared recurrence this
model wraps, and ``extended_action_v1`` for the sibling model that fits alpha.
"""

from .analysis import evaluate_pure_action_split_metrics
from .fit import (
    PureActionEpochRecord,
    PureActionFitRunResult,
    PureActionMultiStartFitResult,
    fit_pure_action_all_initializations,
    fit_pure_action_single_initialization,
)
from .initialization import (
    PureActionInitializationSpec,
    validate_pure_action_initialization_table,
)
from .parameters import (
    PURE_ACTION_FIXED_ALPHA,
    PURE_ACTION_PARAMETER_ORDER,
    PureActionDecodedParameters,
    decode_pure_action_raw_parameters,
    encode_pure_action_constrained_parameters,
)
from .pipeline import (
    PureActionPreparedFitData,
    prepare_registered_pure_action_fit_data,
)
from .registry import PureActionRegistry, load_pure_action_model_registry
from .session import (
    M1SessionTrials,
    PureActionMouseObjectiveResult,
    PureActionSessionEvaluation,
    evaluate_pure_action_mouse_objective,
    evaluate_pure_action_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "PURE_ACTION_FIXED_ALPHA",
    "PURE_ACTION_PARAMETER_ORDER",
    "decode_pure_action_raw_parameters",
    "encode_pure_action_constrained_parameters",
    "PureActionDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "PureActionMouseObjectiveResult",
    "PureActionSessionEvaluation",
    "evaluate_pure_action_mouse_objective",
    "evaluate_pure_action_session",
    "PureActionRegistry",
    "load_pure_action_model_registry",
    "PureActionPreparedFitData",
    "prepare_registered_pure_action_fit_data",
    "PureActionInitializationSpec",
    "validate_pure_action_initialization_table",
    "PureActionEpochRecord",
    "PureActionFitRunResult",
    "PureActionMultiStartFitResult",
    "fit_pure_action_all_initializations",
    "fit_pure_action_single_initialization",
    "evaluate_pure_action_split_metrics",
]
