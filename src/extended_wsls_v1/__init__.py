"""Extended_WSLS_v1: a fitted-learning-rate win-stay/lose-shift-on-category rule.

No sensory evidence enters this model at all: the belief pi_t is a scalar
updated only from the previous trial's true category, with a fitted
learning rate alpha=sigmoid(a), then mixed with the same category-dependent
lapse rate used by every other family in this repository. See
``wsls_core_v1`` for the shared recurrence this model wraps, and
``pure_wsls_v1`` for the sibling model that fixes alpha at exactly 1.0.
"""

from .analysis import evaluate_extended_wsls_split_metrics
from .fit import (
    ExtendedWslsEpochRecord,
    ExtendedWslsFitRunResult,
    ExtendedWslsMultiStartFitResult,
    fit_extended_wsls_all_initializations,
    fit_extended_wsls_single_initialization,
)
from .initialization import (
    ExtendedWslsInitializationSpec,
    validate_extended_wsls_initialization_table,
)
from .parameters import (
    EXTENDED_WSLS_PARAMETER_ORDER,
    ExtendedWslsDecodedParameters,
    decode_extended_wsls_raw_parameters,
    encode_extended_wsls_constrained_parameters,
)
from .pipeline import (
    ExtendedWslsPreparedFitData,
    prepare_registered_extended_wsls_fit_data,
)
from .registry import ExtendedWslsRegistry, load_extended_wsls_model_registry
from .session import (
    M1SessionTrials,
    ExtendedWslsMouseObjectiveResult,
    ExtendedWslsSessionEvaluation,
    evaluate_extended_wsls_mouse_objective,
    evaluate_extended_wsls_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "EXTENDED_WSLS_PARAMETER_ORDER",
    "decode_extended_wsls_raw_parameters",
    "encode_extended_wsls_constrained_parameters",
    "ExtendedWslsDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "ExtendedWslsMouseObjectiveResult",
    "ExtendedWslsSessionEvaluation",
    "evaluate_extended_wsls_mouse_objective",
    "evaluate_extended_wsls_session",
    "ExtendedWslsRegistry",
    "load_extended_wsls_model_registry",
    "ExtendedWslsPreparedFitData",
    "prepare_registered_extended_wsls_fit_data",
    "ExtendedWslsInitializationSpec",
    "validate_extended_wsls_initialization_table",
    "ExtendedWslsEpochRecord",
    "ExtendedWslsFitRunResult",
    "ExtendedWslsMultiStartFitResult",
    "fit_extended_wsls_all_initializations",
    "fit_extended_wsls_single_initialization",
    "evaluate_extended_wsls_split_metrics",
]
