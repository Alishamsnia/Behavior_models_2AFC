"""Pure_WSLS_v1: a literal win-stay/lose-shift-on-category rule.

No sensory evidence enters this model at all: the belief pi_t is a scalar
updated only from the previous trial's true category (alpha fixed at
exactly 1.0, so pi_t = C_{t-1} exactly), then mixed with the same
category-dependent lapse rate used by every other family in this
repository. See ``wsls_core_v1`` for the shared recurrence this model
wraps, and ``extended_wsls_v1`` for the sibling model that fits alpha.
"""

from .analysis import evaluate_pure_wsls_split_metrics
from .fit import (
    PureWslsEpochRecord,
    PureWslsFitRunResult,
    PureWslsMultiStartFitResult,
    fit_pure_wsls_all_initializations,
    fit_pure_wsls_single_initialization,
)
from .initialization import (
    PureWslsInitializationSpec,
    validate_pure_wsls_initialization_table,
)
from .parameters import (
    PURE_WSLS_FIXED_ALPHA,
    PURE_WSLS_PARAMETER_ORDER,
    PureWslsDecodedParameters,
    decode_pure_wsls_raw_parameters,
    encode_pure_wsls_constrained_parameters,
)
from .pipeline import (
    PureWslsPreparedFitData,
    prepare_registered_pure_wsls_fit_data,
)
from .registry import PureWslsRegistry, load_pure_wsls_model_registry
from .session import (
    M1SessionTrials,
    PureWslsMouseObjectiveResult,
    PureWslsSessionEvaluation,
    evaluate_pure_wsls_mouse_objective,
    evaluate_pure_wsls_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "PURE_WSLS_FIXED_ALPHA",
    "PURE_WSLS_PARAMETER_ORDER",
    "decode_pure_wsls_raw_parameters",
    "encode_pure_wsls_constrained_parameters",
    "PureWslsDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "PureWslsMouseObjectiveResult",
    "PureWslsSessionEvaluation",
    "evaluate_pure_wsls_mouse_objective",
    "evaluate_pure_wsls_session",
    "PureWslsRegistry",
    "load_pure_wsls_model_registry",
    "PureWslsPreparedFitData",
    "prepare_registered_pure_wsls_fit_data",
    "PureWslsInitializationSpec",
    "validate_pure_wsls_initialization_table",
    "PureWslsEpochRecord",
    "PureWslsFitRunResult",
    "PureWslsMultiStartFitResult",
    "fit_pure_wsls_all_initializations",
    "fit_pure_wsls_single_initialization",
    "evaluate_pure_wsls_split_metrics",
]
