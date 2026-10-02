"""M2_known_switch_v1: M2's fitted static subjective prototypes and Bayesian
sensory readout, evaluated under the true (oracle) block-switch timing
instead of a fitted hazard. See ``docs/models/M2_known_switch_v1.md`` for the
full scientific contract.
"""

from .fit import (
    M2KnownSwitchEpochRecord,
    M2KnownSwitchFitRunResult,
    M2KnownSwitchMultiStartFitResult,
    fit_m2_known_switch_all_initializations,
    fit_m2_known_switch_single_initialization,
)
from .parameters import (
    M2_KNOWN_SWITCH_PARAMETER_ORDER,
    M2KnownSwitchParameters,
    decode_m2_known_switch_raw_parameters,
    encode_m2_known_switch_constrained_parameters,
)
from .pipeline import (
    M2KnownSwitchPreparedFitData,
    prepare_registered_m2_known_switch_fit_data,
)
from .registry import (
    M2KnownSwitchInitializationSpec,
    M2KnownSwitchRegistry,
    load_m2_known_switch_model_registry,
)
from .session import (
    M2KnownSwitchMouseObjectiveResult,
    M2KnownSwitchPredictions,
    M2KnownSwitchSessionEvaluation,
    M2KnownSwitchSessionTrials,
    evaluate_m2_known_switch_mouse_objective,
    evaluate_m2_known_switch_session,
    m2_known_switch_session_trials_from_frame,
)

__all__ = [
    "M2_KNOWN_SWITCH_PARAMETER_ORDER",
    "M2KnownSwitchParameters",
    "decode_m2_known_switch_raw_parameters",
    "encode_m2_known_switch_constrained_parameters",
    "M2KnownSwitchMouseObjectiveResult",
    "M2KnownSwitchPredictions",
    "M2KnownSwitchSessionEvaluation",
    "M2KnownSwitchSessionTrials",
    "evaluate_m2_known_switch_mouse_objective",
    "evaluate_m2_known_switch_session",
    "m2_known_switch_session_trials_from_frame",
    "M2KnownSwitchInitializationSpec",
    "M2KnownSwitchRegistry",
    "load_m2_known_switch_model_registry",
    "M2KnownSwitchPreparedFitData",
    "prepare_registered_m2_known_switch_fit_data",
    "M2KnownSwitchEpochRecord",
    "M2KnownSwitchFitRunResult",
    "M2KnownSwitchMultiStartFitResult",
    "fit_m2_known_switch_all_initializations",
    "fit_m2_known_switch_single_initialization",
]
