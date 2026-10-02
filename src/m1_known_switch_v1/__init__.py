"""M1_known_switch_v1: M1's fixed true-rate prototypes and Bayesian sensory
readout, evaluated under the true (oracle) block-switch timing instead of a
fitted hazard. See ``docs/models/M1_known_switch_v1.md`` for the full
scientific contract.
"""

from .fit import (
    M1KnownSwitchEpochRecord,
    M1KnownSwitchFitRunResult,
    M1KnownSwitchMultiStartFitResult,
    fit_m1_known_switch_all_initializations,
    fit_m1_known_switch_single_initialization,
)
from .parameters import (
    M1_KNOWN_SWITCH_FIXED_PROTOTYPES,
    M1_KNOWN_SWITCH_LOG_LIKELIHOOD_RATIO,
    M1_KNOWN_SWITCH_PARAMETER_ORDER,
    decode_m1_known_switch_raw_parameters,
    encode_m1_known_switch_constrained_parameters,
)
from .pipeline import (
    M1KnownSwitchPreparedFitData,
    prepare_registered_m1_known_switch_fit_data,
)
from .registry import M1KnownSwitchRegistry, load_m1_known_switch_model_registry
from .session import (
    M1KnownSwitchMouseObjectiveResult,
    M1KnownSwitchPredictions,
    M1KnownSwitchSessionEvaluation,
    M1KnownSwitchSessionTrials,
    evaluate_m1_known_switch_mouse_objective,
    evaluate_m1_known_switch_session,
    m1_known_switch_session_trials_from_frame,
)

__all__ = [
    "M1_KNOWN_SWITCH_FIXED_PROTOTYPES",
    "M1_KNOWN_SWITCH_LOG_LIKELIHOOD_RATIO",
    "M1_KNOWN_SWITCH_PARAMETER_ORDER",
    "decode_m1_known_switch_raw_parameters",
    "encode_m1_known_switch_constrained_parameters",
    "M1KnownSwitchMouseObjectiveResult",
    "M1KnownSwitchPredictions",
    "M1KnownSwitchSessionEvaluation",
    "M1KnownSwitchSessionTrials",
    "evaluate_m1_known_switch_mouse_objective",
    "evaluate_m1_known_switch_session",
    "m1_known_switch_session_trials_from_frame",
    "M1KnownSwitchRegistry",
    "load_m1_known_switch_model_registry",
    "M1KnownSwitchPreparedFitData",
    "prepare_registered_m1_known_switch_fit_data",
    "M1KnownSwitchEpochRecord",
    "M1KnownSwitchFitRunResult",
    "M1KnownSwitchMultiStartFitResult",
    "fit_m1_known_switch_all_initializations",
    "fit_m1_known_switch_single_initialization",
]
