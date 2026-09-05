"""Additive implementation of the Phase-I constant-hazard delta learner.

This package intentionally imports the already verified M0/M1/M2 scientific
primitives without changing their modules or public behavior.
"""

from .parameters import (
    M3_PARAMETER_ORDER,
    M3DecodedParameters,
    decode_m3_raw_parameters,
    encode_m3_constrained_parameters,
)
from .session import (
    M3MouseObjectiveResult,
    M3Predictions,
    M3SessionEvaluation,
    M3SessionTrials,
    evaluate_m3_mouse_objective,
    evaluate_m3_session,
    m3_session_trials_from_frame,
)
from .registry import M3DeltaRegistry, load_m3_model_registry
from .pipeline import M3PreparedFitData, prepare_registered_m3_fit_data
from .fit import (
    M3FitRunResult,
    M3MultiStartFitResult,
    fit_m3_all_initializations,
    fit_m3_single_initialization,
)

__all__ = [
    "M3_PARAMETER_ORDER",
    "M3DecodedParameters",
    "M3MouseObjectiveResult",
    "M3Predictions",
    "M3SessionEvaluation",
    "M3SessionTrials",
    "M3DeltaRegistry",
    "M3PreparedFitData",
    "M3FitRunResult",
    "M3MultiStartFitResult",
    "decode_m3_raw_parameters",
    "encode_m3_constrained_parameters",
    "evaluate_m3_mouse_objective",
    "evaluate_m3_session",
    "m3_session_trials_from_frame",
    "load_m3_model_registry",
    "prepare_registered_m3_fit_data",
    "fit_m3_single_initialization",
    "fit_m3_all_initializations",
]
