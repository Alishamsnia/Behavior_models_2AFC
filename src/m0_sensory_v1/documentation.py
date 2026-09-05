"""Automated Part VI scientific-docstring contract checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect


REQUIRED_SCIENTIFIC_DOCSTRING_SECTIONS = (
    "Purpose",
    "Mathematical definition",
    "Parameter interpretation",
    "Input units and shapes",
    "Returned quantities and shapes",
    "Constraints",
    "Numerical stability",
    "Causal restrictions",
    "Important edge cases",
    "Related tests",
)


def registered_scientific_functions() -> dict[str, Callable]:
    """Return every public scientific function governed by Part VI."""

    from .model import (  # Imported lazily to avoid package import cycles.
        bayesian_choice_probability,
        calculate_evaluation_metrics,
        category_variances,
        decode_raw_parameters,
        encode_constrained_parameters,
        evaluate_mouse_objective,
        evaluate_session,
        evaluate_split_metrics,
        fit_all_initializations,
        fit_single_initialization,
        fit_single_initialization_with_strategy,
        fixed_neutral_prior,
        lapse_mixture,
        log_odds_coefficients,
        positive_decision_threshold,
        predict_m0_trials,
        predict_m0_trials_with_parameters,
        sensory_standard_deviation,
        session_trials_from_frame,
        simulate_m0_sessions,
        solve_positive_decision_region,
        select_winning_start,
        run_parameter_recovery_case,
        validate_initialization_table,
    )
    from .pipeline import (
        discover_registered_session_paths,
        prepare_registered_fit_data,
    )
    from .m1_registry import load_m1_model_registry
    from .m2_registry import load_m2_model_registry
    from .model import (
        constant_hazard_transition_matrix,
        decode_m1_raw_parameters,
        encode_m1_constrained_parameters,
        evaluate_m1_mouse_objective,
        evaluate_m1_session,
        integrate_general_decision_region,
        m1_bayesian_choice_probability,
        m1_session_trials_from_frame,
        decode_m2_raw_parameters,
        encode_m2_constrained_parameters,
        evaluate_m2_hazard_contrast_surface,
        evaluate_m2_mouse_objective,
        evaluate_m2_session,
        m2_session_trials_from_frame,
        run_m2_real_design_recovery_case,
        run_m2_recovery_case,
        simulate_m2_choices,
        simulate_m2_hmm_sessions,
        simulate_m2_real_design_sessions,
        validate_m2_initialization_table,
    )
    from .model.m2_parameters import M2DecodedParameters

    functions = (
        decode_raw_parameters,
        encode_constrained_parameters,
        category_variances,
        log_odds_coefficients,
        solve_positive_decision_region,
        positive_decision_threshold,
        sensory_standard_deviation,
        bayesian_choice_probability,
        lapse_mixture,
        fixed_neutral_prior,
        predict_m0_trials_with_parameters,
        predict_m0_trials,
        session_trials_from_frame,
        evaluate_session,
        evaluate_mouse_objective,
        validate_initialization_table,
        fit_single_initialization,
        fit_single_initialization_with_strategy,
        fit_all_initializations,
        select_winning_start,
        calculate_evaluation_metrics,
        evaluate_split_metrics,
        simulate_m0_sessions,
        run_parameter_recovery_case,
        discover_registered_session_paths,
        prepare_registered_fit_data,
        decode_m1_raw_parameters,
        encode_m1_constrained_parameters,
        integrate_general_decision_region,
        m1_bayesian_choice_probability,
        m1_session_trials_from_frame,
        constant_hazard_transition_matrix,
        evaluate_m1_session,
        evaluate_m1_mouse_objective,
        load_m1_model_registry,
        M2DecodedParameters.prototypes,
        decode_m2_raw_parameters,
        encode_m2_constrained_parameters,
        validate_m2_initialization_table,
        m2_session_trials_from_frame,
        evaluate_m2_session,
        evaluate_m2_mouse_objective,
        simulate_m2_choices,
        simulate_m2_hmm_sessions,
        simulate_m2_real_design_sessions,
        run_m2_recovery_case,
        run_m2_real_design_recovery_case,
        evaluate_m2_hazard_contrast_surface,
        load_m2_model_registry,
    )
    return {function.__name__: function for function in functions}


def audit_scientific_docstrings(
    functions: Mapping[str, Callable] | None = None,
) -> tuple[str, ...]:
    """Return Part VI violations, or an empty tuple when all checks pass."""

    selected = registered_scientific_functions() if functions is None else functions
    violations: list[str] = []
    for name, function in selected.items():
        docstring = inspect.getdoc(function) or ""
        for section in REQUIRED_SCIENTIFIC_DOCSTRING_SECTIONS:
            if section not in docstring:
                violations.append(f"{name}: missing section {section!r}")
        if ".. math::" not in docstring:
            violations.append(f"{name}: missing Sphinx-compatible '.. math::' equation")
    return tuple(violations)
