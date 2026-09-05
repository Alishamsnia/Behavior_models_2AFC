"""Design-preserving parameter and latent-trajectory recovery for M4."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from m0_sensory_v1.model.fit import LBFGSSettings
from m3_hmm_delta_v1.fit import fit_m3_all_initializations
from m3_hmm_delta_v1.session import evaluate_m3_mouse_objective

from .fit import M4MultiStartFitResult, fit_m4_all_initializations
from .parameters import decode_m4_raw_parameters, encode_m4_constrained_parameters
from .session import (
    M4DecodedParameters,
    M4SessionTrials,
    evaluate_m4_mouse_objective,
    evaluate_m4_session,
)
from .simulation import simulate_m4_choices


M4_RECOVERY_ALPHA_GRID = (0.0, 0.02, 0.05, 0.10)
M4_RECOVERY_DELTA_GRID = (0.20, float(np.log(4.0)), 2.20)
M4_RECOVERY_HAZARD_AT_CENTER_GRID = (0.02, 1.0 / 23.0, 0.08)
M4_RECOVERY_BH_GRID = (0.0, 0.05, 0.10, 0.20, 0.40)


@dataclass(frozen=True)
class M4TrajectoryRecoveryMetrics:
    trial_count: int
    delta_rmse: float
    expected_run_length_rmse: float
    expected_hazard_rmse: float
    tail_probability_rmse: float
    block_probability_rmse: float
    generating_projection_fraction: float
    recovered_projection_fraction: float


@dataclass(frozen=True)
class M4RecoveryResult:
    generating_parameters: tuple[float, ...]
    recovered_parameters: tuple[float, ...]
    parameter_errors: tuple[float, ...]
    run_length_center: int
    maximum_represented_run_length: int
    trajectory_metrics: M4TrajectoryRecoveryMetrics
    winner_start_id: int
    multistart_result: M4MultiStartFitResult


@dataclass(frozen=True)
class M3M4ModelRecoveryResult:
    """Held-out model recovery with the nested M3 boundary represented exactly."""

    generating_model: str
    generating_bh: float
    validation_valid_trial_count: int
    validation_session_count: int
    m3_validation_elpd: float
    m4_validation_elpd: float
    delta_elpd_m4_vs_m3: float
    delta_bits_per_trial_m4_vs_m3: float
    session_level_standard_error_per_trial: float
    selected_model_by_one_standard_error_rule: str
    m3_winner_start_id: int
    m4_winner_start_id: int


def _decode_generating_parameters(constrained: torch.Tensor, registry) -> M4DecodedParameters:
    """Decode an interior vector or construct exact alpha/bh nesting cases."""

    if constrained.shape != (9,):
        raise ValueError("M4 generating parameters must have shape (9,)")
    interior = constrained.clone()
    exact_alpha_zero = float(interior[7]) == 0.0
    exact_bh_zero = float(interior[8]) == 0.0
    if exact_alpha_zero:
        interior[7] = 0.05
    if exact_bh_zero:
        interior[8] = 0.10
    decoded = decode_m4_raw_parameters(
        encode_m4_constrained_parameters(
            interior, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        ),
        lapse_max=registry.lapse_max,
        epsilon=registry.parameter_epsilon,
    )
    if exact_alpha_zero:
        decoded = decoded.with_explicit_alpha(0.0)
    if exact_bh_zero:
        decoded = decoded.with_explicit_bh(0.0)
    return decoded


def iter_registered_m4_recovery_grid(
    base_common_parameters: Iterable[float],
) -> Iterator[torch.Tensor]:
    """Yield the registered M4 dynamic recovery grid.

    ``base_common_parameters`` supplies ``(w,s0,ell_short,ell_long,kappa)``. The grid explicitly
    includes the exact M3 boundary ``bh=0``, the no-learning boundary
    ``alpha_theta=0``, and the empirical 23-trial duration reference through
    ``hazard_at_center=1/23``.
    """

    common = tuple(float(value) for value in base_common_parameters)
    if len(common) != 5:
        raise ValueError("M4 recovery grid requires w,s0,ell_short,ell_long,kappa")
    for hazard, delta0, alpha, bh in product(
        M4_RECOVERY_HAZARD_AT_CENTER_GRID,
        M4_RECOVERY_DELTA_GRID,
        M4_RECOVERY_ALPHA_GRID,
        M4_RECOVERY_BH_GRID,
    ):
        ah = float(torch.logit(torch.tensor(hazard, dtype=torch.float64)))
        yield torch.tensor((*common, ah, delta0, alpha, bh), dtype=torch.float64)


def _trajectory_metrics(sessions, generated, recovered, registry):
    def evaluations(parameters):
        return tuple(evaluate_m4_session(
            session, parameters, registry.task_moments,
            run_length_center=registry.run_length_center,
            maximum_run_length=registry.maximum_represented_run_length,
            first_long_probability=registry.first_long_probability,
            delta_min=registry.delta_min,
        ) for session in sessions)

    left, right = evaluations(generated), evaluations(recovered)

    def concatenate(items, attribute):
        return torch.cat([getattr(item.predictions, attribute).detach().reshape(-1) for item in items]).cpu().numpy()

    def rmse(attribute):
        generated_value, recovered_value = concatenate(left, attribute), concatenate(right, attribute)
        return float(np.sqrt(np.mean((recovered_value - generated_value) ** 2)))

    feedback = sum(item.feedback_trial_count for item in left)
    left_projection = sum(item.projection_activation_count for item in left)
    right_projection = sum(item.projection_activation_count for item in right)
    return M4TrajectoryRecoveryMetrics(
        trial_count=sum(item.duration_s.numel() for item in left),
        delta_rmse=rmse("delta_pre"),
        expected_run_length_rmse=rmse("expected_run_length_pre"),
        expected_hazard_rmse=rmse("expected_hazard_pre"),
        tail_probability_rmse=rmse("tail_probability_pre"),
        block_probability_rmse=rmse("pre_choice_block_probability"),
        generating_projection_fraction=left_projection / feedback if feedback else 0.0,
        recovered_projection_fraction=right_projection / feedback if feedback else 0.0,
    )


def run_m4_design_preserving_recovery(
    generating_parameters: torch.Tensor,
    training_templates: Iterable[M4SessionTrials],
    validation_templates: Iterable[M4SessionTrials],
    registry,
    *,
    random_seed: int,
    maximum_steps: int,
    output_directory: str | Path | None = None,
    parallel_workers: int = 8,
    cpu_threads_per_worker: int = 1,
) -> M4RecoveryResult:
    r"""Simulate and recover all eight pooled coordinates with eight starts.

    Purpose
    -------
    Test parameter and latent-trajectory recoverability on the observed design.
    Mathematical definition
    -----------------------
    .. math:: \hat\theta=\arg\min_\theta -\sum_t\log p(y_t\mid\theta).

    Choices follow :math:`y_t\sim Bernoulli(p_t(\theta))`; recovery minimizes
    the unchanged choice NLL over all eight raw coordinates.
    Parameter interpretation
    ------------------------
    Generating order is ``w,s0,ell_short,ell_long,kappa,ah,delta0,alpha_theta,bh``.
    Input units and shapes
    ----------------------
    Generating values have shape ``(8,)`` and templates contain duration seconds.
    Returned quantities and shapes
    ------------------------------
    Returns generating/recovered vectors, errors, trajectory metrics, and all starts.
    Constraints
    -----------
    The exact ``alpha=0`` and ``bh=0`` boundaries are allowed for generation.
    Numerical stability
    -------------------
    Fitting uses float64 full-batch L-BFGS and registered convergence gates.
    Causal restrictions
    -------------------
    Only training likelihood fits parameters; validation is held out and test stays locked.
    Important edge cases
    --------------------
    No accepted converged start raises an error rather than reporting recovery.
    Related tests
    -------------
    ``test_recovery_grid_contains_registered_boundaries`` and CPU multistart parity.
    """

    if generating_parameters.shape != (9,):
        raise ValueError("M4 recovery generating_parameters must have shape (9,)")
    if float(generating_parameters[6]) < registry.delta_min:
        raise ValueError("Generating delta0 must respect the registered delta_min")
    decoded = _decode_generating_parameters(generating_parameters, registry)
    simulation_options = dict(
        moments=registry.task_moments,
        run_length_center=registry.run_length_center,
        maximum_run_length=registry.maximum_represented_run_length,
        first_long_probability=registry.first_long_probability,
        delta_min=registry.delta_min,
    )
    training = simulate_m4_choices(
        training_templates, decoded, seed=random_seed, **simulation_options
    )
    validation = simulate_m4_choices(
        validation_templates, decoded, seed=random_seed + 1, **simulation_options
    )
    fit = fit_m4_all_initializations(
        training, validation, registry, optimizer_strategy="lbfgs",
        lbfgs_settings=LBFGSSettings(maximum_steps=maximum_steps),
        output_directory=output_directory,
        parallel_workers=parallel_workers,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )
    if fit.winner_start_id is None:
        raise RuntimeError("No accepted converged M4 recovery start")
    winner = next(item for item in fit.runs if item.start_id == fit.winner_start_id)
    recovered_decoded = decode_m4_raw_parameters(
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon,
    )
    generated = tuple(float(value) for value in generating_parameters)
    recovered = winner.best_constrained_parameters
    return M4RecoveryResult(
        generating_parameters=generated, recovered_parameters=recovered,
        parameter_errors=tuple(r-ell_long for r, ell_long in zip(recovered, generated, strict=True)),
        run_length_center=registry.run_length_center,
        maximum_represented_run_length=registry.maximum_represented_run_length,
        trajectory_metrics=_trajectory_metrics(
            (*training, *validation), decoded, recovered_decoded, registry
        ),
        winner_start_id=fit.winner_start_id, multistart_result=fit,
    )


def run_m3_m4_model_recovery(
    generating_parameters: torch.Tensor,
    training_templates: Iterable[M4SessionTrials],
    validation_templates: Iterable[M4SessionTrials],
    m4_registry,
    *,
    random_seed: int,
    maximum_steps: int,
    parallel_workers: int = 8,
    cpu_threads_per_worker: int = 1,
) -> M3M4ModelRecoveryResult:
    """Simulate M3/M4 cases, fit both eight-start pipelines, and compare validation.

    ``bh=0`` is the exact M3 generating boundary. Selection favors M3 unless
    M4's mean validation log-score gain exceeds one session-level standard
    error. Both fits use the existing full-batch L-BFGS optimization and CPU
    parallel multi-start execution; no MAP objective is introduced.
    """

    training_templates = tuple(training_templates)
    validation_templates = tuple(validation_templates)
    decoded = _decode_generating_parameters(generating_parameters, m4_registry)
    options = dict(
        moments=m4_registry.task_moments,
        run_length_center=m4_registry.run_length_center,
        maximum_run_length=m4_registry.maximum_represented_run_length,
        first_long_probability=m4_registry.first_long_probability,
        delta_min=m4_registry.delta_min,
    )
    training = simulate_m4_choices(training_templates, decoded, seed=random_seed, **options)
    validation = simulate_m4_choices(validation_templates, decoded, seed=random_seed + 1, **options)
    settings = LBFGSSettings(maximum_steps=maximum_steps)
    m3_registry = m4_registry.source_m3_registry
    m3_fit = fit_m3_all_initializations(
        training, validation, m3_registry, optimizer_strategy="lbfgs",
        lbfgs_settings=settings, parallel_workers=parallel_workers,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )
    m4_fit = fit_m4_all_initializations(
        training, validation, m4_registry, optimizer_strategy="lbfgs",
        lbfgs_settings=settings, parallel_workers=parallel_workers,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )
    if m3_fit.winner_start_id is None or m4_fit.winner_start_id is None:
        raise RuntimeError("Both M3 and M4 require an accepted model-recovery winner")
    m3_winner = next(run for run in m3_fit.runs if run.start_id == m3_fit.winner_start_id)
    m4_winner = next(run for run in m4_fit.runs if run.start_id == m4_fit.winner_start_id)
    m3_raw = torch.tensor(m3_winner.best_raw_parameters, dtype=torch.float64)
    m4_raw = torch.tensor(m4_winner.best_raw_parameters, dtype=torch.float64)
    m3_objective = evaluate_m3_mouse_objective(
        validation, m3_raw, m3_registry.task_moments,
        objective_weighting=m3_registry.objective_weighting,
        required_split_role="validation", lapse_max=m3_registry.lapse_max,
        parameter_epsilon=m3_registry.parameter_epsilon,
        delta_min=m3_registry.delta_min,
        first_long_probability=m3_registry.first_long_probability,
    )
    m4_objective = evaluate_m4_mouse_objective(
        validation, m4_raw, m4_registry.task_moments,
        objective_weighting=m4_registry.objective_weighting,
        required_split_role="validation", lapse_max=m4_registry.lapse_max,
        parameter_epsilon=m4_registry.parameter_epsilon,
        run_length_center=m4_registry.run_length_center,
        maximum_run_length=m4_registry.maximum_represented_run_length,
        delta_min=m4_registry.delta_min,
        first_long_probability=m4_registry.first_long_probability,
    )
    if m3_objective.total_valid_trial_count != m4_objective.total_valid_trial_count:
        raise AssertionError("M3/M4 recovery validation masks differ")
    session_gain = []
    for m3_session, m4_session in zip(
        m3_objective.session_evaluations, m4_objective.session_evaluations, strict=True
    ):
        if m3_session.session_id != m4_session.session_id or m3_session.valid_trial_count != m4_session.valid_trial_count:
            raise AssertionError("M3/M4 validation sessions are not aligned")
        session_gain.append(float(
            (m3_session.negative_log_likelihood - m4_session.negative_log_likelihood)
            / m4_session.valid_trial_count
        ))
    session_se = (
        float(np.std(session_gain, ddof=1) / np.sqrt(len(session_gain)))
        if len(session_gain) > 1 else float("nan")
    )
    m3_elpd = -float(m3_objective.total_negative_log_likelihood)
    m4_elpd = -float(m4_objective.total_negative_log_likelihood)
    difference = m4_elpd - m3_elpd
    count = m4_objective.total_valid_trial_count
    mean_gain = difference / count
    return M3M4ModelRecoveryResult(
        generating_model="M3_hmm_delta_v1" if float(generating_parameters[8]) == 0.0 else "M4_hsmm_delta_v1",
        generating_bh=float(generating_parameters[8]),
        validation_valid_trial_count=count,
        validation_session_count=len(session_gain),
        m3_validation_elpd=m3_elpd,
        m4_validation_elpd=m4_elpd,
        delta_elpd_m4_vs_m3=difference,
        delta_bits_per_trial_m4_vs_m3=difference / (count * np.log(2.0)),
        session_level_standard_error_per_trial=session_se,
        selected_model_by_one_standard_error_rule=(
            "M4_hsmm_delta_v1"
            if np.isfinite(session_se) and mean_gain > session_se
            else "M3_hmm_delta_v1"
        ),
        m3_winner_start_id=m3_fit.winner_start_id,
        m4_winner_start_id=m4_fit.winner_start_id,
    )
