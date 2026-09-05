"""Design-preserving all-parameter and trajectory recovery for M3."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from m0_sensory_v1.model.fit import LBFGSSettings, fit_all_initializations
from m0_sensory_v1.model.m2_session import evaluate_m2_mouse_objective

from .fit import M3MultiStartFitResult, fit_m3_all_initializations
from .parameters import decode_m3_raw_parameters, encode_m3_constrained_parameters
from .session import M3DecodedParameters, M3SessionTrials, evaluate_m3_session
from .simulation import simulate_m3_choices


M3_RECOVERY_ALPHA_GRID = (0.0, 0.02, 0.05, 0.10, 0.20)
M3_RECOVERY_DELTA_GRID = (0.20, float(np.log(4.0)), 2.20)
M3_RECOVERY_HAZARD_GRID = (0.02, 1.0 / 23.0, 0.08, 0.20)


@dataclass(frozen=True)
class M3TrajectoryRecoveryMetrics:
    """Trialwise latent-trajectory agreement on the simulated design."""

    trial_count: int
    delta_rmse: float
    delta_correlation: float
    theta_short_rmse: float
    theta_long_rmse: float
    update_signal_rmse: float
    generating_projection_count: int
    recovered_projection_count: int
    generating_projection_fraction: float
    recovered_projection_fraction: float


@dataclass(frozen=True)
class M3RecoveryResult:
    generating_parameters: tuple[float, ...]
    recovered_parameters: tuple[float, ...]
    parameter_errors: tuple[float, ...]
    generating_hazard: float
    recovered_hazard: float
    empirical_median_block_length_trials: int
    trajectory_metrics: M3TrajectoryRecoveryMetrics
    winner_start_id: int
    multistart_result: M3MultiStartFitResult


@dataclass(frozen=True)
class M2M3ModelConfusionResult:
    """Held-out evidence after fitting both models to one simulated dataset."""

    generating_model: str
    generating_alpha_theta: float
    validation_valid_trial_count: int
    m2_validation_elpd: float
    m3_validation_elpd: float
    delta_elpd_m3_vs_m2: float
    delta_bits_per_trial_m3_vs_m2: float
    selected_model_by_validation_elpd: str
    m2_winner_start_id: int
    m3_winner_start_id: int


def _decode_generating_parameters(
    constrained: torch.Tensor,
    registry,
) -> M3DecodedParameters:
    """Decode the free interior or construct the exact alpha=0 M2 boundary."""

    alpha = float(constrained[7])
    if alpha == 0.0:
        interior = constrained.clone()
        interior[7] = 0.5
        decoded = decode_m3_raw_parameters(
            encode_m3_constrained_parameters(
                interior, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            ),
            lapse_max=registry.lapse_max,
            epsilon=registry.parameter_epsilon,
        )
        return decoded.with_explicit_alpha(0.0)
    raw = encode_m3_constrained_parameters(
        constrained, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    return decode_m3_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )


def iter_registered_m3_recovery_grid(
    base_common_parameters: Iterable[float],
) -> Iterator[torch.Tensor]:
    """Yield the 60 registered dynamic recovery cases.

    ``base_common_parameters`` is `(w,s0,ell_short,ell_long,kappa)`; hazard includes `1/23`
    because the observed median block length is 23 trials.
    """

    common = tuple(float(value) for value in base_common_parameters)
    if len(common) != 5:
        raise ValueError("M3 recovery grid requires w,s0,ell_short,ell_long,kappa")
    for hazard, delta0, alpha in product(
        M3_RECOVERY_HAZARD_GRID, M3_RECOVERY_DELTA_GRID, M3_RECOVERY_ALPHA_GRID
    ):
        ah = float(torch.logit(torch.tensor(hazard, dtype=torch.float64)))
        yield torch.tensor((*common, ah, delta0, alpha), dtype=torch.float64)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _trajectory_metrics(sessions, generated, recovered, registry):
    generated_evaluations = [
        evaluate_m3_session(
            session, generated, registry.task_moments,
            first_long_probability=registry.first_long_probability,
            delta_min=registry.delta_min,
        ) for session in sessions
    ]
    recovered_evaluations = [
        evaluate_m3_session(
            session, recovered, registry.task_moments,
            first_long_probability=registry.first_long_probability,
            delta_min=registry.delta_min,
        ) for session in sessions
    ]

    def concatenate(evaluations, attribute):
        return torch.cat([
            getattr(item.predictions, attribute).detach().reshape(-1)
            for item in evaluations
        ]).cpu().numpy()

    generated_delta = concatenate(generated_evaluations, "delta_pre")
    recovered_delta = concatenate(recovered_evaluations, "delta_pre")
    generated_theta = torch.cat([
        item.predictions.subjective_prototypes_pre.detach() for item in generated_evaluations
    ]).cpu().numpy()
    recovered_theta = torch.cat([
        item.predictions.subjective_prototypes_pre.detach() for item in recovered_evaluations
    ]).cpu().numpy()
    generated_signal = concatenate(generated_evaluations, "delta_update_signal")
    recovered_signal = concatenate(recovered_evaluations, "delta_update_signal")
    finite_signal = np.isfinite(generated_signal) & np.isfinite(recovered_signal)
    feedback_count = sum(item.feedback_trial_count for item in generated_evaluations)
    generated_projection = sum(item.projection_activation_count for item in generated_evaluations)
    recovered_projection = sum(item.projection_activation_count for item in recovered_evaluations)
    return M3TrajectoryRecoveryMetrics(
        trial_count=len(generated_delta),
        delta_rmse=float(np.sqrt(np.mean((recovered_delta-generated_delta) ** 2))),
        delta_correlation=_correlation(generated_delta, recovered_delta),
        theta_short_rmse=float(np.sqrt(np.mean((recovered_theta[:, 1]-generated_theta[:, 1]) ** 2))),
        theta_long_rmse=float(np.sqrt(np.mean((recovered_theta[:, 2]-generated_theta[:, 2]) ** 2))),
        update_signal_rmse=(
            float(np.sqrt(np.mean((recovered_signal[finite_signal]-generated_signal[finite_signal]) ** 2)))
            if finite_signal.any() else float("nan")
        ),
        generating_projection_count=generated_projection,
        recovered_projection_count=recovered_projection,
        generating_projection_fraction=(generated_projection / feedback_count if feedback_count else 0.0),
        recovered_projection_fraction=(recovered_projection / feedback_count if feedback_count else 0.0),
    )


def run_m3_design_preserving_recovery(
    generating_parameters: torch.Tensor,
    training_templates: Iterable[M3SessionTrials],
    validation_templates: Iterable[M3SessionTrials],
    registry,
    *,
    random_seed: int,
    maximum_steps: int,
    output_directory: str | Path | None = None,
) -> M3RecoveryResult:
    """Simulate and recover all seven pooled coordinates on an observed design."""

    if generating_parameters.shape != (8,):
        raise ValueError("M3 recovery generating_parameters must have shape (8,)")
    if float(generating_parameters[6]) < registry.delta_min:
        raise ValueError("Generating delta0 must respect the registered delta_min")
    decoded = _decode_generating_parameters(generating_parameters, registry)
    training = simulate_m3_choices(
        training_templates, decoded, registry.task_moments, seed=random_seed,
        first_long_probability=registry.first_long_probability, delta_min=registry.delta_min,
    )
    validation = simulate_m3_choices(
        validation_templates, decoded, registry.task_moments, seed=random_seed + 1,
        first_long_probability=registry.first_long_probability, delta_min=registry.delta_min,
    )
    fit = fit_m3_all_initializations(
        training, validation, registry, optimizer_strategy="lbfgs",
        lbfgs_settings=LBFGSSettings(maximum_steps=maximum_steps),
        output_directory=output_directory,
    )
    if fit.winner_start_id is None:
        raise RuntimeError("No accepted converged M3 recovery start")
    winner = next(item for item in fit.runs if item.start_id == fit.winner_start_id)
    generated = tuple(float(value) for value in generating_parameters)
    recovered = winner.best_constrained_parameters
    recovered_decoded = decode_m3_raw_parameters(
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon,
    )
    trajectory = _trajectory_metrics(
        (*training, *validation), decoded, recovered_decoded, registry
    )
    return M3RecoveryResult(
        generating_parameters=generated,
        recovered_parameters=recovered,
        parameter_errors=tuple(r-ell_long for r, ell_long in zip(recovered, generated, strict=True)),
        generating_hazard=float(decoded.hazard),
        recovered_hazard=float(recovered_decoded.hazard),
        empirical_median_block_length_trials=registry.empirical_median_block_length_trials,
        trajectory_metrics=trajectory,
        winner_start_id=fit.winner_start_id,
        multistart_result=fit,
    )


def run_m2_m3_model_confusion(
    generating_parameters: torch.Tensor,
    training_templates: Iterable[M3SessionTrials],
    validation_templates: Iterable[M3SessionTrials],
    m3_registry,
    m2_registry,
    *,
    random_seed: int,
    maximum_steps: int,
) -> M2M3ModelConfusionResult:
    """Simulate one static/dynamic case, fit both models, and compare validation ELPD."""

    if m2_registry.task_moments != m3_registry.task_moments:
        raise ValueError("M2/M3 confusion requires identical frozen task moments")
    decoded = _decode_generating_parameters(generating_parameters, m3_registry)
    training = simulate_m3_choices(
        training_templates, decoded, m3_registry.task_moments, seed=random_seed,
        first_long_probability=m3_registry.first_long_probability,
        delta_min=m3_registry.delta_min,
    )
    validation = simulate_m3_choices(
        validation_templates, decoded, m3_registry.task_moments, seed=random_seed+1,
        first_long_probability=m3_registry.first_long_probability,
        delta_min=m3_registry.delta_min,
    )
    settings = LBFGSSettings(maximum_steps=maximum_steps)
    m2_fit = fit_all_initializations(
        training, validation, m2_registry, optimizer_strategy="lbfgs",
        lbfgs_settings=settings,
    )
    m3_fit = fit_m3_all_initializations(
        training, validation, m3_registry, optimizer_strategy="lbfgs",
        lbfgs_settings=settings,
    )
    if m2_fit.winner_start_id is None or m3_fit.winner_start_id is None:
        raise RuntimeError("Both M2 and M3 require an accepted recovery winner")
    m2_winner = next(run for run in m2_fit.runs if run.start_id == m2_fit.winner_start_id)
    m3_winner = next(run for run in m3_fit.runs if run.start_id == m3_fit.winner_start_id)
    with torch.no_grad():
        m2_objective = evaluate_m2_mouse_objective(
            validation, torch.tensor(m2_winner.best_raw_parameters, dtype=torch.float64),
            m2_registry.task_moments, objective_weighting=m2_registry.objective_weighting,
            required_split_role="validation", lapse_max=m2_registry.lapse_max,
            parameter_epsilon=m2_registry.parameter_epsilon,
            first_long_probability=m2_registry.first_long_probability,
        )
        m3_objective = evaluate_m3_mouse_objective(
            validation, torch.tensor(m3_winner.best_raw_parameters, dtype=torch.float64),
            m3_registry.task_moments, objective_weighting=m3_registry.objective_weighting,
            required_split_role="validation", lapse_max=m3_registry.lapse_max,
            parameter_epsilon=m3_registry.parameter_epsilon, delta_min=m3_registry.delta_min,
            first_long_probability=m3_registry.first_long_probability,
        )
    count = m3_objective.total_valid_trial_count
    if count != m2_objective.total_valid_trial_count:
        raise AssertionError("M2/M3 recovery validation masks differ")
    m2_elpd = -float(m2_objective.total_negative_log_likelihood)
    m3_elpd = -float(m3_objective.total_negative_log_likelihood)
    difference = m3_elpd-m2_elpd
    alpha = float(generating_parameters[7])
    return M2M3ModelConfusionResult(
        generating_model="M2_hmm_static_v1" if alpha == 0.0 else "M3_hmm_delta_v1",
        generating_alpha_theta=alpha,
        validation_valid_trial_count=count,
        m2_validation_elpd=m2_elpd,
        m3_validation_elpd=m3_elpd,
        delta_elpd_m3_vs_m2=difference,
        delta_bits_per_trial_m3_vs_m2=difference/(count*np.log(2.0)),
        selected_model_by_validation_elpd=("M3_hmm_delta_v1" if difference > 0 else "M2_hmm_static_v1"),
        m2_winner_start_id=m2_fit.winner_start_id,
        m3_winner_start_id=m3_fit.winner_start_id,
    )
