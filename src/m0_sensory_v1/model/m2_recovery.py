"""Synthetic full-parameter recovery helpers for M2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .fit import (
    FitRunResult,
    MultiStartFitResult,
    fit_all_initializations,
    fit_single_initialization,
)
from .m2_initialization import M2InitializationSpec
from .m2_simulation import simulate_m2_hmm_sessions, simulate_m2_real_design_sessions
from .m2_session import M2SessionTrials


@dataclass(frozen=True)
class M2RecoveryResult:
    """Generating and recovered values for one synthetic M2 condition."""

    generating_parameters: tuple[float, ...]
    recovered_parameters: tuple[float, ...]
    generating_hazard: float
    recovered_hazard: float
    generating_delta0: float
    recovered_delta0: float
    absolute_hazard_error: float
    absolute_delta0_error: float
    initial_training_mean_nll: float
    recovered_training_mean_nll: float
    validation_mean_nll: float
    fit_result: FitRunResult


@dataclass(frozen=True)
class M2RealDesignRecoveryResult:
    """Complete real-design recovery result for one hazard/contrast condition."""

    generating_parameters: tuple[float, ...]
    recovered_parameters: tuple[float, ...]
    parameter_errors: tuple[float, ...]
    generating_hazard: float
    recovered_hazard: float
    generating_delta0: float
    recovered_delta0: float
    multistart_result: MultiStartFitResult


def run_m2_recovery_case(
    generating_parameters: torch.Tensor,
    training_durations: Iterable[torch.Tensor],
    validation_durations: Iterable[torch.Tensor],
    registry,
    initialization: M2InitializationSpec,
    *,
    random_seed: int,
    maximum_epochs: int,
) -> M2RecoveryResult:
    r"""Generate M2 data and refit the complete seven-parameter vector.

    Purpose
    -------
    Run a bounded fully generative recovery case for all seven M2 parameters.

    Mathematical definition
    -----------------------
    .. math::

       \widehat\Theta=\arg\min_{\Theta}J_{train}(\Theta),\qquad
       \Theta=(w,s_0,\ell_short,ell_long,\kappa,a_h,\delta_0).

    Parameter interpretation
    ------------------------
    Generating and recovered vectors include sensory, lapse, hazard, and contrast.

    Input units and shapes
    ----------------------
    Durations are float64 seconds and ``generating_parameters`` has shape ``(7,)``.

    Returned quantities and shapes
    ------------------------------
    Returns complete vectors, hazard/contrast errors, likelihoods, and fit record.

    Constraints
    -----------
    Positive epoch count and a registered initialization are required.

    Numerical stability
    -------------------
    Generation and fitting reuse registered float64 transforms and objective checks.

    Causal restrictions
    -------------------
    Validation is report-only and cannot influence parameter updates.

    Important edge cases
    --------------------
    This legacy helper regenerates block schedules; real-design recovery uses the
    separate template-preserving helper.

    Related tests
    -------------
    M2 simulation, optimizer, and real-design recovery tests.
    """

    if maximum_epochs <= 0:
        raise ValueError("maximum_epochs must be positive")
    training, _ = simulate_m2_hmm_sessions(
        training_durations,
        generating_parameters,
        registry.task_moments,
        random_seed=random_seed,
        split_role="train",
        first_long_probability=registry.first_long_probability,
    )
    validation, _ = simulate_m2_hmm_sessions(
        validation_durations,
        generating_parameters,
        registry.task_moments,
        random_seed=random_seed + 1,
        split_role="validation",
        first_long_probability=registry.first_long_probability,
    )
    fit = fit_single_initialization(
        training,
        validation,
        registry,
        initialization,
        max_epochs_override=maximum_epochs,
    )
    recovered = torch.tensor(fit.best_constrained_parameters, dtype=torch.float64)
    generated_h = float(torch.sigmoid(generating_parameters[4]))
    recovered_h = float(torch.sigmoid(recovered[4]))
    return M2RecoveryResult(
        generating_parameters=tuple(float(value) for value in generating_parameters),
        recovered_parameters=tuple(float(value) for value in recovered),
        generating_hazard=generated_h,
        recovered_hazard=recovered_h,
        generating_delta0=float(generating_parameters[5]),
        recovered_delta0=float(recovered[5]),
        absolute_hazard_error=abs(recovered_h - generated_h),
        absolute_delta0_error=abs(float(recovered[5] - generating_parameters[5])),
        initial_training_mean_nll=fit.initial_training_mean_nll,
        recovered_training_mean_nll=fit.recalculated_best_training_mean_nll,
        validation_mean_nll=fit.final_validation_mean_nll,
        fit_result=fit,
    )


def run_m2_real_design_recovery_case(
    generating_parameters: torch.Tensor,
    training_templates: Iterable[M2SessionTrials],
    validation_templates: Iterable[M2SessionTrials],
    training_true_blocks: dict[str, Iterable[object]],
    validation_true_blocks: dict[str, Iterable[object]],
    registry,
    *,
    random_seed: int,
    maximum_epochs: int,
    output_directory=None,
) -> M2RealDesignRecoveryResult:
    r"""Recover all seven parameters on preserved real experimental designs.

    Purpose
    -------
    Run the primary M2 recovery using all eight starts and real design templates.

    Mathematical definition
    -----------------------
    .. math::

       \widehat\Theta=
       \arg\min_{r\in\{1,\ldots,8\}}J_{train}(\widehat\Theta_r),
       \qquad \Theta=(w,s_0,\ell_short,ell_long,\kappa,a_h,\delta_0).

    Parameter interpretation
    ------------------------
    All seven parameters are free during recovery; the input vector only generates
    synthetic categories and choices on the preserved design.

    Input units and shapes
    ----------------------
    Templates contain real session lengths, durations, omissions, and feedback;
    block maps provide one retrospective state label per task trial.

    Returned quantities and shapes
    ------------------------------
    Returns complete generating/recovered vectors, coordinate errors, and the
    full eight-start fit result.

    Constraints
    -----------
    A positive epoch budget and at least one converged registered start are required.

    Numerical stability
    -------------------
    Simulation and fitting use float64 registered transforms and fresh objectives.

    Causal restrictions
    -------------------
    True blocks are used only for offline generation. Fitting receives ordinary
    M2 sessions without true-block inputs; validation never controls optimization.

    Important edge cases
    --------------------
    If no start passes convergence, the complete start files remain saved but this
    helper raises instead of reporting an unconverged parameter estimate.

    Related tests
    -------------
    Real-design simulation preservation and existing eight-start selection tests.
    """

    if maximum_epochs <= 0:
        raise ValueError("maximum_epochs must be positive")
    training = simulate_m2_real_design_sessions(
        training_templates,
        training_true_blocks,
        generating_parameters,
        registry.task_moments,
        random_seed=random_seed,
        first_long_probability=registry.first_long_probability,
    )
    validation = simulate_m2_real_design_sessions(
        validation_templates,
        validation_true_blocks,
        generating_parameters,
        registry.task_moments,
        random_seed=random_seed + 1,
        first_long_probability=registry.first_long_probability,
    )
    fit = fit_all_initializations(
        training,
        validation,
        registry,
        output_directory=output_directory,
        max_epochs_override=maximum_epochs,
        optimizer_strategy="adam",
    )
    if fit.winner_start_id is None:
        raise RuntimeError("No converged M2 recovery start was available")
    winner = next(run for run in fit.runs if run.start_id == fit.winner_start_id)
    recovered = tuple(float(value) for value in winner.best_constrained_parameters)
    generated = tuple(float(value) for value in generating_parameters)
    generating_hazard = float(torch.sigmoid(generating_parameters[4]))
    recovered_hazard = float(
        torch.sigmoid(torch.tensor(recovered[4], dtype=torch.float64))
    )
    return M2RealDesignRecoveryResult(
        generating_parameters=generated,
        recovered_parameters=recovered,
        parameter_errors=tuple(r - ell_long for r, ell_long in zip(recovered, generated, strict=True)),
        generating_hazard=generating_hazard,
        recovered_hazard=recovered_hazard,
        generating_delta0=generated[5],
        recovered_delta0=recovered[5],
        multistart_result=fit,
    )
