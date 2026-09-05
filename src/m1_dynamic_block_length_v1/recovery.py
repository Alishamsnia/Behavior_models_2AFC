"""Design-preserving parameter and latent-trajectory recovery for
M1_dynamic_block_length_v1."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from m0_sensory_v1.model.fit import LBFGSSettings

from .fit import M1DynamicBlockLengthMultiStartFitResult, fit_m1_dynamic_block_length_all_initializations
from .parameters import (
    M1DynamicBlockLengthParameters,
    decode_m1_dynamic_block_length_raw_parameters,
    encode_m1_dynamic_block_length_constrained_parameters,
)
from .session import M1DynamicBlockLengthSessionTrials, evaluate_m1_dynamic_block_length_session
from .simulation import simulate_m1_dynamic_block_length_choices


M1_DYNAMIC_BLOCK_LENGTH_RECOVERY_HAZARD_AT_CENTER_GRID = (0.02, 1.0 / 23.0, 0.08)
M1_DYNAMIC_BLOCK_LENGTH_RECOVERY_BH_GRID = (0.0, 0.05, 0.10, 0.20, 0.40)


@dataclass(frozen=True)
class M1DynamicBlockLengthTrajectoryRecoveryMetrics:
    trial_count: int
    expected_run_length_rmse: float
    expected_hazard_rmse: float
    tail_probability_rmse: float
    block_probability_rmse: float


@dataclass(frozen=True)
class M1DynamicBlockLengthRecoveryResult:
    generating_parameters: tuple[float, ...]
    recovered_parameters: tuple[float, ...]
    parameter_errors: tuple[float, ...]
    run_length_center: int
    maximum_represented_run_length: int
    trajectory_metrics: M1DynamicBlockLengthTrajectoryRecoveryMetrics
    winner_start_id: int
    multistart_result: M1DynamicBlockLengthMultiStartFitResult


def _decode_generating_parameters(constrained: torch.Tensor, registry) -> M1DynamicBlockLengthParameters:
    """Decode an interior vector or construct the exact ``bh=0`` nesting case."""

    if constrained.shape != (7,):
        raise ValueError("M1_dynamic_block_length generating parameters must have shape (7,)")
    interior = constrained.clone()
    exact_bh_zero = float(interior[6]) == 0.0
    if exact_bh_zero:
        interior[6] = 0.10
    decoded = decode_m1_dynamic_block_length_raw_parameters(
        encode_m1_dynamic_block_length_constrained_parameters(
            interior, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        ),
        lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon,
    )
    if exact_bh_zero:
        decoded = decoded.with_explicit_bh(0.0)
    return decoded


def iter_registered_m1_dynamic_block_length_recovery_grid(
    base_common_parameters: Iterable[float],
) -> Iterator[torch.Tensor]:
    """Yield the registered dynamic recovery grid: hazard-at-center x bh.

    ``base_common_parameters`` supplies ``(w,s0,ell_short,ell_long,kappa)``. The
    grid includes the exact ``bh=0`` boundary (nests real M1 exactly) and the
    empirical 23-trial duration reference ``hazard_at_center=1/23``.
    """

    common = tuple(float(value) for value in base_common_parameters)
    if len(common) != 5:
        raise ValueError("M1_dynamic_block_length recovery grid requires w,s0,ell_short,ell_long,kappa")
    for hazard, bh in product(
        M1_DYNAMIC_BLOCK_LENGTH_RECOVERY_HAZARD_AT_CENTER_GRID,
        M1_DYNAMIC_BLOCK_LENGTH_RECOVERY_BH_GRID,
    ):
        ah = float(torch.logit(torch.tensor(hazard, dtype=torch.float64)))
        yield torch.tensor((*common, ah, bh), dtype=torch.float64)


def _trajectory_metrics(sessions, generated, recovered, registry):
    def evaluations(parameters):
        return tuple(evaluate_m1_dynamic_block_length_session(
            session, parameters, registry.task_moments,
            run_length_center=registry.run_length_center,
            maximum_run_length=registry.maximum_represented_run_length,
            first_long_probability=registry.first_long_probability,
        ) for session in sessions)

    left, right = evaluations(generated), evaluations(recovered)

    def concatenate(items, attribute):
        return torch.cat([getattr(item.predictions, attribute).detach().reshape(-1) for item in items]).cpu().numpy()

    def rmse(attribute):
        generated_value, recovered_value = concatenate(left, attribute), concatenate(right, attribute)
        return float(np.sqrt(np.mean((recovered_value - generated_value) ** 2)))

    return M1DynamicBlockLengthTrajectoryRecoveryMetrics(
        trial_count=sum(item.duration_s.numel() for item in left),
        expected_run_length_rmse=rmse("expected_run_length_pre"),
        expected_hazard_rmse=rmse("expected_hazard_pre"),
        tail_probability_rmse=rmse("tail_probability_pre"),
        block_probability_rmse=rmse("pre_choice_block_probability"),
    )


def run_m1_dynamic_block_length_design_preserving_recovery(
    generating_parameters: torch.Tensor,
    training_templates: Iterable[M1DynamicBlockLengthSessionTrials],
    validation_templates: Iterable[M1DynamicBlockLengthSessionTrials],
    registry,
    *,
    random_seed: int,
    maximum_steps: int,
    output_directory: str | Path | None = None,
    parallel_workers: int = 8,
    cpu_threads_per_worker: int = 1,
) -> M1DynamicBlockLengthRecoveryResult:
    """Simulate and recover all seven pooled coordinates with eight starts."""

    if generating_parameters.shape != (7,):
        raise ValueError("M1_dynamic_block_length recovery generating_parameters must have shape (7,)")
    decoded = _decode_generating_parameters(generating_parameters, registry)
    simulation_options = dict(
        moments=registry.task_moments, run_length_center=registry.run_length_center,
        maximum_run_length=registry.maximum_represented_run_length,
        first_long_probability=registry.first_long_probability,
    )
    training = simulate_m1_dynamic_block_length_choices(training_templates, decoded, seed=random_seed, **simulation_options)
    validation = simulate_m1_dynamic_block_length_choices(validation_templates, decoded, seed=random_seed + 1, **simulation_options)
    fit = fit_m1_dynamic_block_length_all_initializations(
        training, validation, registry, optimizer_strategy="lbfgs",
        lbfgs_settings=LBFGSSettings(maximum_steps=maximum_steps),
        output_directory=output_directory, parallel_workers=parallel_workers,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )
    if fit.winner_start_id is None:
        raise RuntimeError("No accepted converged M1_dynamic_block_length recovery start")
    winner = next(item for item in fit.runs if item.start_id == fit.winner_start_id)
    recovered_decoded = decode_m1_dynamic_block_length_raw_parameters(
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon,
    )
    generated = tuple(float(value) for value in generating_parameters)
    recovered = winner.best_constrained_parameters
    return M1DynamicBlockLengthRecoveryResult(
        generating_parameters=generated, recovered_parameters=recovered,
        parameter_errors=tuple(r - g for r, g in zip(recovered, generated, strict=True)),
        run_length_center=registry.run_length_center,
        maximum_represented_run_length=registry.maximum_represented_run_length,
        trajectory_metrics=_trajectory_metrics((*training, *validation), decoded, recovered_decoded, registry),
        winner_start_id=fit.winner_start_id, multistart_result=fit,
    )
