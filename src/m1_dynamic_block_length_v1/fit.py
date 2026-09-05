"""Full-batch Adam, L-BFGS, and hybrid fitting for M1_dynamic_block_length_v1.

Mirrors ``m4_hsmm_delta_v1/fit.py``'s optimizer-loop shape (per-start
dataclass, multistart orchestration, ProcessPoolExecutor worker pattern,
"fresh" recomputed final loss). There is no projection machinery here (no
delta-update rule exists in this model), and unlike M4's fit.py, this
module explicitly asserts the evaluated convergence-check names exactly
equal the registry's declared ``required_checks`` before computing
``converged`` -- closing a gap M4's own fit.py does not close.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import platform
import threading
from typing import Callable, Iterable, Literal

import numpy as np
import torch

from m0_sensory_v1.model.fit import LBFGSSettings

from .initialization import M1DynamicBlockLengthInitializationSpec
from .parameters import decode_m1_dynamic_block_length_raw_parameters
from .registry import M1DynamicBlockLengthRegistry
from .session import (
    M1DynamicBlockLengthMouseObjectiveResult,
    M1DynamicBlockLengthSessionTrials,
    evaluate_m1_dynamic_block_length_mouse_objective,
)


M1DynamicBlockLengthOptimizerStrategy = Literal["adam", "lbfgs", "hybrid"]

_WORKER_CONTEXT: tuple[object, ...] | None = None
_WORKER_PROGRESS_QUEUE = None


@dataclass(frozen=True)
class M1DynamicBlockLengthFitStepRecord:
    step: int
    optimizer_phase: str
    fresh_training_mean_nll: float
    validation_mean_nll: float
    gradient_norm: float
    objective_evaluation_count: int
    closure_evaluation_count: int
    training_checkpoint_improved: bool
    learning_rate: float
    raw_parameters: tuple[float, ...]
    constrained_parameters: tuple[float, ...]
    mean_tail_probability_pre: float
    maximum_tail_probability_pre: float


@dataclass(frozen=True)
class M1DynamicBlockLengthBoundaryProximity:
    w_from_zero: float
    s0_from_zero: float
    ell_short_fraction_to_nearest_boundary: float
    ell_long_fraction_to_nearest_boundary: float
    hazard_at_center_fraction_to_nearest_boundary: float
    bh_from_zero: float


@dataclass(frozen=True)
class M1DynamicBlockLengthFitRunResult:
    result_schema_version: str
    model_id: str
    mouse_id: str
    start_id: int
    source_m1_start_id: int
    start_purpose: str
    random_seed: int
    optimizer_strategy: str
    initial_constrained_parameters: tuple[float, ...]
    initial_raw_parameters: tuple[float, ...]
    initial_training_mean_nll: float
    initial_validation_mean_nll: float
    best_raw_parameters: tuple[float, ...]
    best_constrained_parameters: tuple[float, ...]
    best_fresh_training_mean_nll: float
    recalculated_best_training_mean_nll: float
    final_raw_parameters: tuple[float, ...]
    final_constrained_parameters: tuple[float, ...]
    fresh_final_training_mean_nll: float
    final_validation_mean_nll: float
    gradient_norm_at_final: float
    final_step: int
    best_step: int
    best_optimizer_phase: str
    adam_step_count: int
    lbfgs_step_count: int
    objective_evaluation_count: int
    termination_reason: str
    converged: bool
    convergence_checks: tuple[tuple[str, bool], ...]
    tail_occupancy_acceptable: bool
    training_mean_tail_probability_pre: float
    training_maximum_tail_probability_pre: float
    validation_mean_tail_probability_pre: float
    validation_maximum_tail_probability_pre: float
    boundary_proximity: M1DynamicBlockLengthBoundaryProximity
    history: tuple[M1DynamicBlockLengthFitStepRecord, ...]
    training_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    registry_config_sha256: str
    optimizer_settings: dict[str, object]
    exception_message: str | None
    runtime: dict[str, str]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class M1DynamicBlockLengthMultiStartFitResult:
    result_schema_version: str
    model_id: str
    mouse_id: str
    runs: tuple[M1DynamicBlockLengthFitRunResult, ...]
    winner_start_id: int | None
    selection_status: str
    identifiability_ridge_detected: bool
    ridge_start_ids: tuple[int, ...]
    additional_starts_recommended: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _objective(
    sessions: tuple[M1DynamicBlockLengthSessionTrials, ...],
    raw: torch.Tensor,
    registry: M1DynamicBlockLengthRegistry,
    role: Literal["train", "validation"],
) -> M1DynamicBlockLengthMouseObjectiveResult:
    return evaluate_m1_dynamic_block_length_mouse_objective(
        sessions,
        raw,
        registry.task_moments,
        objective_weighting=registry.objective_weighting,
        required_split_role=role,
        lapse_max=registry.lapse_max,
        parameter_epsilon=registry.parameter_epsilon,
        run_length_center=registry.run_length_center,
        maximum_run_length=registry.maximum_represented_run_length,
        log_probability_epsilon=registry.log_probability_epsilon,
        prior_probability_epsilon=registry.prior_probability_epsilon,
        quadratic_tolerance=registry.quadratic_tolerance,
        discriminant_tolerance=registry.discriminant_tolerance,
        first_long_probability=registry.first_long_probability,
    )


def _floats(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu())


def _decoded(raw: torch.Tensor, registry: M1DynamicBlockLengthRegistry) -> tuple[float, ...]:
    return _floats(decode_m1_dynamic_block_length_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    ).as_tensor())


def _current_gradient_norm(raw: torch.Tensor) -> float:
    if raw.grad is None or not bool(torch.all(torch.isfinite(raw.grad))):
        raise FloatingPointError("M1_dynamic_block_length raw gradient is missing or nonfinite")
    return float(torch.linalg.vector_norm(raw.grad).detach().cpu())


def _stable(history: list[M1DynamicBlockLengthFitStepRecord], registry: M1DynamicBlockLengthRegistry) -> bool:
    window = registry.convergence.stable_training_window_epochs
    if len(history) < window:
        return False
    values = np.asarray([item.fresh_training_mean_nll for item in history[-window:]])
    return bool(
        np.all(np.isfinite(values))
        and np.ptp(values) <= registry.convergence.meaningful_improvement_tolerance
    )


def _boundary(values: tuple[float, ...], lapse_max: float) -> M1DynamicBlockLengthBoundaryProximity:
    w, s0, ell_short, ell_long, _kappa, ah, bh = values
    hazard = 1.0 / (1.0 + np.exp(-ah))
    return M1DynamicBlockLengthBoundaryProximity(
        w_from_zero=w,
        s0_from_zero=s0,
        ell_short_fraction_to_nearest_boundary=min(ell_short, lapse_max - ell_short) / lapse_max,
        ell_long_fraction_to_nearest_boundary=min(ell_long, lapse_max - ell_long) / lapse_max,
        hazard_at_center_fraction_to_nearest_boundary=min(hazard, 1.0 - hazard),
        bh_from_zero=bh,
    )


def _registered_lbfgs_settings(registry: M1DynamicBlockLengthRegistry) -> LBFGSSettings:
    return LBFGSSettings(
        maximum_steps=registry.fitting.lbfgs_maximum_steps,
        learning_rate=registry.fitting.lbfgs_learning_rate,
        history_size=registry.fitting.lbfgs_history_size,
        maximum_evaluations_per_step=registry.fitting.lbfgs_maximum_evaluations_per_step,
        tolerance_grad=registry.fitting.lbfgs_tolerance_grad,
        tolerance_change=registry.fitting.lbfgs_tolerance_change,
        line_search_fn=registry.fitting.lbfgs_line_search_fn,
    )


def _classify_fit_error(error: BaseException) -> tuple[bool, bool]:
    message = str(error).lower()
    transform = isinstance(error, ValueError) and any(
        token in message for token in ("parameter", "transform", "raw", "lapse_max")
    )
    normalization = any(
        token in message
        for token in ("normalizer", "normalized", "normalization", "belief", "probability")
    )
    return transform, normalization


def _initialize_worker(
    training: tuple[M1DynamicBlockLengthSessionTrials, ...],
    validation: tuple[M1DynamicBlockLengthSessionTrials, ...],
    registry: M1DynamicBlockLengthRegistry,
    optimizer_strategy: M1DynamicBlockLengthOptimizerStrategy,
    max_epochs_override: int | None,
    initial_learning_rate_override: float | None,
    adam_warmup_epochs: int,
    lbfgs_settings: LBFGSSettings | None,
    allow_short_convergence_budget: bool,
    cpu_threads_per_worker: int,
    progress_queue,
) -> None:
    global _WORKER_CONTEXT, _WORKER_PROGRESS_QUEUE
    torch.set_num_threads(cpu_threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_CONTEXT = (
        training, validation, registry, optimizer_strategy, max_epochs_override,
        initial_learning_rate_override, adam_warmup_epochs, lbfgs_settings,
        allow_short_convergence_budget,
    )
    _WORKER_PROGRESS_QUEUE = progress_queue


def _fit_start_in_worker(
    initialization: M1DynamicBlockLengthInitializationSpec,
) -> M1DynamicBlockLengthFitRunResult:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("M1_dynamic_block_length parallel worker was not initialized")
    (
        training, validation, registry, optimizer_strategy, max_epochs_override,
        initial_learning_rate_override, adam_warmup_epochs, lbfgs_settings,
        allow_short_convergence_budget,
    ) = _WORKER_CONTEXT
    initial_callback = None
    epoch_callback = None
    if _WORKER_PROGRESS_QUEUE is not None:
        initial_callback = lambda train, validation: _WORKER_PROGRESS_QUEUE.put(
            ("initial", initialization.start_id, train, validation)
        )
        epoch_callback = lambda record: _WORKER_PROGRESS_QUEUE.put(
            ("step", initialization.start_id, record)
        )
    return fit_m1_dynamic_block_length_single_initialization(
        training, validation, registry, initialization,
        optimizer_strategy=optimizer_strategy, max_epochs_override=max_epochs_override,
        initial_learning_rate_override=initial_learning_rate_override,
        adam_warmup_epochs=adam_warmup_epochs, lbfgs_settings=lbfgs_settings,
        allow_short_convergence_budget=allow_short_convergence_budget,
        initial_callback=initial_callback, epoch_callback=epoch_callback,
    )


def fit_m1_dynamic_block_length_single_initialization(
    training_sessions: Iterable[M1DynamicBlockLengthSessionTrials],
    validation_sessions: Iterable[M1DynamicBlockLengthSessionTrials],
    registry: M1DynamicBlockLengthRegistry,
    initialization: M1DynamicBlockLengthInitializationSpec,
    *,
    optimizer_strategy: M1DynamicBlockLengthOptimizerStrategy = "adam",
    max_epochs_override: int | None = None,
    initial_learning_rate_override: float | None = None,
    adam_warmup_epochs: int = 1000,
    lbfgs_settings: LBFGSSettings | None = None,
    allow_short_convergence_budget: bool = False,
    initial_callback: Callable[[float, float], None] | None = None,
    epoch_callback: Callable[[M1DynamicBlockLengthFitStepRecord], None] | None = None,
) -> M1DynamicBlockLengthFitRunResult:
    """Fit one start with three full-batch optimizer strategies."""

    training = tuple(training_sessions)
    validation = tuple(validation_sessions)
    if not training or not validation:
        raise ValueError("M1_dynamic_block_length fitting requires nonempty train and validation sets")
    if initialization not in registry.initializations:
        raise ValueError("M1_dynamic_block_length initialization is not registered")
    if optimizer_strategy not in {"adam", "lbfgs", "hybrid"}:
        raise ValueError(f"Unknown M1_dynamic_block_length optimizer strategy: {optimizer_strategy}")
    if {item.mouse_id for item in (*training, *validation)} != {training[0].mouse_id}:
        raise ValueError("All M1_dynamic_block_length fit sessions must belong to one mouse")
    if any(item.split_role != "train" for item in training):
        raise ValueError("M1_dynamic_block_length training input contains a non-training session")
    if any(item.split_role != "validation" for item in validation):
        raise ValueError("M1_dynamic_block_length validation input contains a non-validation session")
    if optimizer_strategy != "adam" and max_epochs_override is not None:
        raise ValueError("max_epochs_override applies only to Adam")
    if optimizer_strategy == "lbfgs" and initial_learning_rate_override is not None:
        raise ValueError("Adam learning-rate override cannot configure pure L-BFGS")
    if adam_warmup_epochs <= 0:
        raise ValueError("adam_warmup_epochs must be positive")

    settings = lbfgs_settings or _registered_lbfgs_settings(registry)
    adam_epochs = (
        registry.optimizer.maximum_epochs if max_epochs_override is None else max_epochs_override
    ) if optimizer_strategy == "adam" else (
        adam_warmup_epochs if optimizer_strategy == "hybrid" else 0
    )
    planned_steps = adam_epochs + (
        settings.maximum_steps if optimizer_strategy in {"lbfgs", "hybrid"} else 0
    )
    if planned_steps <= 0:
        raise ValueError("Optimizer step budget must be positive")
    if (
        optimizer_strategy != "adam"
        and planned_steps < registry.convergence.stable_training_window_epochs
        and not allow_short_convergence_budget
    ):
        raise ValueError("Optimizer step budget cannot fill the registered stability window")

    random_seed = registry.seeds.global_seed + initialization.start_id
    torch.manual_seed(random_seed)
    initial_raw = initialization.raw_tensor(lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon)
    raw = torch.nn.Parameter(initial_raw.clone())
    with torch.no_grad():
        initial_train = _objective(training, raw, registry, "train")
        initial_validation = _objective(validation, raw, registry, "validation")
    if initial_callback is not None:
        initial_callback(float(initial_train.loss), float(initial_validation.loss))

    history: list[M1DynamicBlockLengthFitStepRecord] = []
    lbfgs_history: list[M1DynamicBlockLengthFitStepRecord] = []
    best_raw = raw.detach().clone()
    best_train = float(initial_train.loss)
    best_step = 0
    best_phase = "initial"
    global_step = adam_steps = lbfgs_steps = 0
    training_evaluations = 1
    epochs_without_improvement = 0
    exception_message: str | None = None
    parameter_transform_error = False
    probability_normalization_error = False
    termination = "maximum_epochs_reached" if optimizer_strategy == "adam" else "maximum_lbfgs_steps_reached"

    if adam_epochs:
        adam_lr = registry.optimizer.initial_learning_rate if initial_learning_rate_override is None else initial_learning_rate_override
        if not np.isfinite(adam_lr) or adam_lr <= 0.0:
            raise ValueError("Adam learning rate must be finite and positive")
        adam = torch.optim.Adam([raw], lr=adam_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            adam, mode="min", factor=registry.scheduler.factor,
            patience=registry.scheduler.patience_epochs,
            threshold=registry.convergence.meaningful_improvement_tolerance,
            threshold_mode="abs", min_lr=registry.scheduler.minimum_learning_rate,
        )
        for _ in range(adam_epochs):
            try:
                adam.zero_grad(set_to_none=True)
                before = _objective(training, raw, registry, "train")
                training_evaluations += 1
                if not bool(torch.isfinite(before.loss)):
                    raise FloatingPointError("Nonfinite M1_dynamic_block_length Adam objective")
                before.loss.backward()
                if raw.grad is None or not bool(torch.all(torch.isfinite(raw.grad))):
                    raise FloatingPointError("Nonfinite M1_dynamic_block_length Adam gradient")
                gradient_norm = float(torch.linalg.vector_norm(raw.grad).detach().cpu())
                adam.step()
                adam_steps += 1
                global_step += 1
                if not bool(torch.all(torch.isfinite(raw))):
                    raise FloatingPointError("Adam produced nonfinite M1_dynamic_block_length parameters")
                with torch.no_grad():
                    fresh = _objective(training, raw, registry, "train")
                    fresh_validation = _objective(validation, raw, registry, "validation")
                training_evaluations += 1
                fresh_value = float(fresh.loss)
                improved = fresh_value < best_train - registry.convergence.meaningful_improvement_tolerance
                # Checkpoint on any strict improvement; the tolerance-gated
                # `improved` flag is reserved for the patience counter only
                # (see the L-BFGS phase for why conflating the two can turn
                # a converged run into a reported non-convergence).
                if fresh_value < best_train:
                    best_raw = raw.detach().clone()
                    best_train = fresh_value
                    best_step = global_step
                    best_phase = "adam"
                if improved:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                scheduler.step(fresh_value)
                record = M1DynamicBlockLengthFitStepRecord(
                    step=global_step, optimizer_phase="adam", fresh_training_mean_nll=fresh_value,
                    validation_mean_nll=float(fresh_validation.loss), gradient_norm=gradient_norm,
                    objective_evaluation_count=training_evaluations, closure_evaluation_count=0,
                    training_checkpoint_improved=improved, learning_rate=float(adam.param_groups[0]["lr"]),
                    raw_parameters=_floats(raw), constrained_parameters=_decoded(raw, registry),
                    mean_tail_probability_pre=fresh.mean_tail_probability_pre,
                    maximum_tail_probability_pre=fresh.maximum_tail_probability_pre,
                )
                history.append(record)
                if epoch_callback is not None:
                    epoch_callback(record)
                if (
                    optimizer_strategy == "adam"
                    and float(adam.param_groups[0]["lr"]) <= registry.scheduler.minimum_learning_rate
                    and epochs_without_improvement >= registry.convergence.no_improvement_patience_epochs
                ):
                    termination = "minimum_lr_plateau_reached"
                    break
            except (FloatingPointError, RuntimeError, ValueError) as error:
                transform_error, normalization_error = _classify_fit_error(error)
                parameter_transform_error |= transform_error
                probability_normalization_error |= normalization_error
                exception_message = f"{type(error).__name__}: {error}"
                termination = "numerical_or_transform_failure"
                break
        if optimizer_strategy == "hybrid":
            with torch.no_grad():
                raw.copy_(best_raw)

    if optimizer_strategy in {"lbfgs", "hybrid"} and exception_message is None:
        optimizer = torch.optim.LBFGS(
            [raw], lr=settings.learning_rate, max_iter=1, max_eval=settings.maximum_evaluations_per_step,
            history_size=settings.history_size, tolerance_grad=settings.tolerance_grad,
            tolerance_change=settings.tolerance_change, line_search_fn=settings.line_search_fn,
        )
        epochs_without_improvement = 0
        for _ in range(settings.maximum_steps):
            previous_raw = raw.detach().clone()
            closure_calls = 0
            try:
                def closure() -> torch.Tensor:
                    nonlocal closure_calls, training_evaluations
                    optimizer.zero_grad(set_to_none=True)
                    value = _objective(training, raw, registry, "train").loss
                    closure_calls += 1
                    training_evaluations += 1
                    if not bool(torch.isfinite(value)):
                        raise FloatingPointError("Nonfinite M1_dynamic_block_length L-BFGS objective")
                    value.backward()
                    if raw.grad is None or not bool(torch.all(torch.isfinite(raw.grad))):
                        raise FloatingPointError("Nonfinite M1_dynamic_block_length L-BFGS gradient")
                    return value

                optimizer.step(closure)
                lbfgs_steps += 1
                global_step += 1
                if not bool(torch.all(torch.isfinite(raw))):
                    raise FloatingPointError("L-BFGS produced nonfinite M1_dynamic_block_length parameters")
                # Recompute the gradient fresh at the parameters L-BFGS
                # actually accepted -- raw.grad right after .step() can
                # still belong to an earlier line-search trial point rather
                # than the point actually settled on (PyTorch's LBFGS only
                # restores parameter data between trials, not .grad), which
                # would otherwise corrupt both the logged gradient norm and
                # the convergence-check decision below.
                optimizer.zero_grad(set_to_none=True)
                fresh = _objective(training, raw, registry, "train")
                fresh.loss.backward()
                if raw.grad is None or not bool(torch.all(torch.isfinite(raw.grad))):
                    raise FloatingPointError(
                        "Nonfinite M1_dynamic_block_length L-BFGS post-step gradient"
                    )
                gradient_norm = _current_gradient_norm(raw)
                with torch.no_grad():
                    fresh_validation = _objective(validation, raw, registry, "validation")
                training_evaluations += 1
                fresh_value = float(fresh.loss)
                improved = fresh_value < best_train - registry.convergence.meaningful_improvement_tolerance
                if fresh_value < best_train:
                    best_raw = raw.detach().clone()
                    best_train = fresh_value
                    best_step = global_step
                    best_phase = "lbfgs"
                if improved:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                record = M1DynamicBlockLengthFitStepRecord(
                    step=global_step, optimizer_phase="lbfgs", fresh_training_mean_nll=fresh_value,
                    validation_mean_nll=float(fresh_validation.loss), gradient_norm=gradient_norm,
                    objective_evaluation_count=training_evaluations, closure_evaluation_count=closure_calls,
                    training_checkpoint_improved=improved, learning_rate=float(optimizer.param_groups[0]["lr"]),
                    raw_parameters=_floats(raw), constrained_parameters=_decoded(raw, registry),
                    mean_tail_probability_pre=fresh.mean_tail_probability_pre,
                    maximum_tail_probability_pre=fresh.maximum_tail_probability_pre,
                )
                history.append(record)
                lbfgs_history.append(record)
                if epoch_callback is not None:
                    epoch_callback(record)
                stability_records = history if optimizer_strategy == "hybrid" else lbfgs_history
                if _stable(stability_records, registry) and gradient_norm <= registry.convergence.gradient_norm_tolerance:
                    termination = "lbfgs_convergence_checks_passed"
                    break
                if _stable(stability_records, registry) and epochs_without_improvement >= registry.convergence.no_improvement_patience_epochs:
                    termination = "lbfgs_stable_plateau_reached"
                    break
            except (FloatingPointError, RuntimeError, ValueError) as error:
                with torch.no_grad():
                    raw.copy_(previous_raw)
                transform_error, normalization_error = _classify_fit_error(error)
                parameter_transform_error |= transform_error
                probability_normalization_error |= normalization_error
                exception_message = f"{type(error).__name__}: {error}"
                termination = "lbfgs_line_search_or_transform_failure"
                break

    with torch.no_grad():
        raw.copy_(best_raw)
    raw.grad = None
    final_training = _objective(training, raw, registry, "train")
    training_evaluations += 1
    final_training.loss.backward()
    final_gradient_norm = _current_gradient_norm(raw)
    with torch.no_grad():
        final_validation = _objective(validation, raw, registry, "validation")
    final_values = _decoded(raw, registry)
    finite = bool(
        np.isfinite(float(final_training.loss.detach().cpu()))
        and np.isfinite(float(final_validation.loss))
        and np.isfinite(final_gradient_norm)
        and all(np.isfinite(final_values))
    )
    tail_occupancy_acceptable = (
        final_training.mean_tail_probability_pre <= registry.maximum_accepted_mean_tail_probability
    )
    stability_records = history if optimizer_strategy != "lbfgs" else lbfgs_history
    checks = tuple((name, bool(value)) for name, value in (
        ("finite_objective", finite),
        ("finite_parameters", all(np.isfinite(final_values))),
        ("finite_gradients", np.isfinite(final_gradient_norm)),
        ("small_gradient_norm", final_gradient_norm <= registry.convergence.gradient_norm_tolerance),
        ("stable_fresh_training_nll", _stable(stability_records, registry)),
        ("no_parameter_transform_errors", not parameter_transform_error),
        ("no_probability_normalization_errors", not probability_normalization_error),
        ("no_optimizer_runtime_errors", exception_message is None),
        ("tail_occupancy_acceptable", tail_occupancy_acceptable),
    ))
    evaluated_names = {name for name, _ in checks}
    declared_names = set(registry.convergence.required_checks)
    if evaluated_names != declared_names:
        raise ValueError(
            "M1_dynamic_block_length evaluated convergence checks "
            f"{sorted(evaluated_names)} do not exactly match the registry's "
            f"declared required_checks {sorted(declared_names)}"
        )
    converged = all(value for _, value in checks)
    if termination == "maximum_epochs_reached":
        termination = "maximum_epochs_convergence_checks_passed" if converged else "maximum_epochs_without_convergence"
    if termination == "maximum_lbfgs_steps_reached":
        termination = "maximum_lbfgs_steps_convergence_checks_passed" if converged else "maximum_lbfgs_steps_without_convergence"

    optimizer_settings = {
        "strategy": optimizer_strategy,
        "adam": {
            "maximum_epochs": adam_epochs,
            "initial_learning_rate": registry.optimizer.initial_learning_rate if initial_learning_rate_override is None else initial_learning_rate_override,
            "scheduler": "registered_reduce_on_plateau",
        },
        "lbfgs": settings.as_dict(),
        "full_batch": True,
        "backend": "torch",
    }
    return M1DynamicBlockLengthFitRunResult(
        result_schema_version=registry.multi_start_selection.result_schema_version,
        model_id=registry.model_id, mouse_id=training[0].mouse_id, start_id=initialization.start_id,
        source_m1_start_id=initialization.source_m1_start_id, start_purpose=initialization.purpose,
        random_seed=random_seed, optimizer_strategy=optimizer_strategy,
        initial_constrained_parameters=_floats(initialization.constrained_tensor()),
        initial_raw_parameters=_floats(initial_raw),
        initial_training_mean_nll=float(initial_train.loss), initial_validation_mean_nll=float(initial_validation.loss),
        best_raw_parameters=_floats(best_raw), best_constrained_parameters=final_values,
        best_fresh_training_mean_nll=best_train,
        recalculated_best_training_mean_nll=float(final_training.loss.detach().cpu()),
        final_raw_parameters=_floats(raw), final_constrained_parameters=final_values,
        fresh_final_training_mean_nll=float(final_training.loss.detach().cpu()),
        final_validation_mean_nll=float(final_validation.loss), gradient_norm_at_final=final_gradient_norm,
        final_step=global_step, best_step=best_step, best_optimizer_phase=best_phase,
        adam_step_count=adam_steps, lbfgs_step_count=lbfgs_steps,
        objective_evaluation_count=training_evaluations, termination_reason=termination,
        converged=converged, convergence_checks=checks, tail_occupancy_acceptable=tail_occupancy_acceptable,
        training_mean_tail_probability_pre=final_training.mean_tail_probability_pre,
        training_maximum_tail_probability_pre=final_training.maximum_tail_probability_pre,
        validation_mean_tail_probability_pre=final_validation.mean_tail_probability_pre,
        validation_maximum_tail_probability_pre=final_validation.maximum_tail_probability_pre,
        boundary_proximity=_boundary(final_values, registry.lapse_max), history=tuple(history),
        training_session_ids=tuple(item.session_id for item in training),
        validation_session_ids=tuple(item.session_id for item in validation),
        registry_config_sha256=sha256(registry.config_path.read_bytes()).hexdigest(),
        optimizer_settings=optimizer_settings, exception_message=exception_message,
        runtime={
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(raw.device), "dtype": str(raw.dtype),
        },
    )


def fit_m1_dynamic_block_length_all_initializations(
    training_sessions: Iterable[M1DynamicBlockLengthSessionTrials],
    validation_sessions: Iterable[M1DynamicBlockLengthSessionTrials],
    registry: M1DynamicBlockLengthRegistry,
    *,
    optimizer_strategy: M1DynamicBlockLengthOptimizerStrategy = "adam",
    max_epochs_override: int | None = None,
    initial_learning_rate_override: float | None = None,
    adam_warmup_epochs: int = 1000,
    lbfgs_settings: LBFGSSettings | None = None,
    allow_short_convergence_budget: bool = False,
    parallel_workers: int = 8,
    cpu_threads_per_worker: int = 1,
    initial_callback: Callable[[int, float, float], None] | None = None,
    epoch_callback: Callable[[int, M1DynamicBlockLengthFitStepRecord], None] | None = None,
    completion_callback: Callable[[M1DynamicBlockLengthFitRunResult], None] | None = None,
    output_directory: str | Path | None = None,
) -> M1DynamicBlockLengthMultiStartFitResult:
    """Fit the eight frozen starts concurrently and select by training NLL."""

    training = tuple(training_sessions)
    validation = tuple(validation_sessions)
    if not training or not validation:
        raise ValueError("M1_dynamic_block_length multistart fitting requires train and validation sessions")
    if not isinstance(parallel_workers, int) or parallel_workers <= 0:
        raise ValueError("parallel_workers must be a positive integer")
    if not isinstance(cpu_threads_per_worker, int) or cpu_threads_per_worker <= 0:
        raise ValueError("cpu_threads_per_worker must be a positive integer")
    initializations = tuple(registry.initializations)
    worker_count = min(parallel_workers, len(initializations), multiprocessing.cpu_count())
    runs: list[M1DynamicBlockLengthFitRunResult] = []
    output = None if output_directory is None else Path(output_directory).resolve()
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    def save_completed_run(run: M1DynamicBlockLengthFitRunResult) -> None:
        if output is not None:
            (output / f"start_{run.start_id:02d}.json").write_text(
                json.dumps(run.as_dict(), indent=2) + "\n", encoding="utf-8"
            )
        if completion_callback is not None:
            completion_callback(run)

    if worker_count == 1:
        for initialization in initializations:
            run = fit_m1_dynamic_block_length_single_initialization(
                training, validation, registry, initialization,
                optimizer_strategy=optimizer_strategy, max_epochs_override=max_epochs_override,
                initial_learning_rate_override=initial_learning_rate_override,
                adam_warmup_epochs=adam_warmup_epochs, lbfgs_settings=lbfgs_settings,
                allow_short_convergence_budget=allow_short_convergence_budget,
                initial_callback=(None if initial_callback is None else
                    lambda train, validation, start_id=initialization.start_id: initial_callback(start_id, train, validation)),
                epoch_callback=(None if epoch_callback is None else
                    lambda record, start_id=initialization.start_id: epoch_callback(start_id, record)),
            )
            runs.append(run)
            save_completed_run(run)
    else:
        spawn_context = multiprocessing.get_context("spawn")
        progress_queue = spawn_context.Queue() if initial_callback is not None or epoch_callback is not None else None
        callback_errors: list[BaseException] = []
        progress_thread = None
        if progress_queue is not None:
            def consume_progress() -> None:
                while True:
                    event = progress_queue.get()
                    if event[0] == "stop":
                        return
                    try:
                        if event[0] == "initial" and initial_callback is not None:
                            initial_callback(event[1], event[2], event[3])
                        elif event[0] == "step" and epoch_callback is not None:
                            epoch_callback(event[1], event[2])
                    except BaseException as error:
                        callback_errors.append(error)

            progress_thread = threading.Thread(target=consume_progress, name="m1d-progress-consumer", daemon=True)
            progress_thread.start()
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=spawn_context, initializer=_initialize_worker,
                initargs=(
                    training, validation, registry, optimizer_strategy, max_epochs_override,
                    initial_learning_rate_override, adam_warmup_epochs, lbfgs_settings,
                    allow_short_convergence_budget, cpu_threads_per_worker, progress_queue,
                ),
            ) as executor:
                futures = {
                    executor.submit(_fit_start_in_worker, initialization): initialization.start_id
                    for initialization in initializations
                }
                for future in as_completed(futures):
                    run = future.result()
                    runs.append(run)
                    save_completed_run(run)
        finally:
            if progress_queue is not None:
                progress_queue.put(("stop",))
                assert progress_thread is not None
                progress_thread.join()
                progress_queue.close()
                progress_queue.join_thread()
        if callback_errors:
            raise RuntimeError("M1_dynamic_block_length progress callback failed") from callback_errors[0]
    runs.sort(key=lambda item: item.start_id)
    eligible = [item for item in runs if item.converged and item.tail_occupancy_acceptable]
    winner = min(eligible, key=lambda item: item.recalculated_best_training_mean_nll) if eligible else None
    ridge_ids: tuple[int, ...] = ()
    ridge = False
    if winner is not None:
        tolerance = registry.multi_start_selection.near_equal_training_nll_tolerance
        scale = np.asarray(registry.multi_start_selection.parameter_distance_scale)
        distant: list[int] = []
        winner_values = np.asarray(winner.best_constrained_parameters)
        for candidate in eligible:
            if candidate.start_id == winner.start_id:
                continue
            if abs(candidate.recalculated_best_training_mean_nll - winner.recalculated_best_training_mean_nll) > tolerance:
                continue
            distance = float(np.linalg.vector_norm((np.asarray(candidate.best_constrained_parameters) - winner_values) / scale))
            if distance >= registry.multi_start_selection.identifiability_parameter_distance_threshold:
                distant.append(candidate.start_id)
        ridge_ids = tuple(sorted({winner.start_id, *distant})) if distant else ()
        ridge = bool(distant)
    result = M1DynamicBlockLengthMultiStartFitResult(
        result_schema_version=registry.multi_start_selection.result_schema_version,
        model_id=registry.model_id, mouse_id=training[0].mouse_id if training else "",
        runs=tuple(runs), winner_start_id=None if winner is None else winner.start_id,
        selection_status="winner_selected" if winner is not None else "no_accepted_converged_run",
        identifiability_ridge_detected=ridge, ridge_start_ids=ridge_ids,
        additional_starts_recommended=winner is None or ridge,
    )
    if output is not None:
        (output / "multistart_summary.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return result
