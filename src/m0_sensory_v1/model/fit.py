"""Registered Adam plus opt-in L-BFGS fitting and multistart orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import threading
from typing import Callable, Literal, TYPE_CHECKING, Iterable

import numpy as np
import torch

from .initialization import InitializationSpec
from .parameters import decode_raw_parameters
from .session import SessionTrials, evaluate_mouse_objective

if TYPE_CHECKING:
    from ..registry import ModelRegistry


OptimizerStrategy = Literal["adam", "lbfgs", "hybrid"]


_MULTISTART_WORKER_CONTEXT: tuple[object, ...] | None = None
_MULTISTART_WORKER_PROGRESS_QUEUE = None


@dataclass(frozen=True)
class EpochRecord:
    """Diagnostics from one complete train-step/fresh-evaluation epoch."""

    epoch: int
    optimization_loss_before_step: float
    fresh_training_mean_nll: float
    validation_mean_nll: float
    gradient_norm_before_step: float
    learning_rate_after_scheduler: float
    training_checkpoint_improved: bool
    raw_parameters_after_step: tuple[float, ...]
    constrained_parameters_after_step: tuple[float, ...]


@dataclass(frozen=True)
class StrategyEpochRecord(EpochRecord):
    """Extended diagnostics for an L-BFGS or hybrid optimizer step."""

    optimizer_phase: str
    phase_step: int
    training_objective_evaluation_count: int
    gradient_norm_after_step: float | None = None
    closure_evaluation_count: int = 0


@dataclass(frozen=True)
class BoundaryProximity:
    """Distances of fitted constrained values from their nearest lower/upper walls."""

    w_from_zero: float
    s0_from_zero: float
    ell_short_fraction_to_nearest_boundary: float
    ell_long_fraction_to_nearest_boundary: float


@dataclass(frozen=True)
class M1BoundaryProximity(BoundaryProximity):
    """M0 diagnostics plus biased-hazard distances and its mean duration.

    ``subjective_mean_block_duration_trials`` is the legacy-compatible name
    for the biased-block value ``1/h_B``; neutral has mean ``2/h_B``.
    """

    hazard_from_zero: float
    hazard_from_one: float
    hazard_fraction_to_nearest_boundary: float
    subjective_mean_block_duration_trials: float


@dataclass(frozen=True)
class M2BoundaryProximity(M1BoundaryProximity):
    """M1 diagnostics plus M2's contrast and implied subjective prototypes."""

    delta0_from_zero: float
    theta_short: float
    theta_long: float
    delta0_distance_from_log4: float


@dataclass(frozen=True)
class FitRunResult:
    """Complete serializable record for one initialization run."""

    result_schema_version: str
    model_id: str
    mouse_id: str
    registry_config_sha256: str
    model_registry_snapshot: dict[str, object]
    training_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    training_valid_trial_count: int
    validation_valid_trial_count: int
    start_id: int
    start_purpose: str
    random_seed: int
    initial_constrained_parameters: tuple[float, ...]
    initial_raw_parameters: tuple[float, ...]
    initial_training_mean_nll: float
    initial_validation_mean_nll: float
    best_raw_parameters: tuple[float, ...]
    best_constrained_parameters: tuple[float, ...]
    best_fresh_training_mean_nll: float
    recalculated_best_training_mean_nll: float
    final_validation_mean_nll: float
    gradient_norm_at_restored_best: float
    best_epoch: int
    final_epoch: int
    final_learning_rate: float
    optimizer_step_count: int
    termination_reason: str
    converged: bool
    convergence_checks: tuple[tuple[str, bool], ...]
    history: tuple[EpochRecord, ...]
    training_history: tuple[tuple[int, float], ...]
    validation_history: tuple[tuple[int, float], ...]
    boundary_proximity: BoundaryProximity | M1BoundaryProximity | M2BoundaryProximity
    nonfinite_event_count: int
    registered_max_epochs: int
    executed_max_epochs: int
    is_registered_duration_run: bool
    registered_initial_learning_rate: float
    executed_initial_learning_rate: float
    is_registered_learning_rate_run: bool
    exception_message: str | None

    def as_dict(self) -> dict:
        """Return a JSON-safe detailed fit record."""

        return asdict(self)


@dataclass(frozen=True)
class AlternativeFitRunResult(FitRunResult):
    """Fit record additions used only by opt-in L-BFGS strategies."""

    optimizer_strategy: str
    optimizer_settings: dict[str, object]
    adam_step_count: int
    lbfgs_step_count: int
    training_objective_evaluation_count: int
    phase_switch_epoch: int | None
    phase_switch_training_mean_nll: float | None
    best_optimizer_phase: str
    convergence_budget_sufficient: bool
    stability_history_source: str


@dataclass(frozen=True)
class LBFGSSettings:
    """Explicit controls for recorded one-iteration L-BFGS outer steps."""

    maximum_steps: int = 250
    learning_rate: float = 1.0
    history_size: int = 10
    maximum_evaluations_per_step: int = 20
    tolerance_grad: float = 1e-9
    tolerance_change: float = 1e-12
    line_search_fn: str = "strong_wolfe"

    def __post_init__(self) -> None:
        """Reject settings that cannot define a finite L-BFGS run."""

        if self.maximum_steps <= 0:
            raise ValueError("L-BFGS maximum_steps must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("L-BFGS learning_rate must be finite and positive")
        if self.history_size <= 0:
            raise ValueError("L-BFGS history_size must be positive")
        if self.maximum_evaluations_per_step <= 0:
            raise ValueError("L-BFGS maximum evaluations must be positive")
        if not np.isfinite(self.tolerance_grad) or self.tolerance_grad <= 0.0:
            raise ValueError("L-BFGS tolerance_grad must be finite and positive")
        if not np.isfinite(self.tolerance_change) or self.tolerance_change <= 0.0:
            raise ValueError("L-BFGS tolerance_change must be finite and positive")
        if self.line_search_fn != "strong_wolfe":
            raise ValueError("L-BFGS line_search_fn must be 'strong_wolfe'")

    def as_dict(self) -> dict[str, object]:
        """Return settings in a JSON-safe form."""

        return asdict(self)


@dataclass(frozen=True)
class MultiStartFitResult:
    """All eight run records plus training-based winner/ridge diagnostics."""

    result_schema_version: str
    model_id: str
    mouse_id: str
    runs: tuple[FitRunResult, ...]
    winner_start_id: int | None
    selection_status: str
    identifiability_ridge_detected: bool
    ridge_start_ids: tuple[int, ...]
    additional_starts_recommended: bool

    def as_dict(self) -> dict:
        """Return a JSON-safe multi-start summary with complete nested runs."""

        return asdict(self)


def _objective(
    sessions: tuple[SessionTrials, ...],
    raw_parameters: torch.Tensor,
    registry: ModelRegistry,
    *,
    role: str,
):
    """Evaluate the registered M0, M1, or M2 objective for one non-test split."""

    if registry.model_id == "M2_hmm_static_v1":
        from .m2_session import evaluate_m2_mouse_objective

        return evaluate_m2_mouse_objective(
            sessions,
            raw_parameters,
            registry.task_moments,
            objective_weighting=registry.objective_weighting,
            required_split_role=role,
            lapse_max=registry.lapse_max,
            parameter_epsilon=registry.parameter_epsilon,
            log_probability_epsilon=registry.log_probability_epsilon,
            prior_probability_epsilon=registry.prior_probability_epsilon,
            quadratic_tolerance=registry.quadratic_tolerance,
            discriminant_tolerance=registry.discriminant_tolerance,
            first_long_probability=registry.first_long_probability,
        )

    if registry.model_id == "M1_hmm_true_v1":
        from .m1_session import evaluate_m1_mouse_objective

        return evaluate_m1_mouse_objective(
            sessions,
            raw_parameters,
            registry.task_moments,
            objective_weighting=registry.objective_weighting,
            required_split_role=role,
            lapse_max=registry.lapse_max,
            parameter_epsilon=registry.parameter_epsilon,
            log_probability_epsilon=registry.log_probability_epsilon,
            prior_probability_epsilon=registry.prior_probability_epsilon,
            quadratic_tolerance=registry.quadratic_tolerance,
            discriminant_tolerance=registry.discriminant_tolerance,
            first_long_probability=registry.first_long_probability,
        )

    return evaluate_mouse_objective(
        sessions,
        raw_parameters,
        registry.task_moments,
        objective_weighting=registry.objective_weighting,
        required_split_role=role,  # type: ignore[arg-type]
        lapse_max=registry.lapse_max,
        parameter_epsilon=registry.parameter_epsilon,
        log_probability_epsilon=registry.log_probability_epsilon,
    )


def _registry_snapshot(registry: ModelRegistry) -> tuple[str, dict[str, object]]:
    """Return the exact JSON registry plus runtime data-policy provenance."""

    registry_bytes = registry.config_path.read_bytes()
    snapshot = json.loads(registry_bytes.decode("utf-8"))
    if registry.model_id in {"M1_hmm_true_v1", "M2_hmm_static_v1"} and "extends" in snapshot:
        if registry.model_id == "M1_hmm_true_v1":
            from ..m1_registry import _read_config
        else:
            from ..m2_registry import _read_config

        overlay = snapshot
        snapshot = _read_config(registry.config_path)
        snapshot["subject_registry_overlay"] = overlay
    snapshot["resolved_artifacts"] = {
        "initialization_table_sha256": registry.initialization_table_sha256,
        "split_assignment_sha256": registry.split.assignment_sha256,
    }
    if registry.model_id in {"M1_hmm_true_v1", "M2_hmm_static_v1"}:
        snapshot["binary_decision_data_policy"] = {
            "invalid_or_omitted_choice": (
                "retain timeline row; omit its Bernoulli choice term; preserve causal "
                "feedback/update/transition handling"
            ),
            "current_category": "post-choice update only; never a current prediction input",
            "true_block_and_run_length": "diagnostic only; never HMM filter inputs",
            "RewardNaive": (
                "use as Reward for binary choice reconstruction; retain original label"
            ),
            "PunishNaive": (
                "use as Punish for binary choice reconstruction; retain original label"
            ),
        }
    else:
        snapshot["binary_decision_data_policy"] = {
            "invalid_or_omitted_choice": (
                "retain timeline row as NaN; skip before model, likelihood, gradient, "
                "and metric calculations"
            ),
            "RewardNaive": (
                "use as Reward for binary choice reconstruction; retain original label"
            ),
            "PunishNaive": (
                "use as Punish for binary choice reconstruction; retain original label"
            ),
        }
    return sha256(registry_bytes).hexdigest(), snapshot


def _gradient_norm(parameter: torch.Tensor) -> float:
    """Return the Euclidean gradient norm and reject a missing gradient."""

    if parameter.grad is None:
        raise RuntimeError("Raw parameter gradient is missing")
    return float(torch.linalg.vector_norm(parameter.grad).detach().cpu())


def _constrained_tuple(
    raw_parameters: torch.Tensor,
    registry: ModelRegistry,
) -> tuple[float, ...]:
    """Decode a registered M0/M1/M2 raw vector to fitted coordinates."""

    if registry.model_id == "M2_hmm_static_v1":
        from .m2_parameters import decode_m2_raw_parameters

        constrained = decode_m2_raw_parameters(
            raw_parameters,
            lapse_max=registry.lapse_max,
            epsilon=registry.parameter_epsilon,
        ).as_tensor()
        return tuple(float(value) for value in constrained.detach().cpu())

    if registry.model_id == "M1_hmm_true_v1":
        from .m1_parameters import decode_m1_raw_parameters

        constrained = decode_m1_raw_parameters(
            raw_parameters,
            lapse_max=registry.lapse_max,
            epsilon=registry.parameter_epsilon,
        ).as_tensor()
        return tuple(float(value) for value in constrained.detach().cpu())

    constrained = decode_raw_parameters(
        raw_parameters,
        lapse_max=registry.lapse_max,
        epsilon=registry.parameter_epsilon,
    ).as_tensor()
    return tuple(float(value) for value in constrained.detach().cpu())  # type: ignore[return-value]


def _tensor_tuple(tensor: torch.Tensor) -> tuple[float, ...]:
    """Convert a one-dimensional parameter tensor to immutable Python floats."""

    return tuple(float(value) for value in tensor.detach().cpu())  # type: ignore[return-value]


def _boundary_proximity(
    constrained: tuple[float, ...],
    lapse_max: float,
) -> BoundaryProximity:
    """Calculate interpretable distances from registered parameter boundaries."""

    if len(constrained) < 4:
        raise ValueError("Boundary diagnostics require common (w,s0,ell_short,ell_long) values")
    w, s0, ell_short, ell_long = constrained[:4]
    common = dict(
        w_from_zero=w,
        s0_from_zero=s0,
        ell_short_fraction_to_nearest_boundary=(
            min(ell_short, lapse_max - ell_short) / lapse_max
        ),
        ell_long_fraction_to_nearest_boundary=(
            min(ell_long, lapse_max - ell_long) / lapse_max
        ),
    )
    if len(constrained) == 6:
        ah = constrained[5]
        hazard = float(1.0 / (1.0 + np.exp(-ah)))
        return M1BoundaryProximity(
            **common,
            hazard_from_zero=hazard,
            hazard_from_one=1.0 - hazard,
            hazard_fraction_to_nearest_boundary=min(hazard, 1.0 - hazard),
            subjective_mean_block_duration_trials=1.0 / hazard,
        )
    if len(constrained) == 7:
        ah, delta0 = constrained[5:7]
        hazard = float(1.0 / (1.0 + np.exp(-ah)))
        theta_long = float(1.0 / (1.0 + np.exp(-delta0)))
        return M2BoundaryProximity(
            **common,
            hazard_from_zero=hazard,
            hazard_from_one=1.0 - hazard,
            hazard_fraction_to_nearest_boundary=min(hazard, 1.0 - hazard),
            subjective_mean_block_duration_trials=1.0 / hazard,
            delta0_from_zero=delta0,
            theta_short=1.0 - theta_long,
            theta_long=theta_long,
            delta0_distance_from_log4=abs(delta0 - np.log(4.0)),
        )
    return BoundaryProximity(**common)


def _stable_training_history(
    history: tuple[EpochRecord, ...] | list[EpochRecord],
    *,
    window: int,
    tolerance: float,
) -> bool:
    """Return whether the last registered window has sufficiently stable NLL."""

    if len(history) < window:
        return False
    values = np.asarray(
        [record.fresh_training_mean_nll for record in history[-window:]],
        dtype=float,
    )
    return bool(np.all(np.isfinite(values)) and np.ptp(values) <= tolerance)


def fit_single_initialization(
    training_sessions: Iterable[SessionTrials],
    validation_sessions: Iterable[SessionTrials],
    registry: ModelRegistry,
    initialization: InitializationSpec,
    *,
    max_epochs_override: int | None = None,
    initial_learning_rate_override: float | None = None,
    initial_callback: Callable[[float, float], None] | None = None,
    epoch_callback: Callable[[EpochRecord], None] | None = None,
) -> FitRunResult:
    r"""Fit one M0 start with one Adam step per complete training epoch.

    Purpose
    -------
    Execute the registered Part VII procedure, restore the best checkpoint by
    fresh training mean NLL, and return every diagnostic needed for Part VIII.

    Mathematical definition
    -----------------------
    Each epoch differentiates the complete pooled objective

    .. math::

       J_{train}(\widetilde\Theta)=
       \frac{\sum_dJ_d(\widetilde\Theta)}{\sum_dN_d}.

    Adam updates the one raw pooled vector once. The updated vector is then
    freshly evaluated on all training and validation sessions. Validation never
    enters gradients, scheduler monitoring, checkpoint selection, or winner
    selection.

    Parameter interpretation
    ------------------------
    Parameters are pooled within one mouse. Session-local state is never pooled;
    M0 creates and discards an empty state at every session boundary.

    Input units and shapes
    ----------------------
    Training and validation inputs are nonempty one-mouse session iterables in
    ``torch.float64``. The initialization stores ``(w,s0,ell_short,ell_long,kappa)``.

    Returned quantities and shapes
    ------------------------------
    Returns one immutable :class:`FitRunResult` with raw/constrained checkpoints,
    epoch histories, convergence checks, boundary diagnostics, and seed.

    Constraints
    -----------
    There is exactly one optimizer step per complete epoch and no optimizer step
    inside a session loop. All sessions contribute before the single
    ``backward()`` call. Test-tagged sessions are rejected upstream.

    Numerical stability
    -------------------
    Every loss, raw vector, and gradient is checked for finiteness. Only
    likelihood-log probabilities are clamped. ReduceLROnPlateau monitors the
    freshly recalculated training mean NLL.

    Causal restrictions
    -------------------
    Training gradients use training sessions only. Validation is evaluated under
    ``torch.no_grad`` and cannot affect model updates or checkpoint choice.

    Important edge cases
    --------------------
    Runtime epoch and learning-rate overrides are recorded explicitly and mark
    the corresponding setting as non-registered. Reaching maximum epochs alone
    never implies convergence. Callbacks are reporting-only hooks invoked after
    fresh values exist; they must not mutate tensors. Failures produce a
    complete non-converged run record when a finite initial checkpoint exists.

    Related tests
    -------------
    ``test_single_start_takes_one_step_per_epoch_and_restores_best`` and
    ``test_validation_choices_do_not_change_training_updates``.
    """

    train = tuple(training_sessions)
    validation = tuple(validation_sessions)
    if not train or not validation:
        raise ValueError("Training and validation session sets must be nonempty")
    if {session.mouse_id for session in (*train, *validation)} != {train[0].mouse_id}:
        raise ValueError("Training and validation sessions must belong to one mouse")
    if initialization not in registry.initializations:
        raise ValueError("Initialization is not part of the frozen registry table")
    registered_epochs = registry.optimizer.maximum_epochs
    executed_epochs = (
        registered_epochs if max_epochs_override is None else max_epochs_override
    )
    if executed_epochs <= 0:
        raise ValueError("Executed maximum epochs must be positive")
    registered_learning_rate = registry.optimizer.initial_learning_rate
    executed_learning_rate = (
        registered_learning_rate
        if initial_learning_rate_override is None
        else initial_learning_rate_override
    )
    if not np.isfinite(executed_learning_rate) or executed_learning_rate <= 0.0:
        raise ValueError("Executed initial learning rate must be finite and positive")

    random_seed = registry.seeds.global_seed + initialization.start_id
    torch.manual_seed(random_seed)
    initial_raw = initialization.raw_tensor(
        lapse_max=registry.lapse_max,
        epsilon=registry.parameter_epsilon,
    )
    raw = torch.nn.Parameter(initial_raw.clone())
    optimizer = torch.optim.Adam(
        [raw],
        lr=executed_learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=registry.scheduler.factor,
        patience=registry.scheduler.patience_epochs,
        threshold=registry.convergence.meaningful_improvement_tolerance,
        threshold_mode="abs",
        min_lr=registry.scheduler.minimum_learning_rate,
    )

    with torch.no_grad():
        initial_train = _objective(train, raw, registry, role="train")
        initial_validation = _objective(
            validation,
            raw,
            registry,
            role="validation",
        )
    registry_checksum, registry_snapshot = _registry_snapshot(registry)
    best_raw = raw.detach().clone()
    best_train = float(initial_train.loss.detach().cpu())
    initial_train_value = best_train
    initial_validation_value = float(initial_validation.loss.detach().cpu())
    if initial_callback is not None:
        initial_callback(initial_train_value, initial_validation_value)
    best_epoch = 0
    history: list[EpochRecord] = []
    nonfinite_events = 0
    optimizer_steps = 0
    epochs_without_improvement = 0
    exception_message: str | None = None
    termination_reason = "maximum_epochs_reached"

    for epoch in range(1, executed_epochs + 1):
        try:
            optimizer.zero_grad(set_to_none=True)
            objective = _objective(train, raw, registry, role="train")
            if not bool(torch.isfinite(objective.loss).detach().cpu()):
                raise FloatingPointError("Optimization loss is nonfinite")
            optimization_loss = float(objective.loss.detach().cpu())
            objective.loss.backward()
            if raw.grad is None or not bool(torch.all(torch.isfinite(raw.grad))):
                raise FloatingPointError("Raw parameter gradient is missing/nonfinite")
            gradient_norm = _gradient_norm(raw)
            optimizer.step()
            optimizer_steps += 1
            if not bool(torch.all(torch.isfinite(raw)).detach().cpu()):
                raise FloatingPointError("Adam produced nonfinite raw parameters")

            with torch.no_grad():
                fresh_train = _objective(train, raw, registry, role="train")
                fresh_validation = _objective(
                    validation,
                    raw,
                    registry,
                    role="validation",
                )
            fresh_train_value = float(fresh_train.loss.detach().cpu())
            validation_value = float(fresh_validation.loss.detach().cpu())
            if not np.isfinite(fresh_train_value) or not np.isfinite(validation_value):
                raise FloatingPointError(
                    "Fresh train/validation objective is nonfinite"
                )

            improved = (
                fresh_train_value
                < best_train - registry.convergence.meaningful_improvement_tolerance
            )
            # Checkpoint on ANY strict improvement, independent of the
            # meaningful-improvement tolerance used for patience below. Near
            # a genuine optimum, NLL stops improving by more than that
            # tolerance well before the gradient norm reaches its own
            # (unrelated) tolerance, so gating the checkpoint on `improved`
            # can restore a stale, larger-gradient point over a later point
            # that is both lower-NLL and much closer to a stationary point.
            if fresh_train_value < best_train:
                best_train = fresh_train_value
                best_raw = raw.detach().clone()
                best_epoch = epoch
            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler.step(fresh_train_value)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            record = EpochRecord(
                epoch=epoch,
                optimization_loss_before_step=optimization_loss,
                fresh_training_mean_nll=fresh_train_value,
                validation_mean_nll=validation_value,
                gradient_norm_before_step=gradient_norm,
                learning_rate_after_scheduler=learning_rate,
                training_checkpoint_improved=improved,
                raw_parameters_after_step=_tensor_tuple(raw.detach()),
                constrained_parameters_after_step=_constrained_tuple(
                    raw.detach(), registry
                ),
            )
            history.append(record)
            if epoch_callback is not None:
                epoch_callback(record)

            stable = _stable_training_history(
                history,
                window=registry.convergence.stable_training_window_epochs,
                tolerance=registry.convergence.meaningful_improvement_tolerance,
            )
            if (
                learning_rate <= registry.scheduler.minimum_learning_rate
                and epochs_without_improvement
                >= registry.convergence.no_improvement_patience_epochs
            ):
                termination_reason = "minimum_lr_plateau_reached"
                break
        except (FloatingPointError, RuntimeError, ValueError) as error:
            nonfinite_events += int(isinstance(error, FloatingPointError))
            exception_message = f"{type(error).__name__}: {error}"
            termination_reason = "numerical_or_transform_failure"
            break

    final_epoch = history[-1].epoch if history else 0
    with torch.no_grad():
        raw.copy_(best_raw)
    optimizer.zero_grad(set_to_none=True)
    restored_train = _objective(train, raw, registry, role="train")
    restored_train.loss.backward()
    restored_gradient_norm = _gradient_norm(raw)
    with torch.no_grad():
        restored_validation = _objective(
            validation,
            raw,
            registry,
            role="validation",
        )
    recalculated_train = float(restored_train.loss.detach().cpu())
    final_validation = float(restored_validation.loss.detach().cpu())
    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    stable = _stable_training_history(
        history,
        window=registry.convergence.stable_training_window_epochs,
        tolerance=registry.convergence.meaningful_improvement_tolerance,
    )
    checks = (
        ("finite_objective", bool(np.isfinite(recalculated_train))),
        ("finite_parameters", bool(torch.all(torch.isfinite(raw)).detach().cpu())),
        (
            "finite_gradients",
            bool(raw.grad is not None and torch.all(torch.isfinite(raw.grad))),
        ),
        (
            "small_gradient_norm",
            restored_gradient_norm <= registry.convergence.gradient_norm_tolerance,
        ),
        ("stable_fresh_training_nll", stable),
        ("no_parameter_transform_errors", exception_message is None),
        (
            "no_probability_normalization_errors",
            exception_message is None and nonfinite_events == 0,
        ),
    )
    converged = all(value for _, value in checks)
    if termination_reason == "maximum_epochs_reached":
        termination_reason = (
            "maximum_epochs_convergence_checks_passed"
            if converged
            else "maximum_epochs_without_convergence"
        )

    best_constrained = _constrained_tuple(raw.detach(), registry)
    return FitRunResult(
        result_schema_version=registry.multi_start_selection.result_schema_version,
        model_id=registry.model_id,
        mouse_id=train[0].mouse_id,
        registry_config_sha256=registry_checksum,
        model_registry_snapshot=registry_snapshot,
        training_session_ids=tuple(session.session_id for session in train),
        validation_session_ids=tuple(session.session_id for session in validation),
        training_valid_trial_count=initial_train.total_valid_trial_count,
        validation_valid_trial_count=initial_validation.total_valid_trial_count,
        start_id=initialization.start_id,
        start_purpose=initialization.purpose,
        random_seed=random_seed,
        initial_constrained_parameters=_tensor_tuple(
            initialization.constrained_tensor()
        ),
        initial_raw_parameters=_tensor_tuple(initial_raw),
        initial_training_mean_nll=initial_train_value,
        initial_validation_mean_nll=initial_validation_value,
        best_raw_parameters=_tensor_tuple(best_raw),
        best_constrained_parameters=best_constrained,
        best_fresh_training_mean_nll=best_train,
        recalculated_best_training_mean_nll=recalculated_train,
        final_validation_mean_nll=final_validation,
        gradient_norm_at_restored_best=restored_gradient_norm,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        final_learning_rate=final_learning_rate,
        optimizer_step_count=optimizer_steps,
        termination_reason=termination_reason,
        converged=converged,
        convergence_checks=checks,
        history=tuple(history),
        training_history=tuple(
            (record.epoch, record.fresh_training_mean_nll) for record in history
        ),
        validation_history=tuple(
            (record.epoch, record.validation_mean_nll) for record in history
        ),
        boundary_proximity=_boundary_proximity(
            best_constrained,
            registry.lapse_max,
        ),
        nonfinite_event_count=nonfinite_events,
        registered_max_epochs=registered_epochs,
        executed_max_epochs=executed_epochs,
        is_registered_duration_run=max_epochs_override is None,
        registered_initial_learning_rate=registered_learning_rate,
        executed_initial_learning_rate=executed_learning_rate,
        is_registered_learning_rate_run=initial_learning_rate_override is None,
        exception_message=exception_message,
    )


def _strategy_epoch_record(
    record: EpochRecord,
    *,
    optimizer_phase: str,
    phase_step: int,
    training_objective_evaluation_count: int,
) -> StrategyEpochRecord:
    """Add optimizer provenance without changing the registered Adam record."""

    return StrategyEpochRecord(
        epoch=record.epoch,
        optimization_loss_before_step=record.optimization_loss_before_step,
        fresh_training_mean_nll=record.fresh_training_mean_nll,
        validation_mean_nll=record.validation_mean_nll,
        gradient_norm_before_step=record.gradient_norm_before_step,
        learning_rate_after_scheduler=record.learning_rate_after_scheduler,
        training_checkpoint_improved=record.training_checkpoint_improved,
        raw_parameters_after_step=record.raw_parameters_after_step,
        constrained_parameters_after_step=record.constrained_parameters_after_step,
        optimizer_phase=optimizer_phase,
        phase_step=phase_step,
        training_objective_evaluation_count=training_objective_evaluation_count,
    )


def fit_single_initialization_with_strategy(
    training_sessions: Iterable[SessionTrials],
    validation_sessions: Iterable[SessionTrials],
    registry: ModelRegistry,
    initialization: InitializationSpec,
    *,
    optimizer_strategy: OptimizerStrategy = "adam",
    max_epochs_override: int | None = None,
    initial_learning_rate_override: float | None = None,
    adam_warmup_epochs: int = 1000,
    lbfgs_settings: LBFGSSettings | None = None,
    allow_short_convergence_budget: bool = False,
    initial_callback: Callable[[float, float], None] | None = None,
    epoch_callback: Callable[[EpochRecord], None] | None = None,
) -> FitRunResult:
    r"""Fit one frozen start with Adam, L-BFGS, or Adam then L-BFGS.

    Purpose
    -------
    Add opt-in full-batch L-BFGS strategies while delegating the default Adam
    strategy unchanged to :func:`fit_single_initialization`.

    Mathematical definition
    -----------------------
    Every strategy minimizes the identical pooled training objective

    .. math::

       J_{train}(\widetilde\Theta)=
       \frac{\sum_dJ_d(\widetilde\Theta)}{\sum_dN_d}.

    Parameter interpretation
    ------------------------
    All optimizers update the same raw ``(w,s0,ell_short,ell_long,log_kappa)`` vector.
    The hybrid strategy restores Adam's best training checkpoint before
    beginning L-BFGS refinement.

    Input units and shapes
    ----------------------
    Sessions contain one-dimensional ``float64`` durations in seconds and
    binary/NaN choices. One frozen constrained initialization supplies five
    scalar parameters.

    Returned quantities and shapes
    ------------------------------
    Adam returns its original :class:`FitRunResult`. Alternative strategies
    return a compatible extended record with phase and closure-evaluation data.

    Constraints
    -----------
    ``max_epochs_override`` retains its original Adam-only meaning. L-BFGS
    uses explicit :class:`LBFGSSettings`; hybrid Adam duration is explicit.
    Normal alternative runs must budget enough combined recorded steps to fill
    the registered stability window. Bounded smoke tests may explicitly waive
    that requirement while remaining ineligible for convergence.

    Numerical stability
    -------------------
    L-BFGS uses recorded one-iteration outer steps and strong-Wolfe line search.
    A failed candidate restores the finite raw vector from before that step.

    Causal restrictions
    -------------------
    Closures, line search, checkpointing, and phase switching use training data
    only. Validation is freshly evaluated under ``torch.no_grad`` for reporting.

    Important edge cases
    --------------------
    Pure L-BFGS starts directly from the frozen initialization. Hybrid fitting
    does not proceed to L-BFGS after an Adam numerical or transform failure.

    Related tests
    -------------
    ``test_pure_lbfgs_preserves_constraints_and_records_closures``,
    ``test_hybrid_has_two_phases_and_training_only_updates``, and the unchanged
    Part VII Adam regression tests.
    """

    if optimizer_strategy == "adam":
        return fit_single_initialization(
            training_sessions,
            validation_sessions,
            registry,
            initialization,
            max_epochs_override=max_epochs_override,
            initial_learning_rate_override=initial_learning_rate_override,
            initial_callback=initial_callback,
            epoch_callback=epoch_callback,
        )
    if optimizer_strategy not in {"lbfgs", "hybrid"}:
        raise ValueError(f"Unknown optimizer strategy: {optimizer_strategy!r}")
    if max_epochs_override is not None:
        raise ValueError("max_epochs_override applies only to the Adam strategy")
    if optimizer_strategy == "lbfgs" and initial_learning_rate_override is not None:
        raise ValueError(
            "initial_learning_rate_override applies to Adam; configure L-BFGS "
            "through LBFGSSettings"
        )
    if adam_warmup_epochs <= 0:
        raise ValueError("adam_warmup_epochs must be positive")
    settings = LBFGSSettings() if lbfgs_settings is None else lbfgs_settings
    planned_history_length = settings.maximum_steps + (
        adam_warmup_epochs if optimizer_strategy == "hybrid" else 0
    )
    convergence_budget_sufficient = (
        planned_history_length
        >= registry.convergence.stable_training_window_epochs
    )
    if not convergence_budget_sufficient and not allow_short_convergence_budget:
        raise ValueError(
            "Optimizer step budget cannot fill the registered stability window: "
            f"planned={planned_history_length}, "
            f"required={registry.convergence.stable_training_window_epochs}. "
            "Increase the phase budget or explicitly allow a short smoke run."
        )

    train = tuple(training_sessions)
    validation = tuple(validation_sessions)
    if not train or not validation:
        raise ValueError("Training and validation session sets must be nonempty")
    if {session.mouse_id for session in (*train, *validation)} != {train[0].mouse_id}:
        raise ValueError("Training and validation sessions must belong to one mouse")
    if initialization not in registry.initializations:
        raise ValueError("Initialization is not part of the frozen registry table")

    adam_result: FitRunResult | None = None
    history: list[StrategyEpochRecord] = []
    training_evaluations = 0
    adam_steps = 0
    phase_switch_epoch: int | None = None
    phase_switch_nll: float | None = None
    exception_message: str | None = None
    nonfinite_events = 0

    if optimizer_strategy == "hybrid":

        def adam_epoch_callback(record: EpochRecord) -> None:
            extended = _strategy_epoch_record(
                record,
                optimizer_phase="adam",
                phase_step=record.epoch,
                training_objective_evaluation_count=1 + 2 * record.epoch,
            )
            if epoch_callback is not None:
                epoch_callback(extended)

        adam_result = fit_single_initialization(
            train,
            validation,
            registry,
            initialization,
            max_epochs_override=adam_warmup_epochs,
            initial_learning_rate_override=initial_learning_rate_override,
            initial_callback=initial_callback,
            epoch_callback=adam_epoch_callback,
        )
        history.extend(
            _strategy_epoch_record(
                record,
                optimizer_phase="adam",
                phase_step=record.epoch,
                training_objective_evaluation_count=1 + 2 * record.epoch,
            )
            for record in adam_result.history
        )
        adam_steps = adam_result.optimizer_step_count
        training_evaluations = 2 + 2 * len(adam_result.history)
        phase_switch_epoch = adam_result.final_epoch
        phase_switch_nll = adam_result.recalculated_best_training_mean_nll
        raw = torch.nn.Parameter(
            torch.tensor(adam_result.best_raw_parameters, dtype=torch.float64)
        )
        initial_train_value = adam_result.initial_training_mean_nll
        initial_validation_value = adam_result.initial_validation_mean_nll
        training_valid_count = adam_result.training_valid_trial_count
        validation_valid_count = adam_result.validation_valid_trial_count
        best_raw = raw.detach().clone()
        best_train = adam_result.best_fresh_training_mean_nll
        best_epoch = adam_result.best_epoch
        best_phase = "initial" if best_epoch == 0 else "adam"
        exception_message = adam_result.exception_message
        nonfinite_events = adam_result.nonfinite_event_count
        registry_checksum = adam_result.registry_config_sha256
        registry_snapshot = dict(adam_result.model_registry_snapshot)
    else:
        random_seed = registry.seeds.global_seed + initialization.start_id
        torch.manual_seed(random_seed)
        initial_raw = initialization.raw_tensor(
            lapse_max=registry.lapse_max,
            epsilon=registry.parameter_epsilon,
        )
        raw = torch.nn.Parameter(initial_raw.clone())
        with torch.no_grad():
            initial_train = _objective(train, raw, registry, role="train")
            initial_validation = _objective(
                validation,
                raw,
                registry,
                role="validation",
            )
        training_evaluations = 1
        initial_train_value = float(initial_train.loss.detach().cpu())
        initial_validation_value = float(initial_validation.loss.detach().cpu())
        training_valid_count = initial_train.total_valid_trial_count
        validation_valid_count = initial_validation.total_valid_trial_count
        best_raw = raw.detach().clone()
        best_train = initial_train_value
        best_epoch = 0
        best_phase = "initial"
        registry_checksum, registry_snapshot = _registry_snapshot(registry)
        if initial_callback is not None:
            initial_callback(initial_train_value, initial_validation_value)

    runtime_settings: dict[str, object] = {
        "strategy": optimizer_strategy,
        "lbfgs": settings.as_dict(),
        "planned_history_length": planned_history_length,
        "stability_window": registry.convergence.stable_training_window_epochs,
        "convergence_budget_sufficient": convergence_budget_sufficient,
        "short_budget_explicitly_allowed": allow_short_convergence_budget,
    }
    if optimizer_strategy == "hybrid":
        runtime_settings["adam_warmup"] = {
            "maximum_epochs": adam_warmup_epochs,
            "initial_learning_rate": (
                registry.optimizer.initial_learning_rate
                if initial_learning_rate_override is None
                else initial_learning_rate_override
            ),
            "scheduler": "registered_reduce_on_plateau",
        }
    registry_snapshot["executed_optimizer"] = runtime_settings

    lbfgs_records: list[StrategyEpochRecord] = []
    lbfgs_steps = 0
    termination_reason = "adam_phase_failure_before_lbfgs"
    if exception_message is None:
        optimizer = torch.optim.LBFGS(
            [raw],
            lr=settings.learning_rate,
            max_iter=1,
            max_eval=settings.maximum_evaluations_per_step,
            tolerance_grad=settings.tolerance_grad,
            tolerance_change=settings.tolerance_change,
            history_size=settings.history_size,
            line_search_fn=settings.line_search_fn,
        )
        termination_reason = "maximum_lbfgs_steps_reached"
        global_offset = len(history)
        epochs_without_improvement = 0
        for phase_step in range(1, settings.maximum_steps + 1):
            before_step = raw.detach().clone()
            try:
                first_closure_loss: float | None = None
                first_closure_gradient_norm: float | None = None
                closure_evaluations_this_step = 0

                def closure() -> torch.Tensor:
                    nonlocal first_closure_gradient_norm
                    nonlocal first_closure_loss
                    nonlocal closure_evaluations_this_step
                    nonlocal training_evaluations
                    optimizer.zero_grad(set_to_none=True)
                    objective = _objective(train, raw, registry, role="train")
                    training_evaluations += 1
                    closure_evaluations_this_step += 1
                    if not bool(torch.isfinite(objective.loss).detach().cpu()):
                        raise FloatingPointError("L-BFGS closure loss is nonfinite")
                    objective.loss.backward()
                    if raw.grad is None or not bool(
                        torch.all(torch.isfinite(raw.grad))
                    ):
                        raise FloatingPointError("L-BFGS closure gradient is nonfinite")
                    if first_closure_loss is None:
                        first_closure_loss = float(objective.loss.detach().cpu())
                        first_closure_gradient_norm = _gradient_norm(raw)
                    return objective.loss

                optimizer.step(closure)
                lbfgs_steps += 1
                if first_closure_loss is None or first_closure_gradient_norm is None:
                    raise RuntimeError(
                        "L-BFGS did not return finite first closure diagnostics"
                    )
                if not bool(torch.all(torch.isfinite(raw)).detach().cpu()):
                    raise FloatingPointError("L-BFGS produced nonfinite raw parameters")

                # Recompute the gradient fresh at the parameters L-BFGS
                # actually accepted. `raw.grad` right after `.step()` can
                # still hold the gradient from an earlier trial point
                # evaluated inside the strong-Wolfe line search rather than
                # the point the optimizer settled on (PyTorch's LBFGS only
                # restores parameter *data* between line-search trials, not
                # the `.grad` attribute), so reading it directly here can
                # misreport both the logged gradient norm and the
                # convergence-check decision below.
                optimizer.zero_grad(set_to_none=True)
                fresh_train = _objective(train, raw, registry, role="train")
                training_evaluations += 1
                fresh_train.loss.backward()
                if raw.grad is None or not bool(torch.all(torch.isfinite(raw.grad))):
                    raise FloatingPointError(
                        "L-BFGS post-step gradient is missing/nonfinite"
                    )
                gradient_norm_after_step = _gradient_norm(raw)
                fresh_train_value = float(fresh_train.loss.detach().cpu())
                with torch.no_grad():
                    fresh_validation = _objective(
                        validation,
                        raw,
                        registry,
                        role="validation",
                    )
                validation_value = float(fresh_validation.loss.detach().cpu())
                if not np.isfinite(fresh_train_value) or not np.isfinite(
                    validation_value
                ):
                    raise FloatingPointError(
                        "Fresh L-BFGS train/validation objective is nonfinite"
                    )
                improved = (
                    fresh_train_value
                    < best_train - registry.convergence.meaningful_improvement_tolerance
                )
                global_epoch = global_offset + phase_step
                # See the matching Adam-phase comment: checkpoint on any
                # strict improvement, and reserve the tolerance-gated
                # `improved` flag for the patience counter only.
                if fresh_train_value < best_train:
                    best_train = fresh_train_value
                    best_raw = raw.detach().clone()
                    best_epoch = global_epoch
                    best_phase = "lbfgs"
                if improved:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                record = StrategyEpochRecord(
                    epoch=global_epoch,
                    optimization_loss_before_step=first_closure_loss,
                    fresh_training_mean_nll=fresh_train_value,
                    validation_mean_nll=validation_value,
                    gradient_norm_before_step=first_closure_gradient_norm,
                    learning_rate_after_scheduler=float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    training_checkpoint_improved=improved,
                    raw_parameters_after_step=_tensor_tuple(raw.detach()),
                    constrained_parameters_after_step=_constrained_tuple(
                        raw.detach(), registry
                    ),
                    optimizer_phase="lbfgs",
                    phase_step=phase_step,
                    training_objective_evaluation_count=training_evaluations,
                    gradient_norm_after_step=gradient_norm_after_step,
                    closure_evaluation_count=closure_evaluations_this_step,
                )
                history.append(record)
                lbfgs_records.append(record)
                if epoch_callback is not None:
                    epoch_callback(record)
                stability_records = (
                    history if optimizer_strategy == "hybrid" else lbfgs_records
                )
                stable_now = _stable_training_history(
                    stability_records,
                    window=registry.convergence.stable_training_window_epochs,
                    tolerance=registry.convergence.meaningful_improvement_tolerance,
                )
                if (
                    stable_now
                    and gradient_norm_after_step
                    <= registry.convergence.gradient_norm_tolerance
                ):
                    termination_reason = "lbfgs_convergence_checks_passed"
                    break
                if (
                    epochs_without_improvement
                    >= registry.convergence.no_improvement_patience_epochs
                    and stable_now
                ):
                    termination_reason = "lbfgs_stable_plateau_reached"
                    break
            except (FloatingPointError, RuntimeError, ValueError) as error:
                with torch.no_grad():
                    raw.copy_(before_step)
                nonfinite_events += int(isinstance(error, FloatingPointError))
                exception_message = f"{type(error).__name__}: {error}"
                termination_reason = "lbfgs_line_search_or_transform_failure"
                break

    final_epoch = history[-1].epoch if history else 0
    with torch.no_grad():
        raw.copy_(best_raw)
    raw.grad = None
    restored_train = _objective(train, raw, registry, role="train")
    training_evaluations += 1
    restored_train.loss.backward()
    restored_gradient_norm = _gradient_norm(raw)
    with torch.no_grad():
        restored_validation = _objective(
            validation,
            raw,
            registry,
            role="validation",
        )
    recalculated_train = float(restored_train.loss.detach().cpu())
    final_validation = float(restored_validation.loss.detach().cpu())
    stability_records = history if optimizer_strategy == "hybrid" else lbfgs_records
    stability_history_source = (
        "combined_adam_lbfgs" if optimizer_strategy == "hybrid" else "lbfgs_only"
    )
    stable = _stable_training_history(
        stability_records,
        window=registry.convergence.stable_training_window_epochs,
        tolerance=registry.convergence.meaningful_improvement_tolerance,
    )
    checks = (
        ("finite_objective", bool(np.isfinite(recalculated_train))),
        ("finite_parameters", bool(torch.all(torch.isfinite(raw)).detach().cpu())),
        (
            "finite_gradients",
            bool(raw.grad is not None and torch.all(torch.isfinite(raw.grad))),
        ),
        (
            "small_gradient_norm",
            restored_gradient_norm <= registry.convergence.gradient_norm_tolerance,
        ),
        ("stable_fresh_training_nll", stable),
        ("no_parameter_transform_errors", exception_message is None),
        (
            "no_probability_normalization_errors",
            exception_message is None and nonfinite_events == 0,
        ),
    )
    converged = all(value for _, value in checks)
    if termination_reason == "maximum_lbfgs_steps_reached":
        termination_reason = (
            "maximum_lbfgs_steps_convergence_checks_passed"
            if converged
            else "maximum_lbfgs_steps_without_convergence"
        )

    best_constrained = _constrained_tuple(raw.detach(), registry)
    initial_raw_values = (
        adam_result.initial_raw_parameters
        if adam_result is not None
        else _tensor_tuple(
            initialization.raw_tensor(
                lapse_max=registry.lapse_max,
                epsilon=registry.parameter_epsilon,
            )
        )
    )
    executed_adam_lr = (
        registry.optimizer.initial_learning_rate
        if initial_learning_rate_override is None
        else initial_learning_rate_override
    )
    return AlternativeFitRunResult(
        result_schema_version=registry.multi_start_selection.result_schema_version,
        model_id=registry.model_id,
        mouse_id=train[0].mouse_id,
        registry_config_sha256=registry_checksum,
        model_registry_snapshot=registry_snapshot,
        training_session_ids=tuple(session.session_id for session in train),
        validation_session_ids=tuple(session.session_id for session in validation),
        training_valid_trial_count=training_valid_count,
        validation_valid_trial_count=validation_valid_count,
        start_id=initialization.start_id,
        start_purpose=initialization.purpose,
        random_seed=registry.seeds.global_seed + initialization.start_id,
        initial_constrained_parameters=_tensor_tuple(
            initialization.constrained_tensor()
        ),
        initial_raw_parameters=initial_raw_values,
        initial_training_mean_nll=initial_train_value,
        initial_validation_mean_nll=initial_validation_value,
        best_raw_parameters=_tensor_tuple(best_raw),
        best_constrained_parameters=best_constrained,
        best_fresh_training_mean_nll=best_train,
        recalculated_best_training_mean_nll=recalculated_train,
        final_validation_mean_nll=final_validation,
        gradient_norm_at_restored_best=restored_gradient_norm,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        final_learning_rate=settings.learning_rate,
        optimizer_step_count=adam_steps + lbfgs_steps,
        termination_reason=termination_reason,
        converged=converged,
        convergence_checks=checks,
        history=tuple(history),
        training_history=tuple(
            (record.epoch, record.fresh_training_mean_nll) for record in history
        ),
        validation_history=tuple(
            (record.epoch, record.validation_mean_nll) for record in history
        ),
        boundary_proximity=_boundary_proximity(
            best_constrained,
            registry.lapse_max,
        ),
        nonfinite_event_count=nonfinite_events,
        registered_max_epochs=registry.optimizer.maximum_epochs,
        executed_max_epochs=(
            settings.maximum_steps
            if optimizer_strategy == "lbfgs"
            else adam_warmup_epochs + settings.maximum_steps
        ),
        is_registered_duration_run=False,
        registered_initial_learning_rate=registry.optimizer.initial_learning_rate,
        executed_initial_learning_rate=(
            settings.learning_rate
            if optimizer_strategy == "lbfgs"
            else executed_adam_lr
        ),
        is_registered_learning_rate_run=False,
        exception_message=exception_message,
        optimizer_strategy=optimizer_strategy,
        optimizer_settings=runtime_settings,
        adam_step_count=adam_steps,
        lbfgs_step_count=lbfgs_steps,
        training_objective_evaluation_count=training_evaluations,
        phase_switch_epoch=phase_switch_epoch,
        phase_switch_training_mean_nll=phase_switch_nll,
        best_optimizer_phase=best_phase,
        convergence_budget_sufficient=convergence_budget_sufficient,
        stability_history_source=stability_history_source,
    )


def _parameter_distance(
    first: FitRunResult,
    second: FitRunResult,
    scale: tuple[float, ...],
) -> float:
    """Return scaled Euclidean distance between constrained best parameters."""

    difference = np.asarray(first.best_constrained_parameters) - np.asarray(
        second.best_constrained_parameters
    )
    return float(np.linalg.vector_norm(difference / np.asarray(scale)))


def select_winning_start(
    runs: Iterable[FitRunResult],
    registry: ModelRegistry,
) -> tuple[FitRunResult, bool, tuple[int, ...]]:
    r"""Select the lowest recalculated training NLL among converged runs.

    Purpose
    -------
    Apply the registered Part VIII winner criterion without validation-based
    model selection and flag near-equal but parameter-distant solutions.

    Mathematical definition
    -----------------------
    Select

    .. math::

       r^*=\arg\min_{r\in\mathcal C}J_{train,r}^{recalculated},

    where :math:`\mathcal C` contains only runs whose convergence checks pass.
    A ridge is flagged when another converged run is within the registered NLL
    tolerance but exceeds the scaled parameter-distance threshold.

    Parameter interpretation
    ------------------------
    Validation NLL is reported after selection but never appears in the argmin.

    Input units and shapes
    ----------------------
    ``runs`` contains one result per frozen initialization.

    Returned quantities and shapes
    ------------------------------
    Returns the winning run, ridge flag, and IDs of near-equal distant runs.

    Constraints
    -----------
    At least one run must be properly converged and all runs must share model
    and mouse identities.

    Numerical stability
    -------------------
    Winner comparison uses freshly recalculated ``float64`` training NLLs.

    Causal restrictions
    -------------------
    Validation and locked-test performance cannot influence the winner.

    Important edge cases
    --------------------
    No converged run raises. Additional starts are only recommended; this
    function never generates them automatically.

    Related tests
    -------------
    ``test_winner_uses_training_not_validation_and_flags_ridge``.
    """

    records = tuple(runs)
    converged = tuple(record for record in records if record.converged)
    if not converged:
        raise ValueError("No properly converged run is available for selection")
    if len({record.mouse_id for record in records}) != 1:
        raise ValueError("Multi-start runs must belong to one mouse")
    winner = min(
        converged,
        key=lambda record: record.recalculated_best_training_mean_nll,
    )
    ridge_ids: list[int] = []
    for candidate in converged:
        if candidate.start_id == winner.start_id:
            continue
        nll_difference = abs(
            candidate.recalculated_best_training_mean_nll
            - winner.recalculated_best_training_mean_nll
        )
        distance = _parameter_distance(
            winner,
            candidate,
            registry.multi_start_selection.parameter_distance_scale,
        )
        if (
            nll_difference
            <= registry.multi_start_selection.near_equal_training_nll_tolerance
            and distance
            >= registry.multi_start_selection.identifiability_parameter_distance_threshold
        ):
            ridge_ids.append(candidate.start_id)
    involved_ids = (winner.start_id, *ridge_ids) if ridge_ids else ()
    return winner, bool(ridge_ids), tuple(sorted(involved_ids))


def _initialize_multistart_worker(
    training: tuple[SessionTrials, ...],
    validation: tuple[SessionTrials, ...],
    registry,
    optimizer_strategy: OptimizerStrategy,
    max_epochs_override: int | None,
    initial_learning_rate_override: float | None,
    adam_warmup_epochs: int,
    lbfgs_settings: LBFGSSettings | None,
    allow_short_convergence_budget: bool,
    cpu_threads_per_worker: int,
    progress_queue,
) -> None:
    """Initialize one isolated M0/M1/M2 start process."""

    global _MULTISTART_WORKER_CONTEXT, _MULTISTART_WORKER_PROGRESS_QUEUE
    torch.set_num_threads(cpu_threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only once per process.
        pass
    _MULTISTART_WORKER_CONTEXT = (
        training,
        validation,
        registry,
        optimizer_strategy,
        max_epochs_override,
        initial_learning_rate_override,
        adam_warmup_epochs,
        lbfgs_settings,
        allow_short_convergence_budget,
    )
    _MULTISTART_WORKER_PROGRESS_QUEUE = progress_queue


def _fit_start_in_worker(initialization: InitializationSpec) -> FitRunResult:
    """Run the unchanged single-start fitter in one worker process."""

    if _MULTISTART_WORKER_CONTEXT is None:
        raise RuntimeError("Parallel multistart worker was not initialized")
    (
        training,
        validation,
        registry,
        optimizer_strategy,
        max_epochs_override,
        initial_learning_rate_override,
        adam_warmup_epochs,
        lbfgs_settings,
        allow_short_convergence_budget,
    ) = _MULTISTART_WORKER_CONTEXT
    initial_callback = None
    epoch_callback = None
    if _MULTISTART_WORKER_PROGRESS_QUEUE is not None:
        initial_callback = lambda train_nll, validation_nll: (
            _MULTISTART_WORKER_PROGRESS_QUEUE.put(
                ("initial", initialization.start_id, train_nll, validation_nll)
            )
        )
        epoch_callback = lambda record: _MULTISTART_WORKER_PROGRESS_QUEUE.put(
            ("epoch", initialization.start_id, record)
        )
    return fit_single_initialization_with_strategy(
        training,
        validation,
        registry,
        initialization,
        optimizer_strategy=optimizer_strategy,
        max_epochs_override=max_epochs_override,
        initial_learning_rate_override=initial_learning_rate_override,
        adam_warmup_epochs=adam_warmup_epochs,
        lbfgs_settings=lbfgs_settings,
        allow_short_convergence_budget=allow_short_convergence_budget,
        initial_callback=initial_callback,
        epoch_callback=epoch_callback,
    )


def fit_all_initializations(
    training_sessions: Iterable[SessionTrials],
    validation_sessions: Iterable[SessionTrials],
    registry: ModelRegistry,
    *,
    output_directory: str | Path | None = None,
    max_epochs_override: int | None = None,
    initial_learning_rate_override: float | None = None,
    optimizer_strategy: OptimizerStrategy = "adam",
    adam_warmup_epochs: int = 1000,
    lbfgs_settings: LBFGSSettings | None = None,
    allow_short_convergence_budget: bool = False,
    parallel_workers: int = 1,
    cpu_threads_per_worker: int = 1,
    initial_callback: Callable[[int, float, float], None] | None = None,
    epoch_callback: Callable[[int, EpochRecord], None] | None = None,
    completion_callback: Callable[[FitRunResult], None] | None = None,
) -> MultiStartFitResult:
    r"""Run all eight frozen starts and apply training-only selection.

    Purpose
    -------
    Execute the registered multi-start protocol with the selected optimizer,
    preserve every run even when it fails convergence, and optionally save
    every detailed result. Independent starts may run in separate processes;
    ``parallel_workers=1`` is the exact sequential reference path.

    Mathematical definition
    -----------------------
    For every frozen start :math:`r`, select, when possible,

    .. math::

       r^*=\arg\min_{r\in\mathcal C}J_{train,r}^{recalculated},

    where :math:`\mathcal C` contains only properly converged runs.

    Parameter interpretation
    ------------------------
    Starts differ in initial ``(w,s0,ell_short,ell_long,kappa)``. Each run fits one parameter
    vector pooled within the same mouse, with independent session state.

    Input units and shapes
    ----------------------
    Nonempty training and validation iterables contain ``float64`` durations in
    seconds. The frozen registry contains exactly eight starts.

    Returned quantities and shapes
    ------------------------------
    Returns one immutable summary with all full run records, an optional
    winner ID, and identifiability-ridge diagnostics.

    Constraints
    -----------
    Every frozen start runs once. Validation never determines the winner and
    additional starts are never generated automatically.

    Numerical stability
    -------------------
    Each run applies finiteness checks and recalculates its restored best training
    objective. Selection excludes runs failing any frozen convergence check.

    Causal restrictions
    -------------------
    Only training NLL controls checkpoints and selection. Validation is reported
    after fitting; locked-test sessions are rejected by the objective.

    Important edge cases
    --------------------
    If no run converges, all records remain available but no winner is declared.
    An epoch override explicitly marks outputs as non-registered smoke/test runs.
    Alternative optimizer strategies also remain explicitly non-registered.

    Related tests
    -------------
    ``test_multistart_runs_all_frozen_initializations`` and
    ``test_winner_uses_training_not_validation_and_flags_ridge``.
    """

    train = tuple(training_sessions)
    validation = tuple(validation_sessions)
    if not train or not validation:
        raise ValueError("Training and validation session sets must be nonempty")
    if not isinstance(parallel_workers, int) or parallel_workers <= 0:
        raise ValueError("parallel_workers must be a positive integer")
    if not isinstance(cpu_threads_per_worker, int) or cpu_threads_per_worker <= 0:
        raise ValueError("cpu_threads_per_worker must be a positive integer")
    initializations = tuple(registry.initializations)
    worker_count = min(
        parallel_workers,
        len(initializations),
        multiprocessing.cpu_count(),
    )
    runs: list[FitRunResult] = []
    output = Path(output_directory).expanduser().resolve() if output_directory else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    def save_completed_run(result: FitRunResult) -> None:
        if output is not None:
            save_fit_run_result(result, output / f"start_{result.start_id:02d}.json")
        if completion_callback is not None:
            completion_callback(result)

    if worker_count == 1:
        for initialization in initializations:
            result = fit_single_initialization_with_strategy(
                train,
                validation,
                registry,
                initialization,
                optimizer_strategy=optimizer_strategy,
                max_epochs_override=max_epochs_override,
                initial_learning_rate_override=initial_learning_rate_override,
                adam_warmup_epochs=adam_warmup_epochs,
                lbfgs_settings=lbfgs_settings,
                allow_short_convergence_budget=allow_short_convergence_budget,
                initial_callback=(
                    None
                    if initial_callback is None
                    else lambda train_nll, validation_nll, start_id=initialization.start_id: initial_callback(
                        start_id,
                        train_nll,
                        validation_nll,
                    )
                ),
                epoch_callback=(
                    None
                    if epoch_callback is None
                    else lambda record, start_id=initialization.start_id: epoch_callback(
                        start_id,
                        record,
                    )
                ),
            )
            runs.append(result)
            save_completed_run(result)
    else:
        spawn_context = multiprocessing.get_context("spawn")
        progress_queue = (
            spawn_context.Queue()
            if initial_callback is not None or epoch_callback is not None
            else None
        )
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
                        elif event[0] == "epoch" and epoch_callback is not None:
                            epoch_callback(event[1], event[2])
                    except BaseException as error:
                        callback_errors.append(error)

            progress_thread = threading.Thread(
                target=consume_progress,
                name="multistart-progress-consumer",
                daemon=True,
            )
            progress_thread.start()
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=spawn_context,
                initializer=_initialize_multistart_worker,
                initargs=(
                    train,
                    validation,
                    registry,
                    optimizer_strategy,
                    max_epochs_override,
                    initial_learning_rate_override,
                    adam_warmup_epochs,
                    lbfgs_settings,
                    allow_short_convergence_budget,
                    cpu_threads_per_worker,
                    progress_queue,
                ),
            ) as executor:
                futures = {
                    executor.submit(_fit_start_in_worker, initialization): initialization.start_id
                    for initialization in initializations
                }
                for future in as_completed(futures):
                    result = future.result()
                    runs.append(result)
                    save_completed_run(result)
        finally:
            if progress_queue is not None:
                progress_queue.put(("stop",))
                assert progress_thread is not None
                progress_thread.join()
                progress_queue.close()
                progress_queue.join_thread()
        if callback_errors:
            raise RuntimeError("Parallel multistart progress callback failed") from callback_errors[0]

    runs.sort(key=lambda result: result.start_id)

    try:
        winner, ridge, ridge_ids = select_winning_start(runs, registry)
        winner_id: int | None = winner.start_id
        status = "winner_selected_from_converged_training_objectives"
    except ValueError:
        winner_id = None
        ridge = False
        ridge_ids = ()
        status = "no_converged_run_no_winner"
    summary = MultiStartFitResult(
        result_schema_version=registry.multi_start_selection.result_schema_version,
        model_id=registry.model_id,
        mouse_id=train[0].mouse_id,
        runs=tuple(runs),
        winner_start_id=winner_id,
        selection_status=status,
        identifiability_ridge_detected=ridge,
        ridge_start_ids=ridge_ids,
        additional_starts_recommended=ridge,
    )
    if output is not None:
        save_multistart_result(summary, output / "multistart_summary.json")
    return summary


def _atomic_json_write(payload: dict, path: Path) -> None:
    """Write JSON through a sibling temporary file and atomically replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_fit_run_result(result: FitRunResult, path: str | Path) -> None:
    """Atomically save every requested Part VIII field for one run."""

    _atomic_json_write(result.as_dict(), Path(path).expanduser().resolve())


def save_multistart_result(result: MultiStartFitResult, path: str | Path) -> None:
    """Atomically save all run records and the training-based selection summary."""

    _atomic_json_write(result.as_dict(), Path(path).expanduser().resolve())
