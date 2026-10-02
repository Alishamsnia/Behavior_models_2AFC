"""Full-batch Adam, L-BFGS, and hybrid fitting for M1_stimulus_kernel_v1.

Independent implementation: does not extend the shared M0/M1/M2 objective
dispatcher in ``m0_sensory_v1.model.fit`` (that dispatcher's ``_objective()``
is a closed if/elif on a fixed set of registered model IDs and would need
editing to add this model -- not permitted while M0-M4 are fitting real
data). Reuses only the genuinely shared ``LBFGSSettings`` dataclass. This
module is a near-verbatim structural copy of ``m1_known_switch_v1/fit.py``'s
multi-start optimizer loop (identical Adam/L-BFGS/hybrid mechanics,
convergence bookkeeping, per-start failure containment, and the
``ProcessPoolExecutor``-based ``parallel_workers``/``cpu_threads_per_worker``
dispatch in :func:`fit_m1_stimulus_kernel_all_initializations`, added here
for the same reason it exists in ``m1_known_switch_v1/fit.py``: a rolling-
window sliding experiment needs to run many independent multistart fits
concurrently); only the decode/objective functions and result types are
model-specific.

``fit_m1_stimulus_kernel_all_initializations`` runs sequentially in this
process (with a single ``torch.set_num_threads`` call) when
``parallel_workers<=1`` -- correctness-equivalent to parallel execution,
each start gets a fresh optimizer, scheduler, and raw parameter tensor,
just slower wall-clock -- and dispatches across a ``spawn``-context
``ProcessPoolExecutor`` (one process per start, each pinned to
``cpu_threads_per_worker`` Torch intra-op threads) when
``parallel_workers>1``. The plain single-mouse CLI (``fit_m1_stimulus_kernel.py``)
uses the sequential path (one multistart fit over the whole split is fast
enough not to need it); the sliding-window CLI
(``run_m1_stimulus_kernel_sliding_window.py``) uses the parallel path so
many rolling-window/single-session fits can share a bounded CPU budget.

Every start's entire fit-and-restore procedure (including the post-loop
restored-best-checkpoint gradient recomputation) is contained in its own
try/except: a single start's numerical failure is recorded as a failed
convergence check, not an exception that aborts the remaining starts.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import torch

from m0_sensory_v1.model.fit import LBFGSSettings
from m0_sensory_v1.model.m1_session import M1SessionTrials

from .initialization import M1StimulusKernelInitializationSpec
from .parameters import decode_m1_stimulus_kernel_raw_parameters
from .registry import M1StimulusKernelRegistry
from .session import evaluate_m1_stimulus_kernel_mouse_objective


M1StimulusKernelOptimizerStrategy = Literal["adam", "lbfgs", "hybrid"]

_WORKER_CONTEXT: tuple[object, ...] | None = None


def _initialize_worker(training, validation, registry, settings, cpu_threads: int) -> None:
    global _WORKER_CONTEXT
    torch.set_num_threads(cpu_threads)
    _WORKER_CONTEXT = (training, validation, registry, settings)


def _fit_worker(initialization):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("M1_stimulus_kernel worker was not initialized")
    training, validation, registry, settings = _WORKER_CONTEXT
    return fit_m1_stimulus_kernel_single_initialization(
        training, validation, registry, initialization, **settings
    )


def _atomic_run_json(run, directory: Path | None) -> None:
    if directory is None:
        return
    path = directory / f"start_{run.start_id:02d}" / "fit_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run.as_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class M1StimulusKernelEpochRecord:
    """One recorded optimizer step's fresh training/validation state."""

    step: int
    optimizer_phase: str
    fresh_training_mean_nll: float
    validation_mean_nll: float
    gradient_norm: float
    learning_rate: float
    constrained_parameters_after_step: tuple[float, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class M1StimulusKernelFitRunResult:
    """Complete record of one start's fit, restored best, and convergence."""

    result_schema_version: str
    model_id: str
    mouse_id: str
    registry_config_sha256: str
    start_id: int
    optimizer_strategy: M1StimulusKernelOptimizerStrategy
    initial_raw_parameters: tuple[float, ...]
    initial_constrained_parameters: tuple[float, ...]
    best_raw_parameters: tuple[float, ...]
    best_constrained_parameters: tuple[float, ...]
    history: tuple[M1StimulusKernelEpochRecord, ...]
    recalculated_best_training_mean_nll: float
    final_validation_mean_nll: float
    gradient_norm_at_restored_best: float
    optimizer_step_count: int
    termination_reason: str
    converged: bool
    checks: tuple[tuple[str, bool], ...]
    exception_message: str | None

    def as_dict(self) -> dict:
        result = asdict(self)
        result["history"] = [record.as_dict() for record in self.history]
        return result


@dataclass(frozen=True)
class M1StimulusKernelMultiStartFitResult:
    """All eight run records plus training-based winner selection."""

    result_schema_version: str
    model_id: str
    mouse_id: str
    runs: tuple[M1StimulusKernelFitRunResult, ...]
    winner_start_id: int | None
    selection_status: str
    identifiability_ridge_detected: bool
    ridge_start_ids: tuple[int, ...]
    additional_starts_recommended: bool
    convergence_escalation_stage: str
    convergence_escalation_detail: dict[str, Any] | None

    def as_dict(self) -> dict:
        return {
            "result_schema_version": self.result_schema_version,
            "model_id": self.model_id,
            "mouse_id": self.mouse_id,
            "runs": [run.as_dict() for run in self.runs],
            "winner_start_id": self.winner_start_id,
            "selection_status": self.selection_status,
            "identifiability_ridge_detected": self.identifiability_ridge_detected,
            "ridge_start_ids": list(self.ridge_start_ids),
            "additional_starts_recommended": self.additional_starts_recommended,
            "convergence_escalation_stage": self.convergence_escalation_stage,
            "convergence_escalation_detail": self.convergence_escalation_detail,
        }


def _gradient_norm(raw: torch.Tensor) -> float:
    if raw.grad is None:
        return float("nan")
    return float(torch.linalg.norm(raw.grad.detach()).cpu())


def _stable_training_history(
    history: list[M1StimulusKernelEpochRecord], *, window: int, tolerance: float
) -> bool:
    """Trailing-window peak-to-peak stability, matching M0/M1's own definition."""

    if len(history) < window:
        return False
    values = np.asarray(
        [record.fresh_training_mean_nll for record in history[-window:]], dtype=float
    )
    return bool(np.all(np.isfinite(values)) and np.ptp(values) <= tolerance)


def _detect_identifiability_ridge(
    eligible: list[tuple[int, float, tuple[float, ...]]],
    winner_start_id: int,
    *,
    near_equal_training_nll_tolerance: float,
    parameter_distance_scale: tuple[float, ...],
    identifiability_parameter_distance_threshold: float,
) -> tuple[bool, tuple[int, ...]]:
    """Flag converged starts within NLL tolerance of the winner but far away in
    scaled constrained-parameter space, exactly as `m1_dynamic_block_length_v1`
    does. Pure/primitive-typed so it can be unit-tested without a real fit.
    """
    winner_nll = next(nll for start_id, nll, _ in eligible if start_id == winner_start_id)
    winner_values = np.asarray(
        next(values for start_id, _, values in eligible if start_id == winner_start_id)
    )
    scale = np.asarray(parameter_distance_scale)
    distant: list[int] = []
    for start_id, nll, values in eligible:
        if start_id == winner_start_id:
            continue
        if abs(nll - winner_nll) > near_equal_training_nll_tolerance:
            continue
        distance = float(np.linalg.norm((np.asarray(values) - winner_values) / scale))
        if distance >= identifiability_parameter_distance_threshold:
            distant.append(start_id)
    ridge_ids = tuple(sorted({winner_start_id, *distant})) if distant else ()
    return bool(distant), ridge_ids


def _warm_start_initialization(run: "M1StimulusKernelFitRunResult") -> M1StimulusKernelInitializationSpec:
    """Build a start spec from a prior run's restored-best checkpoint.

    Used only for convergence-escalation continuation: re-optimizing from
    the best point a start already reached, rather than from its original
    (now irrelevant) registry-frozen initialization.
    """

    w, s0, ell_short, ell_long, kappa, a = run.best_constrained_parameters
    return M1StimulusKernelInitializationSpec(
        start_id=run.start_id, w=w, s0=s0, ell_short=ell_short, ell_long=ell_long,
        kappa=kappa, a=a,
        purpose=f"convergence_escalation_warm_start_from_start_{run.start_id}",
    )


def _only_gradient_norm_check_failed(run: "M1StimulusKernelFitRunResult") -> bool:
    """True iff ``small_gradient_norm`` is the sole reason this run is not
    converged -- every other check already passes and the run raised no
    exception.

    Narrower than "any exception-free run": a run whose training NLL is
    still unstable, or whose finite-parameter/gradient checks fail, has a
    different (and typically unrecoverable-by-more-steps-or-a-looser-
    tolerance) problem, so spending the escalation ladder's extra L-BFGS
    steps on it would not plausibly help and would only cost CPU time. This
    also makes stage 3's tolerance relaxation meaningful only for runs where
    gradient norm was genuinely the last remaining obstacle, rather than
    silently reinterpreting an unrelated failure as "close to converging."
    """

    if run.exception_message is not None:
        return False
    checks = dict(run.checks)
    if checks.get("small_gradient_norm") is not False:
        # Either the check is genuinely missing (a malformed/future-schema
        # record) or it already passed -- neither is "gradient norm is the
        # sole failure," so this must not be silently treated as one.
        return False
    return all(value for name, value in run.checks if name != "small_gradient_norm")


def _apply_stage2_result(
    runs: list["M1StimulusKernelFitRunResult"],
    original: "M1StimulusKernelFitRunResult",
    stage2_run: "M1StimulusKernelFitRunResult",
) -> list["M1StimulusKernelFitRunResult"]:
    """Replace ``original`` with ``stage2_run`` in ``runs`` only if the
    continuation actually converged.

    A stage-2 continuation always restarts with an empty step history (a
    fresh optimizer, fresh history buffer), so even a continuation whose
    gradient norm improves can regress an already-passing
    ``stable_fresh_training_nll`` check simply by not having accumulated
    ``stable_training_window_epochs`` records yet, or by a different early
    trajectory being less stable than the original run's. `original` is
    only ever selected for stage 2 because every one of its OTHER checks
    (including stability) already passed -- so an unconditional replacement
    risks discarding a legitimate near-miss (with a still-usable history for
    stage 3's tolerance relaxation) in favor of a strictly worse record.
    Only ever replacing on a converged continuation means this step can
    never make the eligible set worse than not running it at all.
    """

    if not stage2_run.converged:
        return runs
    return [stage2_run if r.start_id == original.start_id else r for r in runs]


def _relax_gradient_norm_tolerance(
    run: "M1StimulusKernelFitRunResult", tolerance: float
) -> "M1StimulusKernelFitRunResult":
    """Re-evaluate one run's ``small_gradient_norm`` check against a relaxed
    tolerance, leaving every other check and every numerical field alone --
    no re-optimization, just a threshold change applied to an already-computed
    restored gradient norm. A run that already failed with an exception is
    left untouched: a relaxed tolerance cannot rescue a numerical failure.
    """

    if run.exception_message is not None:
        return run
    relaxed_ok = bool(
        np.isfinite(run.gradient_norm_at_restored_best)
        and run.gradient_norm_at_restored_best <= tolerance
    )
    checks = tuple(
        (name, relaxed_ok if name == "small_gradient_norm" else value)
        for name, value in run.checks
    )
    converged = all(value for _, value in checks)
    termination_reason = run.termination_reason
    if converged and termination_reason == "maximum_epochs_without_convergence":
        termination_reason = "maximum_epochs_convergence_checks_passed_relaxed_gradient_tolerance"
    return replace(run, checks=checks, converged=converged, termination_reason=termination_reason)


def fit_m1_stimulus_kernel_single_initialization(
    training_sessions: Iterable[M1SessionTrials],
    validation_sessions: Iterable[M1SessionTrials],
    registry: M1StimulusKernelRegistry,
    initialization: M1StimulusKernelInitializationSpec,
    *,
    optimizer_strategy: M1StimulusKernelOptimizerStrategy = "adam",
    max_epochs_override: int | None = None,
    initial_learning_rate_override: float | None = None,
    adam_warmup_epochs: int | None = None,
    lbfgs_settings: LBFGSSettings | None = None,
    epoch_callback: Callable[[M1StimulusKernelEpochRecord], None] | None = None,
) -> M1StimulusKernelFitRunResult:
    r"""Fit one M1_stimulus_kernel start with full-batch Adam/L-BFGS/hybrid.

    Purpose
    -------
    Execute one registered optimizer strategy on the pooled six-parameter
    raw vector, restore the best fresh-training-NLL checkpoint, and return
    every diagnostic needed for multi-start winner selection -- without
    ever raising out of this function for a single start's numerical
    failure.

    Mathematical definition
    -----------------------
    .. math::

       J_{train}(\widetilde\Theta)=\frac{\sum_dJ_d(\widetilde\Theta)}{\sum_dN_d}.

    Parameter interpretation
    ------------------------
    Parameters are pooled within one mouse; the scalar prior is reset to
    0.5 at every session boundary inside the objective.

    Input units and shapes
    ----------------------
    ``training_sessions``/``validation_sessions`` are this mouse's frozen
    non-test sessions; durations are seconds.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`M1StimulusKernelFitRunResult` with the restored
    best raw/constrained parameters, complete step history, and
    convergence checks.

    Constraints
    -----------
    Validation never enters gradients, scheduler monitoring, or checkpoint
    selection.

    Numerical stability
    -------------------
    A start that raises during optimization or during the restored-best
    gradient recomputation is recorded as
    ``numerical_or_transform_failure`` with every check False, rather than
    propagating the exception.

    Causal restrictions
    -------------------
    This function only fits parameters; it never inspects test sessions.

    Important edge cases
    --------------------
    Hybrid mode runs Adam for ``adam_warmup_epochs``, restores its best
    checkpoint, then runs L-BFGS from there. Pure L-BFGS skips the Adam
    phase entirely.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_fit_single_initialization_all_strategies``.
    """

    train = tuple(training_sessions)
    validation = tuple(validation_sessions)
    if not train:
        raise ValueError("At least one M1_stimulus_kernel training session is required")
    registry_checksum = sha256(registry.config_path.read_bytes()).hexdigest()

    try:
        raw = initialization.raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        ).clone().requires_grad_(True)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        checks = tuple(
            (name, False) for name in (
                "finite_objective", "finite_parameters", "finite_gradients",
                "small_gradient_norm", "stable_fresh_training_nll",
                "no_parameter_transform_errors", "no_optimizer_runtime_errors",
            )
        )
        nan_vector = tuple(float("nan") for _ in registry.parameter_order)
        return M1StimulusKernelFitRunResult(
            result_schema_version=registry.multi_start_selection.result_schema_version,
            model_id=registry.model_id,
            mouse_id=train[0].mouse_id,
            registry_config_sha256=registry_checksum,
            start_id=initialization.start_id,
            optimizer_strategy=optimizer_strategy,
            initial_raw_parameters=nan_vector,
            initial_constrained_parameters=nan_vector,
            best_raw_parameters=nan_vector,
            best_constrained_parameters=nan_vector,
            history=(),
            recalculated_best_training_mean_nll=float("nan"),
            final_validation_mean_nll=float("nan"),
            gradient_norm_at_restored_best=float("nan"),
            optimizer_step_count=0,
            termination_reason="numerical_or_transform_failure",
            converged=False,
            checks=checks,
            exception_message=f"{type(error).__name__}: {error}",
        )
    initial_raw = tuple(raw.detach().cpu().tolist())
    try:
        initial_constrained = tuple(
            decode_m1_stimulus_kernel_raw_parameters(
                raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            ).as_tensor().detach().cpu().tolist()
        )
    except (FloatingPointError, RuntimeError, ValueError):
        initial_constrained = tuple(float("nan") for _ in registry.parameter_order)

    def objective(sessions, role: str):
        return evaluate_m1_stimulus_kernel_mouse_objective(
            sessions, raw, registry.task_moments,
            objective_weighting=registry.objective_weighting,
            required_split_role=role,
            lapse_max=registry.lapse_max,
            parameter_epsilon=registry.parameter_epsilon,
            log_probability_epsilon=registry.log_probability_epsilon,
            prior_probability_epsilon=registry.prior_probability_epsilon,
            quadratic_tolerance=registry.quadratic_tolerance,
            discriminant_tolerance=registry.discriminant_tolerance,
            initial_pi=registry.initial_pi,
        )

    history: list[M1StimulusKernelEpochRecord] = []
    best_raw = raw.detach().clone()
    best_training_nll = float("inf")
    exception_message: str | None = None
    termination_reason = "maximum_epochs_reached"
    optimizer_step_count = 0

    def run_adam(max_epochs: int, learning_rate: float, phase: str) -> None:
        nonlocal best_raw, best_training_nll, termination_reason, optimizer_step_count
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=registry.scheduler.factor,
            patience=registry.scheduler.patience_epochs,
            min_lr=registry.scheduler.minimum_learning_rate,
        )
        epochs_without_improvement = 0
        for step in range(1, max_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            train_result = objective(train, "train")
            train_result.loss.backward()
            optimizer.step()
            with torch.no_grad():
                fresh_train = objective(train, "train")
                fresh_validation = objective(validation, "validation") if validation else None
                constrained_now = tuple(fresh_train.parameters.as_tensor().detach().cpu().tolist())
            fresh_nll = float(fresh_train.loss.detach().cpu())
            if fresh_nll < best_training_nll - registry.convergence.meaningful_improvement_tolerance:
                best_training_nll = fresh_nll
                best_raw = raw.detach().clone()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            learning_rate_now = float(optimizer.param_groups[0]["lr"])
            record = M1StimulusKernelEpochRecord(
                step=step, optimizer_phase=phase,
                fresh_training_mean_nll=fresh_nll,
                validation_mean_nll=(
                    float(fresh_validation.loss.detach().cpu()) if fresh_validation is not None else float("nan")
                ),
                gradient_norm=_gradient_norm(raw),
                learning_rate=learning_rate_now,
                constrained_parameters_after_step=constrained_now,
            )
            history.append(record)
            optimizer_step_count += 1
            if epoch_callback is not None:
                epoch_callback(record)
            scheduler.step(fresh_nll)
            if (
                learning_rate_now <= registry.scheduler.minimum_learning_rate
                and epochs_without_improvement >= registry.convergence.no_improvement_patience_epochs
            ):
                termination_reason = "minimum_lr_plateau_reached"
                return

    def run_lbfgs(settings: LBFGSSettings, phase: str) -> None:
        nonlocal best_raw, best_training_nll, termination_reason, optimizer_step_count
        optimizer = torch.optim.LBFGS(
            [raw], lr=settings.learning_rate, max_iter=1,
            history_size=settings.history_size,
            max_eval=settings.maximum_evaluations_per_step,
            tolerance_grad=settings.tolerance_grad,
            tolerance_change=settings.tolerance_change,
            line_search_fn=settings.line_search_fn,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            result = objective(train, "train")
            result.loss.backward()
            return result.loss

        for step in range(1, settings.maximum_steps + 1):
            optimizer.step(closure)
            with torch.no_grad():
                fresh_train = objective(train, "train")
                fresh_validation = objective(validation, "validation") if validation else None
                constrained_now = tuple(fresh_train.parameters.as_tensor().detach().cpu().tolist())
            fresh_nll = float(fresh_train.loss.detach().cpu())
            if fresh_nll < best_training_nll - registry.convergence.meaningful_improvement_tolerance:
                best_training_nll = fresh_nll
                best_raw = raw.detach().clone()
            record = M1StimulusKernelEpochRecord(
                step=step, optimizer_phase=phase,
                fresh_training_mean_nll=fresh_nll,
                validation_mean_nll=(
                    float(fresh_validation.loss.detach().cpu()) if fresh_validation is not None else float("nan")
                ),
                gradient_norm=_gradient_norm(raw),
                learning_rate=settings.learning_rate,
                constrained_parameters_after_step=constrained_now,
            )
            history.append(record)
            optimizer_step_count += 1
            if epoch_callback is not None:
                epoch_callback(record)
            grad_norm = record.gradient_norm
            if np.isfinite(grad_norm) and grad_norm <= settings.tolerance_grad:
                termination_reason = "lbfgs_gradient_tolerance_reached"
                return

    try:
        if optimizer_strategy == "adam":
            epochs = max_epochs_override or registry.convergence.maximum_epochs
            rate = initial_learning_rate_override or registry.fitting.optional_adam_learning_rate
            run_adam(epochs, rate, "adam")
        elif optimizer_strategy == "lbfgs":
            settings = lbfgs_settings or LBFGSSettings(
                maximum_steps=registry.fitting.lbfgs_maximum_steps,
                learning_rate=registry.fitting.lbfgs_learning_rate,
                history_size=registry.fitting.lbfgs_history_size,
                maximum_evaluations_per_step=registry.fitting.lbfgs_maximum_evaluations_per_step,
                tolerance_grad=registry.fitting.lbfgs_tolerance_grad,
                tolerance_change=registry.fitting.lbfgs_tolerance_change,
                line_search_fn=registry.fitting.lbfgs_line_search_fn,
            )
            run_lbfgs(settings, "lbfgs")
        elif optimizer_strategy == "hybrid":
            warmup = adam_warmup_epochs or registry.fitting.optional_adam_warmup_steps
            rate = initial_learning_rate_override or registry.fitting.optional_adam_learning_rate
            run_adam(warmup, rate, "adam_warmup")
            with torch.no_grad():
                raw.copy_(best_raw)
            settings = lbfgs_settings or LBFGSSettings(
                maximum_steps=registry.fitting.lbfgs_maximum_steps,
                learning_rate=registry.fitting.lbfgs_learning_rate,
                history_size=registry.fitting.lbfgs_history_size,
                maximum_evaluations_per_step=registry.fitting.lbfgs_maximum_evaluations_per_step,
                tolerance_grad=registry.fitting.lbfgs_tolerance_grad,
                tolerance_change=registry.fitting.lbfgs_tolerance_change,
                line_search_fn=registry.fitting.lbfgs_line_search_fn,
            )
            run_lbfgs(settings, "lbfgs")
        else:
            raise ValueError(f"Unknown M1_stimulus_kernel optimizer strategy: {optimizer_strategy!r}")
    except (FloatingPointError, RuntimeError, ValueError) as error:
        exception_message = f"{type(error).__name__}: {error}"
        termination_reason = "numerical_or_transform_failure"

    try:
        with torch.no_grad():
            raw.copy_(best_raw)
        raw.grad = None
        restored_train = objective(train, "train")
        restored_train.loss.backward()
        restored_gradient_norm = _gradient_norm(raw)
        with torch.no_grad():
            restored_validation = objective(validation, "validation") if validation else None
        recalculated_train = float(restored_train.loss.detach().cpu())
        final_validation = (
            float(restored_validation.loss.detach().cpu()) if restored_validation is not None else float("nan")
        )
        restore_failed = False
    except (FloatingPointError, RuntimeError, ValueError) as error:
        if exception_message is None:
            exception_message = f"{type(error).__name__}: {error}"
        termination_reason = "numerical_or_transform_failure"
        recalculated_train = float("nan")
        final_validation = float("nan")
        restored_gradient_norm = float("nan")
        restore_failed = True

    try:
        best_constrained = tuple(
            decode_m1_stimulus_kernel_raw_parameters(
                best_raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            ).as_tensor().detach().cpu().tolist()
        )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        best_constrained = tuple(float("nan") for _ in registry.parameter_order)
        if exception_message is None:
            exception_message = f"{type(error).__name__}: {error}"
            termination_reason = "numerical_or_transform_failure"

    stable = _stable_training_history(
        history,
        window=registry.convergence.stable_training_window_epochs,
        tolerance=registry.convergence.meaningful_improvement_tolerance,
    )
    checks = (
        ("finite_objective", bool(np.isfinite(recalculated_train))),
        ("finite_parameters", not restore_failed and bool(torch.all(torch.isfinite(raw)).detach().cpu())),
        ("finite_gradients", not restore_failed and bool(raw.grad is not None and torch.all(torch.isfinite(raw.grad)))),
        ("small_gradient_norm", not restore_failed and restored_gradient_norm <= registry.convergence.gradient_norm_tolerance),
        ("stable_fresh_training_nll", stable),
        ("no_parameter_transform_errors", exception_message is None),
        ("no_optimizer_runtime_errors", exception_message is None),
    )
    converged = all(value for _, value in checks)
    if termination_reason == "maximum_epochs_reached":
        termination_reason = (
            "maximum_epochs_convergence_checks_passed" if converged else "maximum_epochs_without_convergence"
        )
    return M1StimulusKernelFitRunResult(
        result_schema_version=registry.multi_start_selection.result_schema_version,
        model_id=registry.model_id,
        mouse_id=train[0].mouse_id,
        registry_config_sha256=registry_checksum,
        start_id=initialization.start_id,
        optimizer_strategy=optimizer_strategy,
        initial_raw_parameters=initial_raw,
        initial_constrained_parameters=initial_constrained,
        best_raw_parameters=tuple(best_raw.detach().cpu().tolist()),
        best_constrained_parameters=best_constrained,
        history=tuple(history),
        recalculated_best_training_mean_nll=recalculated_train,
        final_validation_mean_nll=final_validation,
        gradient_norm_at_restored_best=restored_gradient_norm,
        optimizer_step_count=optimizer_step_count,
        termination_reason=termination_reason,
        converged=converged,
        checks=checks,
        exception_message=exception_message,
    )


def fit_m1_stimulus_kernel_all_initializations(
    training_sessions: Iterable[M1SessionTrials],
    validation_sessions: Iterable[M1SessionTrials],
    registry: M1StimulusKernelRegistry,
    *,
    optimizer_strategy: M1StimulusKernelOptimizerStrategy = "adam",
    max_epochs_override: int | None = None,
    initial_learning_rate_override: float | None = None,
    adam_warmup_epochs: int | None = None,
    lbfgs_settings: LBFGSSettings | None = None,
    enable_convergence_escalation: bool = True,
    escalated_lbfgs_additional_steps: int = 250,
    gradient_norm_tolerance_relaxation_factor: float = 2.0,
    initial_callback: Callable[[int], None] | None = None,
    epoch_callback: Callable[[int, M1StimulusKernelEpochRecord], None] | None = None,
    completion_callback: Callable[[M1StimulusKernelFitRunResult], None] | None = None,
    parallel_workers: int = 1,
    cpu_threads_per_worker: int | None = None,
    output_directory: str | Path | None = None,
) -> M1StimulusKernelMultiStartFitResult:
    r"""Fit all eight registered M1_stimulus_kernel starts, then select a winner.

    Purpose
    -------
    Run every frozen start (sequentially in this process when
    ``parallel_workers<=1``, or across a ``ProcessPoolExecutor`` of up to
    ``parallel_workers`` processes otherwise -- see module docstring), then
    select a training-only winner among converged runs.

    Mathematical definition
    -----------------------
    Winner is the converged run with lowest
    :math:`J_{train}(\widetilde\Theta^\star)`; ties are not specially
    resolved (first lowest wins).

    Parameter interpretation
    ------------------------
    Every start is independent; no fitted winner from a prior start seeds
    another. ``parallel_workers``/``cpu_threads_per_worker`` affect only
    wall-clock time and event-callback timing (parallel workers replay
    their recorded history to ``epoch_callback`` only after completing,
    since a live per-step callback cannot cross a process boundary), never
    the fitted result.

    Input units and shapes
    ----------------------
    Identical to :func:`fit_m1_stimulus_kernel_single_initialization`.

    Returned quantities and shapes
    ------------------------------
    Returns all eight run records plus ``winner_start_id`` (``None`` if no
    run converged) and a ``selection_status`` string.

    Constraints
    -----------
    Validation never enters winner selection. ``parallel_workers`` and
    ``cpu_threads_per_worker`` (when given) must be positive.

    Numerical stability
    -------------------
    One start's failure (caught inside
    :func:`fit_m1_stimulus_kernel_single_initialization`) does not stop the
    remaining starts.

    Causal restrictions
    -------------------
    Test sessions must never be passed to this function.

    Important edge cases
    --------------------
    If zero runs converge, ``winner_start_id`` is ``None`` and
    ``selection_status`` records that no eligible run exists.
    ``output_directory``, when given, receives one atomically-written
    ``start_<id>/fit_result.json`` per completed run regardless of strategy,
    plus one ``multistart_summary.json`` (this result's own ``as_dict()``)
    once every start has completed. A converged start within
    ``near_equal_training_nll_tolerance`` of the winner's training NLL but
    at least ``identifiability_parameter_distance_threshold`` away in
    scaled constrained-parameter space (see ``registry.multi_start_selection``)
    flags ``identifiability_ridge_detected``/``ridge_start_ids``, mirroring
    ``m1_dynamic_block_length_v1``'s ridge check.

    Convergence escalation
    ----------------------
    When ``enable_convergence_escalation`` is true (the default) and
    ``optimizer_strategy`` is ``"lbfgs"`` or ``"hybrid"``, zero converged
    starts at the baseline step budget triggers a two-stage recovery ladder
    before ``winner_start_id`` is given up as ``None``:

    1. **Extend steps (targeted).** Only starts whose sole failing check is
       ``small_gradient_norm`` (see :func:`_only_gradient_norm_check_failed`)
       -- i.e. every other check already passes and the run raised no
       exception -- are warm-started from their own restored-best checkpoint
       and re-optimized (pure L-BFGS) for up to
       ``escalated_lbfgs_additional_steps`` further steps. A run failing on
       other grounds (e.g. an unstable training trajectory or a numerical
       exception) is left untouched: more steps would not plausibly fix a
       different problem, and skipping it saves the CPU time that would
       otherwise be spent re-optimizing a run with no realistic chance of
       recovering. A targeted run's entry in ``runs`` is only ever replaced
       by its stage-2 continuation if that continuation itself converges
       (see :func:`_apply_stage2_result`) -- the continuation restarts with
       an empty step history, so an unconverged continuation could
       otherwise regress an already-passing ``stable_fresh_training_nll``
       check on the original run; discarding it instead means stage 2 can
       never make the eligible set worse than skipping it entirely.
    2. **Relax tolerance.** If stage 1 still leaves zero converged starts,
       every run's ``small_gradient_norm`` check (and only that check) is
       re-evaluated against ``registry.convergence.gradient_norm_tolerance
       * gradient_norm_tolerance_relaxation_factor`` -- a threshold change
       on an already-computed gradient norm, not further optimization. This
       is applied to every run (not just the stage-1 targeted subset)
       because it is a cheap, self-limiting re-check: a run failing on any
       other ground still has ``converged=False`` after the relaxation,
       since every other check is left untouched.

    Either stage recovering at least one converged start sets
    ``convergence_escalation_stage`` to ``"stage2_extended_steps"`` or
    ``"stage3_relaxed_tolerance"`` (``"stage1_baseline"`` when no escalation
    was needed, ``"unrecovered"`` when the ladder was attempted and still
    failed) and populates ``convergence_escalation_detail`` with the
    attempted/recovered start IDs and the old/new numeric thresholds, so a
    winner selected this way is auditable rather than silently
    indistinguishable from a clean stage-1 pass. Every stage transition is
    also printed to stdout as it happens.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_fit_all_initializations_selects_training_winner``.
    """

    train = tuple(training_sessions)
    validation = tuple(validation_sessions)
    if parallel_workers <= 0 or (cpu_threads_per_worker is not None and cpu_threads_per_worker <= 0):
        raise ValueError("parallel_workers and cpu_threads_per_worker must be positive")
    if escalated_lbfgs_additional_steps <= 0:
        raise ValueError("escalated_lbfgs_additional_steps must be positive")
    if not np.isfinite(gradient_norm_tolerance_relaxation_factor) or (
        gradient_norm_tolerance_relaxation_factor < 1.0
    ):
        raise ValueError("gradient_norm_tolerance_relaxation_factor must be a finite number >= 1.0")
    settings = dict(
        optimizer_strategy=optimizer_strategy,
        max_epochs_override=max_epochs_override,
        initial_learning_rate_override=initial_learning_rate_override,
        adam_warmup_epochs=adam_warmup_epochs,
        lbfgs_settings=lbfgs_settings,
    )
    output = None if output_directory is None else Path(output_directory)
    runs: list[M1StimulusKernelFitRunResult] = []
    workers = min(parallel_workers, len(registry.initializations), multiprocessing.cpu_count())
    if workers == 1:
        if cpu_threads_per_worker is not None:
            torch.set_num_threads(cpu_threads_per_worker)
        for initialization in registry.initializations:
            if initial_callback is not None:
                initial_callback(initialization.start_id)
            run = fit_m1_stimulus_kernel_single_initialization(
                train, validation, registry, initialization,
                epoch_callback=(
                    (lambda record, sid=initialization.start_id: epoch_callback(sid, record))
                    if epoch_callback is not None else None
                ), **settings,
            )
            runs.append(run)
            _atomic_run_json(run, output)
            if completion_callback is not None:
                completion_callback(run)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_initialize_worker,
            initargs=(train, validation, registry, settings, cpu_threads_per_worker or 1),
        ) as executor:
            futures = {executor.submit(_fit_worker, init): init for init in registry.initializations}
            for future in as_completed(futures):
                run = future.result()
                runs.append(run)
                _atomic_run_json(run, output)
                if epoch_callback is not None:
                    for record in run.history:
                        epoch_callback(run.start_id, record)
                if completion_callback is not None:
                    completion_callback(run)
        runs.sort(key=lambda run: run.start_id)

    mouse_id = train[0].mouse_id if train else ""
    eligible = [run for run in runs if run.converged]
    convergence_escalation_stage = "stage1_baseline"
    convergence_escalation_detail: dict[str, Any] | None = None

    if (
        not eligible
        and enable_convergence_escalation
        and optimizer_strategy in ("lbfgs", "hybrid")
    ):
        targeted_runs = [run for run in runs if _only_gradient_norm_check_failed(run)]
        print(
            f"M1_stimulus_kernel convergence escalation stage=2 mouse={mouse_id} "
            f"reason=no_converged_run_at_baseline_steps "
            f"targeted_start_ids={[run.start_id for run in targeted_runs]} "
            f"additional_steps={escalated_lbfgs_additional_steps}",
            flush=True,
        )
        recovered_stage2: list[int] = []
        if targeted_runs:
            base_settings = lbfgs_settings or LBFGSSettings(
                maximum_steps=registry.fitting.lbfgs_maximum_steps,
                learning_rate=registry.fitting.lbfgs_learning_rate,
                history_size=registry.fitting.lbfgs_history_size,
                maximum_evaluations_per_step=registry.fitting.lbfgs_maximum_evaluations_per_step,
                tolerance_grad=registry.fitting.lbfgs_tolerance_grad,
                tolerance_change=registry.fitting.lbfgs_tolerance_change,
                line_search_fn=registry.fitting.lbfgs_line_search_fn,
            )
            stage2_settings = replace(base_settings, maximum_steps=escalated_lbfgs_additional_steps)
            stage2_output = None if output is None else output / "escalation_stage2"
            for run in targeted_runs:
                warm_start = _warm_start_initialization(run)
                if initial_callback is not None:
                    initial_callback(warm_start.start_id)
                stage2_run = fit_m1_stimulus_kernel_single_initialization(
                    train, validation, registry, warm_start,
                    optimizer_strategy="lbfgs", lbfgs_settings=stage2_settings,
                    epoch_callback=(
                        (lambda record, sid=run.start_id: epoch_callback(sid, record))
                        if epoch_callback is not None else None
                    ),
                )
                _atomic_run_json(stage2_run, stage2_output)
                if completion_callback is not None:
                    completion_callback(stage2_run)
                runs = _apply_stage2_result(runs, run, stage2_run)
                if stage2_run.converged:
                    recovered_stage2.append(run.start_id)
            eligible = [run for run in runs if run.converged]
        convergence_escalation_detail = {
            "stage2_additional_max_steps": escalated_lbfgs_additional_steps,
            "stage2_targeting_rule": "sole_failing_check_is_small_gradient_norm",
            "stage2_attempted_start_ids": [run.start_id for run in targeted_runs],
            "stage2_recovered_start_ids": recovered_stage2,
        }
        print(
            f"M1_stimulus_kernel convergence escalation stage=2 mouse={mouse_id} "
            f"result={'recovered' if eligible else 'still_unconverged'} "
            f"recovered_start_ids={recovered_stage2}",
            flush=True,
        )
        if eligible:
            convergence_escalation_stage = "stage2_extended_steps"
        else:
            tolerance = registry.convergence.gradient_norm_tolerance
            relaxed_tolerance = tolerance * gradient_norm_tolerance_relaxation_factor
            print(
                f"M1_stimulus_kernel convergence escalation stage=3 mouse={mouse_id} "
                f"reason=no_converged_run_after_extended_steps "
                f"gradient_norm_tolerance={tolerance:.3e}->{relaxed_tolerance:.3e}",
                flush=True,
            )
            relaxed_runs = [_relax_gradient_norm_tolerance(run, relaxed_tolerance) for run in runs]
            relaxed_eligible = [run for run in relaxed_runs if run.converged]
            print(
                f"M1_stimulus_kernel convergence escalation stage=3 mouse={mouse_id} "
                f"result={'recovered' if relaxed_eligible else 'unrecovered'} "
                f"recovered_start_ids={[run.start_id for run in relaxed_eligible]}",
                flush=True,
            )
            convergence_escalation_detail.update(
                stage3_gradient_norm_tolerance_original=tolerance,
                stage3_gradient_norm_tolerance_relaxed=relaxed_tolerance,
                stage3_recovered_start_ids=[run.start_id for run in relaxed_eligible],
            )
            if relaxed_eligible:
                runs = relaxed_runs
                eligible = relaxed_eligible
                convergence_escalation_stage = "stage3_relaxed_tolerance"
            else:
                convergence_escalation_stage = "unrecovered"

    if not eligible:
        result = M1StimulusKernelMultiStartFitResult(
            result_schema_version=registry.multi_start_selection.result_schema_version,
            model_id=registry.model_id,
            mouse_id=mouse_id,
            runs=tuple(runs),
            winner_start_id=None,
            selection_status="no_converged_run",
            identifiability_ridge_detected=False,
            ridge_start_ids=(),
            additional_starts_recommended=True,
            convergence_escalation_stage=convergence_escalation_stage,
            convergence_escalation_detail=convergence_escalation_detail,
        )
    else:
        winner = min(eligible, key=lambda run: run.recalculated_best_training_mean_nll)
        ridge, ridge_ids = _detect_identifiability_ridge(
            [(run.start_id, run.recalculated_best_training_mean_nll, run.best_constrained_parameters) for run in eligible],
            winner.start_id,
            near_equal_training_nll_tolerance=registry.multi_start_selection.near_equal_training_nll_tolerance,
            parameter_distance_scale=registry.multi_start_selection.parameter_distance_scale,
            identifiability_parameter_distance_threshold=registry.multi_start_selection.identifiability_parameter_distance_threshold,
        )
        result = M1StimulusKernelMultiStartFitResult(
            result_schema_version=registry.multi_start_selection.result_schema_version,
            model_id=registry.model_id,
            mouse_id=mouse_id,
            runs=tuple(runs),
            winner_start_id=winner.start_id,
            selection_status="training_only_winner_selected",
            identifiability_ridge_detected=ridge,
            ridge_start_ids=ridge_ids,
            additional_starts_recommended=ridge,
            convergence_escalation_stage=convergence_escalation_stage,
            convergence_escalation_detail=convergence_escalation_detail,
        )
    if output is not None:
        destination = output / "multistart_summary.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    return result
