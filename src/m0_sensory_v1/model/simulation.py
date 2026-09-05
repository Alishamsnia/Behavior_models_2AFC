"""Deterministic-seed M0 simulation and parameter-recovery utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import torch

from .fit import FitRunResult, fit_single_initialization
from .initialization import InitializationSpec
from .m0 import predict_m0_trials
from .parameters import encode_constrained_parameters
from .session import SessionTrials

if TYPE_CHECKING:
    from ..data.task_moments import TaskMoments
    from ..registry import ModelRegistry


@dataclass(frozen=True)
class ParameterRecoveryResult:
    """Generating/fitted parameters and predictive recovery diagnostics."""

    generating_parameters: tuple[float, float, float, float]
    recovered_parameters: tuple[float, float, float, float]
    absolute_parameter_errors: tuple[float, float, float, float]
    initial_training_mean_nll: float
    recovered_training_mean_nll: float
    training_nll_improvement: float
    probability_rmse_on_duration_grid: float
    fit_result: FitRunResult


def simulate_m0_sessions(
    durations_by_session: Iterable[torch.Tensor],
    constrained_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    random_seed: int,
    mouse_id: str = "synthetic_recovery_mouse",
    split_role: str = "train",
) -> tuple[SessionTrials, ...]:
    r"""Simulate completed Bernoulli choices from known M0 parameters.

    Purpose
    -------
    Generate reproducible synthetic sessions for Part X recovery tests without
    changing the model probability implementation.

    Mathematical definition
    -----------------------
    For each supplied duration, calculate

    .. math::

       p_i=(1-\epsilon_i)\Phi\!\left(
       \frac{s_i-T}{\sqrt{ws_i^2+s_0^2}}\right)+\epsilon_i/2,
       \qquad y_i\sim\operatorname{Bernoulli}(p_i).

    Parameter interpretation
    ------------------------
    ``(w,s0,ell_short,ell_long,kappa)`` is the known generating vector. It is encoded to raw form
    only so simulation calls the exact production M0 forward path.

    Input units and shapes
    ----------------------
    Each duration tensor is one-dimensional ``float64`` seconds. The constrained
    parameter tensor has shape ``(5,)``.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`SessionTrials` per input duration tensor with sampled
    binary choices and the requested non-test split role.

    Constraints
    -----------
    Durations must be finite and positive. Parameters must satisfy registered
    open constraints, and only train/validation simulation is permitted.

    Numerical stability
    -------------------
    Production parameter transforms and analytical probabilities are reused in
    ``float64``. Uniform random draws use an isolated seeded Torch generator.

    Causal restrictions
    -------------------
    Choices depend only on the current duration, fixed task moments, generating
    parameters, and independent random draws—never block or history.

    Important edge cases
    --------------------
    Empty session collections and empty duration tensors raise. Both physical
    stimulus categories must be represented to estimate both lapse rates.

    Related tests
    -------------
    ``test_simulation_is_seeded_and_uses_only_current_duration`` and
    ``test_parameter_recovery_improves_training_nll``.
    """

    duration_tuple = tuple(durations_by_session)
    if not duration_tuple:
        raise ValueError("At least one synthetic session is required")
    if split_role not in {"train", "validation"}:
        raise ValueError("Recovery simulation may use only train/validation")
    if constrained_parameters.shape != (5,):
        raise ValueError("constrained_parameters must have shape (5,)")
    raw = encode_constrained_parameters(constrained_parameters)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    sessions: list[SessionTrials] = []
    for index, duration in enumerate(duration_tuple, start=1):
        if duration.ndim != 1 or duration.dtype != torch.float64:
            raise TypeError("Synthetic durations must be one-dimensional float64")
        if duration.numel() == 0 or not bool(
            torch.all(torch.isfinite(duration) & (duration > 0.0))
        ):
            raise ValueError("Synthetic durations must be nonempty, finite, and positive")
        short = duration <= moments.max_short_s
        long = duration >= moments.min_long_s
        if not bool(torch.all(short | long)):
            raise ValueError(
                "Synthetic durations must belong to the registered short or long support"
            )
        category = long.to(torch.float64)
        placeholder = torch.zeros_like(duration)
        with torch.no_grad():
            prediction = predict_m0_trials(
                duration, category, placeholder, raw, moments
            )
            draws = torch.rand(duration.shape, dtype=torch.float64, generator=generator)
            choices = (draws < prediction.probability_long).to(torch.float64)
        sessions.append(
            SessionTrials(
                mouse_id=mouse_id,
                session_id=f"{split_role}_{index:03d}",
                duration_s=duration.clone(),
                category_long=category.clone(),
                choice_long=choices,
                split_role=split_role,  # type: ignore[arg-type]
            )
        )
    return tuple(sessions)


def run_parameter_recovery_case(
    generating_parameters: torch.Tensor,
    training_durations: Iterable[torch.Tensor],
    validation_durations: Iterable[torch.Tensor],
    registry: ModelRegistry,
    initialization: InitializationSpec,
    *,
    random_seed: int,
    maximum_epochs: int,
) -> ParameterRecoveryResult:
    r"""Simulate and refit one known M0 vector with Torch autograd.

    Purpose
    -------
    Execute one reproducible Part X recovery case and quantify both parameter
    and predictive recovery.

    Mathematical definition
    -----------------------
    After simulating :math:`y_i\sim\mathrm{Bernoulli}(p_i(\Theta^*))`, fit
    :math:`\widehat\Theta` by the registered mean NLL and report

    .. math::

       |\widehat\Theta-\Theta^*|,
       \qquad \operatorname{RMSE}_p=
       \sqrt{N^{-1}\sum_i[p_i(\widehat\Theta)-p_i(\Theta^*)]^2}.

    Parameter interpretation
    ------------------------
    Exact parameter recovery tests identifiability; probability RMSE separates
    an equivalent predictive ridge from a genuinely incorrect fit.

    Input units and shapes
    ----------------------
    The generating vector has shape ``(5,)`` with ``s0`` in seconds. Duration
    iterables contain one-dimensional ``float64`` tensors in seconds.

    Returned quantities and shapes
    ------------------------------
    Returns generating/recovered five-vectors, absolute errors, objective
    improvement, probability RMSE, weak-lapse warning, and the full fit record.

    Constraints
    -----------
    The initialization must be one of the frozen eight. ``maximum_epochs`` is a
    positive recovery-test override, so the nested fit is not a registered real
    fit and cannot become the real-data winner.

    Numerical stability
    -------------------
    Simulation, fitting, and comparison all use the production ``float64``
    forward equations and finite-value checks.

    Causal restrictions
    -------------------
    Synthetic validation does not affect gradients, scheduling, or checkpoints.
    Locked real test data are never accessed.

    Important edge cases
    --------------------
    A design containing only one physical stimulus category cannot identify the
    other category's lapse rate and is therefore inappropriate for recovery.

    Related tests
    -------------
    ``test_parameter_recovery_improves_training_nll`` and the saved recovery
    report generated by ``scripts/run_m0_recovery.py``.
    """

    if maximum_epochs <= 0:
        raise ValueError("maximum_epochs must be positive")
    train_duration_tuple = tuple(training_durations)
    validation_duration_tuple = tuple(validation_durations)
    training = simulate_m0_sessions(
        train_duration_tuple,
        generating_parameters,
        registry.task_moments,
        random_seed=random_seed,
        split_role="train",
    )
    validation = simulate_m0_sessions(
        validation_duration_tuple,
        generating_parameters,
        registry.task_moments,
        random_seed=random_seed + 1,
        split_role="validation",
    )
    fit = fit_single_initialization(
        training,
        validation,
        registry,
        initialization,
        max_epochs_override=maximum_epochs,
    )
    recovered = torch.tensor(fit.best_constrained_parameters, dtype=torch.float64)
    generating = generating_parameters.detach().to(dtype=torch.float64)
    comparison_duration = torch.unique(torch.cat(train_duration_tuple)).sort().values
    comparison_category = (
        comparison_duration >= registry.task_moments.min_long_s
    ).to(torch.float64)
    placeholder = torch.zeros_like(comparison_duration)
    with torch.no_grad():
        expected = predict_m0_trials(
            comparison_duration,
            comparison_category,
            placeholder,
            encode_constrained_parameters(generating),
            registry.task_moments,
        ).probability_long
        fitted = predict_m0_trials(
            comparison_duration,
            comparison_category,
            placeholder,
            torch.tensor(fit.best_raw_parameters, dtype=torch.float64),
            registry.task_moments,
        ).probability_long
        probability_rmse = float(torch.sqrt(torch.mean((fitted - expected) ** 2)))
    error = torch.abs(recovered - generating)
    return ParameterRecoveryResult(
        generating_parameters=tuple(float(value) for value in generating),  # type: ignore[arg-type]
        recovered_parameters=tuple(float(value) for value in recovered),  # type: ignore[arg-type]
        absolute_parameter_errors=tuple(float(value) for value in error),  # type: ignore[arg-type]
        initial_training_mean_nll=fit.initial_training_mean_nll,
        recovered_training_mean_nll=fit.recalculated_best_training_mean_nll,
        training_nll_improvement=(
            fit.initial_training_mean_nll
            - fit.recalculated_best_training_mean_nll
        ),
        probability_rmse_on_duration_grid=probability_rmse,
        fit_result=fit,
    )
