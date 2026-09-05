"""Causal true-rate constant-hazard HMM evaluation for M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch

from ..data.task_moments import TaskMoments
from .bayesian import lapse_mixture
from .m1_bayesian import m1_bayesian_choice_probability
from .m1_parameters import M1DecodedParameters, decode_m1_raw_parameters


M1SplitRole = Literal["train", "validation", "test"]
M1ObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]


@dataclass(frozen=True)
class M1SessionTrials:
    """One full task timeline with causal post-choice category feedback."""

    mouse_id: str
    session_id: str
    duration_s: torch.Tensor
    choice_long: torch.Tensor
    category_long: torch.Tensor
    feedback_valid: torch.Tensor
    split_role: M1SplitRole

    def __post_init__(self) -> None:
        """Validate timeline shape, registered precision, and binary fields."""

        if not self.mouse_id or not self.session_id:
            raise ValueError("mouse_id and session_id must be nonempty")
        if self.split_role not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split role: {self.split_role!r}")
        floating = (self.duration_s, self.choice_long, self.category_long)
        if any(value.ndim != 1 for value in floating) or self.feedback_valid.ndim != 1:
            raise ValueError("M1 session tensors must be one-dimensional")
        shapes = {tuple(value.shape) for value in (*floating, self.feedback_valid)}
        if len(shapes) != 1:
            raise ValueError("M1 session tensors must share one timeline shape")
        if self.duration_s.numel() == 0:
            raise ValueError("M1 sessions must contain at least one task trial")
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("M1 duration, choice, and category tensors must be floating")
        if any(value.dtype != torch.float64 for value in floating):
            raise TypeError("Registered M1 scientific tensors must use torch.float64")
        if self.feedback_valid.dtype != torch.bool:
            raise TypeError("M1 feedback_valid must use torch.bool")
        devices = {value.device for value in (*floating, self.feedback_valid)}
        if len(devices) != 1:
            raise ValueError("M1 session tensors must share a device")
        feedback_categories = self.category_long[self.feedback_valid]
        if feedback_categories.numel() and not bool(
            torch.all(
                torch.isfinite(feedback_categories)
                & ((feedback_categories == 0.0) | (feedback_categories == 1.0))
            )
        ):
            raise ValueError("Every feedback-valid M1 trial requires a binary category")


@dataclass(frozen=True)
class M1Predictions:
    """Full-timeline M1 predictions and causal hidden-state diagnostics."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    prior_long: torch.Tensor
    positive_threshold_s: torch.Tensor
    decision_lower_boundary_s: torch.Tensor
    decision_upper_boundary_s: torch.Tensor
    decision_regime: tuple[str, ...]
    sensory_sd_s: torch.Tensor
    emission_normalizer: torch.Tensor
    pre_choice_block_probability: torch.Tensor
    post_feedback_block_probability: torch.Tensor
    valid_choice_mask: torch.Tensor
    feedback_update_mask: torch.Tensor
    parameters: M1DecodedParameters


@dataclass(frozen=True)
class M1SessionEvaluation:
    """Differentiable M1 session choice NLL and state trajectories."""

    mouse_id: str
    session_id: str
    split_role: M1SplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M1Predictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M1MouseObjectiveResult:
    """One pooled M1 objective built from independently reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: M1ObjectiveWeighting
    session_evaluations: tuple[M1SessionEvaluation, ...]
    parameters: M1DecodedParameters


def m1_session_trials_from_frame(
    frame: pd.DataFrame,
    *,
    split_role: M1SplitRole,
    device: torch.device | str | None = None,
) -> M1SessionTrials:
    r"""Convert a canonical session table to M1's full causal timeline.

    Purpose
    -------
    Retain every task trial because omissions and invalid choices can still
    advance the hidden block process and may, when explicitly marked, deliver
    category feedback.

    Mathematical definition
    -----------------------
    The adapter preserves

    .. math::

       \{(s_t,Y_t,C_t,F_t)\}_{t=1}^{T_d},

    where ``F_t`` declares post-choice category availability.

    Parameter interpretation
    ------------------------
    Category is an authoritative post-choice observation; ``feedback_valid``
    controls whether it updates M1's hidden state.

    Input units and shapes
    ----------------------
    Durations are seconds and every output tensor is one-dimensional with one
    element per original task trial.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`M1SessionTrials` using ``torch.float64`` and a boolean
    feedback mask.

    Constraints
    -----------
    The frame must contain exactly one mouse/session and all feedback-valid
    categories must be binary.

    Numerical stability
    -------------------
    Pandas missing values become IEEE NaN without being coerced to a category.

    Causal restrictions
    -------------------
    True block and true run length are deliberately not transferred.

    Important edge cases
    --------------------
    A feedback-valid omission is retained and can update future belief without
    adding a current Bernoulli likelihood term.

    Related tests
    -------------
    ``test_m1_frame_adapter_retains_feedback_fields_and_omissions``.
    """

    required = {
        "mouse_id",
        "session_id",
        "duration_s",
        "choice_long",
        "category_long",
        "feedback_valid",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Canonical M1 frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot construct M1SessionTrials from an empty frame")
    identifiers = frame[["mouse_id", "session_id"]].drop_duplicates()
    if len(identifiers) != 1:
        raise ValueError("M1 frame must contain exactly one mouse/session pair")

    def numeric(column: str) -> np.ndarray:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        ).copy()

    feedback_numeric = numeric("feedback_valid")
    if np.any(~np.isin(feedback_numeric[np.isfinite(feedback_numeric)], [0.0, 1.0])):
        raise ValueError("feedback_valid must be binary or missing")
    feedback = np.isfinite(feedback_numeric) & (feedback_numeric == 1.0)
    identifier = identifiers.iloc[0]
    return M1SessionTrials(
        mouse_id=str(identifier["mouse_id"]),
        session_id=str(identifier["session_id"]),
        duration_s=torch.as_tensor(numeric("duration_s"), dtype=torch.float64, device=device),
        choice_long=torch.as_tensor(numeric("choice_long"), dtype=torch.float64, device=device),
        category_long=torch.as_tensor(numeric("category_long"), dtype=torch.float64, device=device),
        feedback_valid=torch.as_tensor(feedback, dtype=torch.bool, device=device),
        split_role=split_role,
    )


def constant_hazard_transition_matrix(
    hazard: torch.Tensor,
    *,
    first_long_probability: float = 0.5,
) -> torch.Tensor:
    r"""Construct M1's ``(N,S,L)`` row-stochastic transition kernel.

    Purpose
    -------
    Encode a neutral exit hazard tied to half the biased-block hazard and
    strict alternation of biased latent blocks without consulting the
    experimenter's realized block sequence.

    Mathematical definition
    -----------------------
    For biased-block hazard :math:`h_B`, derived neutral hazard
    :math:`h_N=h_B/2`, and fixed design probability :math:`\omega`, return

    .. math::

       T(h_B)=\begin{pmatrix}
       1-h_N & h_N(1-\omega) & h_N\omega\\
       0 & 1-h_B & h_B\\
       0 & h_B & 1-h_B
       \end{pmatrix}.

    Parameter interpretation
    ------------------------
    ``hazard`` is :math:`h_B`; :math:`h_N=h_B/2` is a fixed task-design tie,
    not another fitted parameter. ``first_long_probability`` is the design
    probability that neutral exits to L; it is not the realized first biased
    block of a session.

    Input units and shapes
    ----------------------
    ``hazard`` is the scalar biased-block probability and ``omega`` is
    dimensionless.

    Returned quantities and shapes
    ------------------------------
    Returns a ``(3,3)`` tensor in state order ``(N,S,L)``.

    Constraints
    -----------
    ``0 <= h <= 1`` and ``0 <= omega <= 1``.

    Numerical stability
    -------------------
    The matrix is assembled from the input tensor, preserving dtype, device,
    and autograd.

    Causal restrictions
    -------------------
    No true block, true run length, category, or session identity is accepted.

    Important edge cases
    --------------------
    Direct ``h_B=0`` and ``h_B=1`` inputs implement the exact endpoint
    kernels; at ``h_B=1``, the tied neutral hazard is ``h_N=0.5``.

    Related tests
    -------------
    ``test_m1_transition_kernel_endpoints_and_routing``.
    """

    if not isinstance(hazard, torch.Tensor) or not hazard.is_floating_point():
        raise TypeError("hazard must be a floating torch.Tensor")
    if hazard.ndim != 0:
        raise ValueError("M1 currently requires a scalar pooled hazard")
    if not 0.0 <= first_long_probability <= 1.0:
        raise ValueError("first_long_probability must lie in [0,1]")
    if not bool(torch.isfinite(hazard) & (hazard >= 0.0) & (hazard <= 1.0)):
        raise ValueError("hazard must be a finite probability")
    one = torch.ones_like(hazard)
    zero = torch.zeros_like(hazard)
    omega = hazard.new_tensor(first_long_probability)
    neutral_hazard = hazard / 2.0
    return torch.stack(
        (
            torch.stack(
                (
                    one - neutral_hazard,
                    neutral_hazard * (one - omega),
                    neutral_hazard * omega,
                )
            ),
            torch.stack((zero, one - hazard, hazard)),
            torch.stack((zero, hazard, one - hazard)),
        )
    )


def evaluate_m1_session(
    session: M1SessionTrials,
    parameters: M1DecodedParameters,
    moments: TaskMoments,
    *,
    first_long_probability: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M1SessionEvaluation:
    r"""Evaluate one M1 session in strict pre-choice/post-feedback order.

    Purpose
    -------
    Produce the causal choice likelihood while rebuilding all recursive HMM
    state locally on every call.

    Mathematical definition
    -----------------------
    With fixed prototypes :math:`\theta=(0.5,0.2,0.8)`, M1 computes

    .. math::

       \pi_t=\xi_t^-\theta,\quad
       \xi_t^+(b)\propto\xi_t^-(b)\theta_b^{C_t}(1-\theta_b)^{1-C_t},
       \quad \xi_{t+1}^-=\xi_t^+T(h_B,h_N),\qquad h_N=h_B/2.

    Parameter interpretation
    ------------------------
    ``hazard`` is the biased-state hazard; the neutral-state hazard is tied to
    half of it. Category feedback updates belief only after the current choice
    has been predicted.

    Input units and shapes
    ----------------------
    Session tensors have length ``T``; durations and ``s0`` are seconds.

    Returned quantities and shapes
    ------------------------------
    Returns scalar choice NLL, full-timeline probabilities, and ``(T,3)`` pre-
    and post-feedback state trajectories.

    Constraints
    -----------
    Hidden-state probabilities must remain finite, nonnegative, and normalized.

    Numerical stability
    -------------------
    Every emission normalizer is checked. Only probabilities entering logs are
    clamped; saved predictions remain unclamped.

    Causal restrictions
    -------------------
    Current category and feedback cannot affect current ``prior_long`` or
    ``probability_long``. True block and true run length are absent.

    Important edge cases
    --------------------
    Invalid choices add no likelihood but still receive the hazard transition.
    Feedback-valid invalid-choice trials can update belief. No transition is
    attempted after the final task trial.

    Related tests
    -------------
    ``test_m1_no_current_trial_leakage_and_feedback_affects_next_trial`` and
    ``test_m1_omission_advances_transition_without_likelihood``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    transition = constant_hazard_transition_matrix(
        parameters.hazard,
        first_long_probability=first_long_probability,
    )
    theta = parameters.w.new_tensor([0.5, 0.2, 0.8])
    xi_pre = parameters.w.new_tensor([1.0, 0.0, 0.0])

    priors: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    non_lapse: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    lower_boundaries: list[torch.Tensor] = []
    upper_boundaries: list[torch.Tensor] = []
    sensory_sd: list[torch.Tensor] = []
    decision_regimes: list[str] = []
    emission_normalizers: list[torch.Tensor] = []
    pre_states: list[torch.Tensor] = []
    post_states: list[torch.Tensor] = []
    valid_nll: list[torch.Tensor] = []

    valid_mask = (
        torch.isfinite(session.duration_s)
        & (session.duration_s > 0.0)
        & torch.isfinite(session.choice_long)
        & ((session.choice_long == 0.0) | (session.choice_long == 1.0))
    )
    for index in range(session.duration_s.numel()):
        if not bool(
            torch.all(torch.isfinite(xi_pre) & (xi_pre >= 0.0)).detach().cpu()
        ) or not bool(torch.isclose(xi_pre.sum(), xi_pre.new_tensor(1.0), atol=1e-12, rtol=0.0)):
            raise FloatingPointError("M1 pre-choice block belief is invalid")
        pre_states.append(xi_pre)
        prior = torch.dot(xi_pre, theta)
        priors.append(prior)

        if bool(valid_mask[index].detach().cpu()):
            result = m1_bayesian_choice_probability(
                session.duration_s[index],
                parameters.common(),
                moments,
                prior,
                prior_probability_epsilon=prior_probability_epsilon,
                quadratic_tolerance=quadratic_tolerance,
                discriminant_tolerance=discriminant_tolerance,
            )
            probability = lapse_mixture(
                result.probability_long,
                session.category_long[index],
                parameters.common(),
            )
            safe = torch.clamp(
                probability,
                min=log_probability_epsilon,
                max=1.0 - log_probability_epsilon,
            )
            observed = session.choice_long[index]
            valid_nll.append(
                -(observed * torch.log(safe) + (1.0 - observed) * torch.log1p(-safe))
            )
            probabilities.append(probability)
            non_lapse.append(result.probability_long)
            lower_boundaries.append(result.decision_region.lower_boundary_s)
            upper_boundaries.append(result.decision_region.upper_boundary_s)
            thresholds.append(result.decision_region.upper_boundary_s)
            decision_regimes.append(result.decision_region.regime)
            sensory_sd.append(result.sensory_sd_s)
        else:
            missing = session.duration_s.new_tensor(torch.nan)
            probabilities.append(missing)
            non_lapse.append(missing.clone())
            thresholds.append(missing.clone())
            lower_boundaries.append(missing.clone())
            upper_boundaries.append(missing.clone())
            sensory_sd.append(missing.clone())
            decision_regimes.append("not_evaluated_invalid_choice")

        if bool(session.feedback_valid[index].detach().cpu()):
            category = session.category_long[index]
            emission = theta.pow(category) * (1.0 - theta).pow(1.0 - category)
            unnormalized = xi_pre * emission
            normalizer = unnormalized.sum()
            if not bool(torch.isfinite(normalizer) & (normalizer > 0.0)):
                raise FloatingPointError("M1 feedback update has invalid normalizer")
            emission_normalizers.append(normalizer)
            xi_post = unnormalized / normalizer
        else:
            emission_normalizers.append(session.duration_s.new_tensor(torch.nan))
            xi_post = xi_pre
        if not bool(
            torch.all(torch.isfinite(xi_post) & (xi_post >= 0.0)).detach().cpu()
        ) or not bool(torch.isclose(xi_post.sum(), xi_post.new_tensor(1.0), atol=1e-12, rtol=0.0)):
            raise FloatingPointError("M1 post-feedback block belief is invalid")
        post_states.append(xi_post)
        if index + 1 < session.duration_s.numel():
            xi_pre = xi_post @ transition
            xi_pre = xi_pre / xi_pre.sum()

    probability_tensor = torch.stack(probabilities)
    non_lapse_tensor = torch.stack(non_lapse)
    threshold_tensor = torch.stack(thresholds)
    lower_boundary_tensor = torch.stack(lower_boundaries)
    upper_boundary_tensor = torch.stack(upper_boundaries)
    sensory_tensor = torch.stack(sensory_sd)
    prior_tensor = torch.stack(priors)
    valid_count = int(valid_mask.sum().detach().cpu())
    if valid_nll:
        valid_nll_tensor = torch.stack(valid_nll)
        total_nll = valid_nll_tensor.sum()
        per_trial_nll = torch.full_like(session.duration_s, torch.nan).masked_scatter(
            valid_mask, valid_nll_tensor
        )
    else:
        total_nll = parameters.as_tensor().sum() * 0.0
        per_trial_nll = torch.full_like(session.duration_s, torch.nan)
    return M1SessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=M1Predictions(
            probability_long=probability_tensor,
            non_lapse_probability_long=non_lapse_tensor,
            prior_long=prior_tensor,
            positive_threshold_s=threshold_tensor,
            decision_lower_boundary_s=lower_boundary_tensor,
            decision_upper_boundary_s=upper_boundary_tensor,
            decision_regime=tuple(decision_regimes),
            sensory_sd_s=sensory_tensor,
            emission_normalizer=torch.stack(emission_normalizers),
            pre_choice_block_probability=torch.stack(pre_states),
            post_feedback_block_probability=torch.stack(post_states),
            valid_choice_mask=valid_mask,
            feedback_update_mask=session.feedback_valid,
            parameters=parameters,
        ),
        duration_s=session.duration_s,
        choice_long=session.choice_long,
    )


def evaluate_m1_mouse_objective(
    sessions: Iterable[M1SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M1ObjectiveWeighting = "pooled_trials",
    required_split_role: M1SplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
    first_long_probability: float = 0.5,
) -> M1MouseObjectiveResult:
    r"""Evaluate M1's full-batch pooled choice objective.

    Purpose
    -------
    Decode one six-parameter mouse vector and independently rebuild the HMM
    trajectory for every supplied session before returning one loss.

    Mathematical definition
    -----------------------
    The primary loss is

    .. math::

       J=\frac{\sum_d J_d}{\sum_d N_d},

    with the same optional equal-session sensitivity objective as M0.

    Parameter interpretation
    ------------------------
    Stable parameters are pooled; recursive block belief resets to neutral for
    each session.

    Input units and shapes
    ----------------------
    ``raw_parameters`` has shape ``(6,)`` and session durations use seconds.

    Returned quantities and shapes
    ------------------------------
    Returns a scalar loss, total NLL/count, decoded parameters, and all session
    evaluations.

    Constraints
    -----------
    Sessions belong to one mouse and the requested non-test split. Test input
    is always rejected.

    Numerical stability
    -------------------
    State normalizers and final loss are checked for finiteness in float64.

    Causal restrictions
    -------------------
    The objective includes choice likelihood only; category emissions update
    future hidden state and are never added as an auxiliary fitting target.

    Important edge cases
    --------------------
    Pooled weighting permits an omission-only session when another session has
    valid choices; equal-session weighting does not.

    Related tests
    -------------
    ``test_m1_pooled_objective_autograd_and_session_order_invariance``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one M1 session is required")
    if raw_parameters.shape != (6,):
        raise ValueError("M1 raw_parameters must have shape (6,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown M1 objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("An M1 pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter M1 objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M1 session must have role {required_split_role!r}")

    parameters = decode_m1_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_m1_session(
            session,
            parameters,
            moments,
            first_long_probability=first_long_probability,
            prior_probability_epsilon=prior_probability_epsilon,
            log_probability_epsilon=log_probability_epsilon,
            quadratic_tolerance=quadratic_tolerance,
            discriminant_tolerance=discriminant_tolerance,
        )
        for session in records
    )
    session_nll = torch.stack([record.negative_log_likelihood for record in evaluations])
    total_nll = session_nll.sum()
    total_valid = sum(record.valid_trial_count for record in evaluations)
    if total_valid == 0:
        raise ValueError("M1 objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(record.valid_trial_count == 0 for record in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M1 session")
        counts = session_nll.new_tensor([record.valid_trial_count for record in evaluations])
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M1 mouse objective is nonfinite")
    return M1MouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
