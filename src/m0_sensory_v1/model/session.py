"""Independent M0 session evaluation and pooled mouse-level objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch

from ..data.task_moments import TaskMoments
from .m0 import MaskedM0Predictions, predict_m0_trials_with_parameters
from .parameters import DecodedParameters, decode_raw_parameters


SplitRole = Literal["train", "validation", "test"]
ObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]


@dataclass(frozen=True)
class SessionTrials:
    """One session's complete trial timeline and immutable identifiers."""

    mouse_id: str
    session_id: str
    duration_s: torch.Tensor
    category_long: torch.Tensor
    choice_long: torch.Tensor
    split_role: SplitRole

    def __post_init__(self) -> None:
        """Validate shape, precision, device, and split metadata."""

        if not self.mouse_id or not self.session_id:
            raise ValueError("mouse_id and session_id must be nonempty")
        if self.split_role not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split role: {self.split_role!r}")
        if self.duration_s.ndim != 1 or self.category_long.ndim != 1 or self.choice_long.ndim != 1:
            raise ValueError("Session trial tensors must be one-dimensional")
        if self.duration_s.shape != self.category_long.shape or self.duration_s.shape != self.choice_long.shape:
            raise ValueError("Session duration, category, and choice tensors must share shape")
        if not all(x.is_floating_point() for x in (self.duration_s, self.category_long, self.choice_long)):
            raise TypeError("Session duration, category, and choice tensors must be floating")
        if len({self.duration_s.dtype, self.category_long.dtype, self.choice_long.dtype}) != 1:
            raise TypeError("Session duration, category, and choice tensors must share dtype")
        if self.duration_s.dtype != torch.float64:
            raise TypeError("Registered M0 session tensors must use torch.float64")
        if self.duration_s.device != self.category_long.device or self.duration_s.device != self.choice_long.device:
            raise ValueError("Session duration, category, and choice tensors must share device")


@dataclass(frozen=True)
class SessionEvaluation:
    """Differentiable likelihood result and full-timeline diagnostics."""

    mouse_id: str
    session_id: str
    split_role: SplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: MaskedM0Predictions
    duration_s: torch.Tensor
    category_long: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class MouseObjectiveResult:
    """One pooled mouse objective and its independently evaluated sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: ObjectiveWeighting
    session_evaluations: tuple[SessionEvaluation, ...]
    parameters: DecodedParameters


def session_trials_from_frame(
    frame: pd.DataFrame,
    *,
    split_role: SplitRole,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SessionTrials:
    r"""Convert one canonical session frame to omission-preserving Torch tensors.

    Purpose
    -------
    Adapt the canonical dataframe into the strict one-session interface without
    dropping rows or converting missing decisions into artificial actions.

    Mathematical definition
    -----------------------
    The resulting tensors retain, for one session,

    .. math::

       \{(s_{d,\tau},y_{d,\tau})\}_{\tau=1}^{n_d}.

    Undefined :math:`y_{d,\tau}` remain NaN and are later excluded from
    :math:`\mathcal V_d`.

    Parameter interpretation
    ------------------------
    ``split_role`` is immutable metadata used to prevent locked-test leakage.

    Input units and shapes
    ----------------------
    ``duration_s`` is seconds; ``choice_long`` is 0, 1, or missing. ``frame``
    must contain exactly one mouse/session pair and any number of trial rows.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`SessionTrials` with one-dimensional same-length tensors.

    Constraints
    -----------
    Registered computations use ``torch.float64``. Session identifiers must be
    constant within the frame.

    Numerical stability
    -------------------
    Pandas nullable values become IEEE NaN, not zero or a sentinel choice.

    Causal restrictions
    -------------------
    Duration, physical short/long stimulus category, and choice eligibility are
    transferred. Block, reward, history, and trial position are not model inputs.

    Important edge cases
    --------------------
    Empty frames and frames combining sessions raise. An all-omission nonempty
    session is permitted and later reports zero valid trials.

    Related tests
    -------------
    ``test_session_frame_adapter_preserves_missing_choices``.
    """

    required = {"mouse_id", "session_id", "duration_s", "category_long", "choice_long"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Canonical session frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot construct SessionTrials from an empty frame")
    identifiers = frame[["mouse_id", "session_id"]].drop_duplicates()
    if len(identifiers) != 1:
        raise ValueError("Frame must contain exactly one mouse/session pair")

    duration = pd.to_numeric(frame["duration_s"], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    ).copy()
    choice = pd.to_numeric(frame["choice_long"], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    ).copy()
    category = pd.to_numeric(frame["category_long"], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    ).copy()
    identifier = identifiers.iloc[0]
    return SessionTrials(
        mouse_id=str(identifier["mouse_id"]),
        session_id=str(identifier["session_id"]),
        duration_s=torch.as_tensor(duration, dtype=dtype, device=device),
        category_long=torch.as_tensor(category, dtype=dtype, device=device),
        choice_long=torch.as_tensor(choice, dtype=dtype, device=device),
        split_role=split_role,
    )


def evaluate_session(
    session: SessionTrials,
    parameters: DecodedParameters,
    moments: TaskMoments,
    *,
    log_probability_epsilon: float = 1e-12,
) -> SessionEvaluation:
    r"""Evaluate one session independently and sum its valid Bernoulli NLL.

    Purpose
    -------
    Apply the common session boundary required by M0 and future stateful models.

    Mathematical definition
    -----------------------
    Let

    .. math::

       \mathcal V_d=\{\tau:y_{d,\tau}\in\{0,1\},\ s_{d,\tau}>0
       \text{ finite}\}.

    The returned session loss is

    .. math::

       J_d=-\sum_{\tau\in\mathcal V_d}
       \left[y_{d,\tau}\log p_{d,\tau}
       +(1-y_{d,\tau})\log(1-p_{d,\tau})\right],

    with :math:`N_d=|\mathcal V_d|`.

    Parameter interpretation
    ------------------------
    The same decoded mouse-level ``(w,s0,ell_short,ell_long,kappa)`` is supplied to every session;
    parameters do not reset at session boundaries.

    Input units and shapes
    ----------------------
    One-dimensional duration/choice tensors describe exactly one session.

    Returned quantities and shapes
    ------------------------------
    Returns scalar ``J_d``, integer ``N_d``, full-timeline predictions, and a
    full-timeline per-trial NLL tensor with NaN on excluded rows.

    Constraints
    -----------
    Every session is evaluated independently. No dynamic or trial-history state
    from one session may enter another session. M0 has no dynamic state.

    Numerical stability
    -------------------
    Only probabilities entering logarithms are clamped to
    ``[epsilon, 1-epsilon]``. Unclamped predictions remain in diagnostics.

    Causal restrictions
    -------------------
    Outcomes determine choice reconstruction upstream but do not enter the
    prediction. Session ID and split role are metadata only.

    Important edge cases
    --------------------
    An all-invalid session returns a differentiable zero NLL and ``N_d=0``;
    pooled/equal objective policies decide whether that dataset is admissible.

    Related tests
    -------------
    ``test_session_evaluator_matches_manual_bernoulli_sum_and_skips_omissions``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0, 0.5)")

    predictions = predict_m0_trials_with_parameters(
        session.duration_s,
        session.category_long,
        session.choice_long,
        parameters,
        moments,
    )
    valid = predictions.valid_choice_mask
    valid_count = int(valid.sum().detach().cpu())
    if valid_count == 0:
        zero = parameters.as_tensor().sum() * 0.0
        return SessionEvaluation(
            mouse_id=session.mouse_id,
            session_id=session.session_id,
            split_role=session.split_role,
            negative_log_likelihood=zero,
            valid_trial_count=0,
            per_trial_negative_log_likelihood=torch.full_like(
                session.duration_s,
                torch.nan,
            ),
            predictions=predictions,
            duration_s=session.duration_s,
            category_long=session.category_long,
            choice_long=session.choice_long,
        )

    probability = predictions.probability_long[valid]
    observed = session.choice_long[valid]
    safe_probability = torch.clamp(
        probability,
        min=log_probability_epsilon,
        max=1.0 - log_probability_epsilon,
    )
    valid_nll = -(
        observed * torch.log(safe_probability)
        + (1.0 - observed) * torch.log1p(-safe_probability)
    )
    per_trial_nll = torch.full_like(session.duration_s, torch.nan).masked_scatter(
        valid,
        valid_nll,
    )
    return SessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=valid_nll.sum(),
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=predictions,
        duration_s=session.duration_s,
        category_long=session.category_long,
        choice_long=session.choice_long,
    )


def evaluate_mouse_objective(
    sessions: Iterable[SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: ObjectiveWeighting = "pooled_trials",
    required_split_role: SplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
) -> MouseObjectiveResult:
    r"""Evaluate all sessions before returning one differentiable mouse loss.

    Purpose
    -------
    Implement the registered pooled training objective while preserving explicit
    session boundaries and offering the equal-session sensitivity objective.

    Mathematical definition
    -----------------------
    The primary registered objective is

    .. math::

       J_{\mathrm{pooled}}=
       \frac{\sum_d J_d}{\sum_d N_d}.

    The optional sensitivity objective is

    .. math::

       J_{\mathrm{equal}}=\frac{1}{D}\sum_d\frac{J_d}{N_d}.

    These are different when session lengths differ.

    Parameter interpretation
    ------------------------
    One raw vector is decoded exactly once and shared across every session of
    one mouse. Session state is not pooled; M0's session state is empty.

    Input units and shapes
    ----------------------
    ``sessions`` is a nonempty iterable of :class:`SessionTrials` from one
    mouse. ``raw_parameters`` has shape ``(5,)`` and uses ``float64``.

    Returned quantities and shapes
    ------------------------------
    Returns scalar loss/total NLL, total valid count, decoded parameters, and
    every independent session evaluation.

    Constraints
    -----------
    All sessions must match ``required_split_role``. Locked test sessions are
    always rejected. The registered primary weighting is ``pooled_trials``.

    Numerical stability
    -------------------
    Session NLLs are accumulated before one normalization. The function checks
    the final loss for finiteness but never calls ``backward`` or an optimizer.

    Causal restrictions
    -------------------
    Only duration and completed binary choice likelihoods contribute. Session
    order, session ID, block, category, outcome, and history cannot change M0.

    Important edge cases
    --------------------
    The pooled objective permits a zero-valid session if another session has
    valid trials. Equal-session weighting rejects any ``N_d=0`` because
    ``J_d/N_d`` is undefined. A dataset with no valid trials always raises.

    Related tests
    -------------
    ``test_pooled_and_equal_session_objectives_match_manual_formulas``,
    ``test_reordering_sessions_preserves_loss_and_gradients``, and
    ``test_locked_test_session_is_rejected``.
    """

    session_tuple = tuple(sessions)
    if not session_tuple:
        raise ValueError("At least one session is required")
    if raw_parameters.shape != (5,):
        raise ValueError("raw_parameters must have shape (5,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown objective weighting: {objective_weighting!r}")

    mouse_ids = {session.mouse_id for session in session_tuple}
    if len(mouse_ids) != 1:
        raise ValueError("A pooled objective may contain sessions from only one mouse")
    if any(session.split_role == "test" for session in session_tuple):
        raise ValueError("Locked test sessions cannot enter objective evaluation")
    if any(session.split_role != required_split_role for session in session_tuple):
        raise ValueError(
            f"Every session must have split_role={required_split_role!r}"
        )

    parameters = decode_raw_parameters(
        raw_parameters,
        lapse_max=lapse_max,
        epsilon=parameter_epsilon,
    )
    evaluations = tuple(
        evaluate_session(
            session,
            parameters,
            moments,
            log_probability_epsilon=log_probability_epsilon,
        )
        for session in session_tuple
    )
    session_nlls = torch.stack(
        [evaluation.negative_log_likelihood for evaluation in evaluations]
    )
    total_nll = session_nlls.sum()
    total_valid = sum(evaluation.valid_trial_count for evaluation in evaluations)
    if total_valid == 0:
        raise ValueError("Mouse objective contains no valid completed choices")

    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(evaluation.valid_trial_count == 0 for evaluation in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid session")
        counts = session_nlls.new_tensor(
            [evaluation.valid_trial_count for evaluation in evaluations]
        )
        loss = torch.mean(session_nlls / counts)

    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("Mouse objective is nonfinite")
    return MouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
