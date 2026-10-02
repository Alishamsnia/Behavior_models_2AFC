"""Causal, oracle-switch-timing evaluation for M1_known_switch_v1.

M1_known_switch_v1 keeps M1's fixed true-rate prototypes ``(0.5,0.2,0.8)`` and
its unchanged Bayesian sensory readout, but the model is told the true (oracle)
trial-by-trial block-switch index instead of inferring switch timing from a
fitted hazard. It is NOT told which category (short/long) a new biased block
is -- only that a switch occurred. This module derives that switch signal
from the canonical ``true_block`` column at the frame-adapter boundary and
never stores block identity in the resulting session tensors: the low-level
evaluator only ever sees a boolean "did a switch happen at this trial" flag.

Because blocks are known to strictly alternate Short<->Long once biased
(verified against real MC06/MC07/MC08 data before this model was written --
every session starts ``N`` and, after leaving it, alternates ``S``/``L`` with
no repeats and no re-entry into ``N``), knowing exact switch timing lets the
model track a scalar log-odds belief over which biased category the block
is, rather than the three-state hazard-driven belief M1 must maintain. The
fitted biased-block hazard is therefore absent from this model: conditioning
on the realized switch/no-switch outcome removes it from the recursion
entirely. See ``docs/models/M1_known_switch_v1.md`` for the full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability
from m0_sensory_v1.model.parameters import DecodedParameters

from .parameters import (
    M1_KNOWN_SWITCH_FIXED_PROTOTYPES,
    M1_KNOWN_SWITCH_LOG_LIKELIHOOD_RATIO,
    decode_m1_known_switch_raw_parameters,
)


M1KnownSwitchSplitRole = Literal["train", "validation", "test"]
M1KnownSwitchObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]

_VALID_TRUE_BLOCK_LABELS = ("N", "S", "L")


@dataclass(frozen=True)
class M1KnownSwitchSessionTrials:
    """One full task timeline plus the oracle block-switch indicator.

    ``is_switch[t]`` is True exactly when trial ``t`` is the first trial of a
    new block (including the trial that leaves Neutral); it carries no
    information about which category the new block is. ``is_switch[0]`` is
    always False by construction: trial 0 is always the session's own start.
    """

    mouse_id: str
    session_id: str
    duration_s: torch.Tensor
    choice_long: torch.Tensor
    category_long: torch.Tensor
    feedback_valid: torch.Tensor
    is_switch: torch.Tensor
    split_role: M1KnownSwitchSplitRole

    def __post_init__(self) -> None:
        """Validate timeline shape, registered precision, and binary fields."""

        if not self.mouse_id or not self.session_id:
            raise ValueError("mouse_id and session_id must be nonempty")
        if self.split_role not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split role: {self.split_role!r}")
        floating = (self.duration_s, self.choice_long, self.category_long)
        boolean = (self.feedback_valid, self.is_switch)
        if any(value.ndim != 1 for value in (*floating, *boolean)):
            raise ValueError("M1_known_switch session tensors must be one-dimensional")
        shapes = {tuple(value.shape) for value in (*floating, *boolean)}
        if len(shapes) != 1:
            raise ValueError("M1_known_switch session tensors must share one timeline shape")
        if self.duration_s.numel() == 0:
            raise ValueError("M1_known_switch sessions must contain at least one task trial")
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("M1_known_switch duration, choice, and category tensors must be floating")
        if any(value.dtype != torch.float64 for value in floating):
            raise TypeError("Registered M1_known_switch scientific tensors must use torch.float64")
        if any(value.dtype != torch.bool for value in boolean):
            raise TypeError("M1_known_switch feedback_valid and is_switch must use torch.bool")
        devices = {value.device for value in (*floating, *boolean)}
        if len(devices) != 1:
            raise ValueError("M1_known_switch session tensors must share a device")
        if bool(self.is_switch[0].detach().cpu()):
            raise ValueError(
                "M1_known_switch is_switch[0] must be False: trial 0 is always the "
                "session's own start, never a mid-session switch. A caller handing "
                "this model a non-session-initial slice is an unsupported, ambiguous "
                "input -- construct sessions only from their true start."
            )
        feedback_categories = self.category_long[self.feedback_valid]
        if feedback_categories.numel() and not bool(
            torch.all(
                torch.isfinite(feedback_categories)
                & ((feedback_categories == 0.0) | (feedback_categories == 1.0))
            )
        ):
            raise ValueError("Every feedback-valid M1_known_switch trial requires a binary category")


@dataclass(frozen=True)
class M1KnownSwitchPredictions:
    """Full-timeline M1_known_switch predictions and belief diagnostics."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    prior_long: torch.Tensor
    positive_threshold_s: torch.Tensor
    decision_lower_boundary_s: torch.Tensor
    decision_upper_boundary_s: torch.Tensor
    decision_regime: tuple[str, ...]
    sensory_sd_s: torch.Tensor
    pre_choice_block_probability: torch.Tensor
    post_feedback_block_probability: torch.Tensor
    biased_log_odds_pre: torch.Tensor
    biased_log_odds_post: torch.Tensor
    in_neutral: torch.Tensor
    valid_choice_mask: torch.Tensor
    feedback_update_mask: torch.Tensor
    is_switch: torch.Tensor
    parameters: DecodedParameters


@dataclass(frozen=True)
class M1KnownSwitchSessionEvaluation:
    """Differentiable M1_known_switch session choice NLL and trajectories."""

    mouse_id: str
    session_id: str
    split_role: M1KnownSwitchSplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M1KnownSwitchPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M1KnownSwitchMouseObjectiveResult:
    """One pooled M1_known_switch objective built from reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: M1KnownSwitchObjectiveWeighting
    session_evaluations: tuple[M1KnownSwitchSessionEvaluation, ...]
    parameters: DecodedParameters


def _validate_true_block_schedule(labels: list[str], *, session_id: str) -> None:
    r"""Reject any real-block-schedule violation of the alternation invariant.

    Purpose
    -------
    This model's log-odds recursion is only correct for a schedule that
    starts Neutral and, once left, strictly alternates Short/Long with no
    repeats and no re-entry into Neutral -- a structural fact that was
    empirically checked against real MC06/MC07/MC08 timelines before this
    model was written, not merely assumed from other models' documentation.
    A session violating it must fail loudly rather than silently produce a
    wrong belief trajectory.

    Mathematical definition
    -----------------------
    Let ``(b_1,...,b_k)`` be the run-length-compressed sequence of distinct
    consecutive ``true_block`` labels (compression already guarantees
    :math:`b_i\neq b_{i-1}`). This function requires :math:`b_1=N` and
    :math:`b_i\in\{S,L\}` for :math:`i>1`. Given only two non-Neutral labels
    and no adjacent repeats, :math:`b_i\neq b_{i-1}` for :math:`i>1`
    restricted to ``{S,L}`` already forces strict alternation -- there is no
    separate alternation check to perform once "starts N" and "never
    re-enters N" both hold.

    Parameter interpretation
    ------------------------
    ``labels`` is the complete, trial-ordered ``true_block`` sequence for one
    session, already restricted to ``{N,S,L}`` by the caller.

    Input units and shapes
    ----------------------
    A Python list of single-character block labels, one per task trial.

    Returned quantities and shapes
    ------------------------------
    None; raises on violation.

    Constraints
    -----------
    Requires at least one trial and every label in ``{N,S,L}``.

    Numerical stability
    -------------------
    Not applicable; this is a discrete structural check.

    Causal restrictions
    -------------------
    Uses only the trial-ordered label sequence, not any fitted quantity.

    Important edge cases
    --------------------
    A session with zero switches (a single Neutral or single biased block)
    passes trivially, since the alternation constraint is vacuous on a
    length-one compressed sequence starting ``N``... except a session that
    is entirely Neutral (never leaves) is otherwise a well-formed input to
    this model (it simply degenerates to a constant 0.5 prior throughout).

    Related tests
    -------------
    ``test_m1_known_switch_rejects_malformed_true_block_schedules``.
    """

    if not labels:
        raise ValueError("Cannot validate an empty true_block schedule")
    compressed: list[str] = []
    for label in labels:
        if not compressed or compressed[-1] != label:
            compressed.append(label)
    if compressed[0] != "N":
        raise ValueError(
            f"Session {session_id!r} true_block schedule must start 'N', got {compressed[0]!r}"
        )
    rest = compressed[1:]
    if any(label == "N" for label in rest):
        raise ValueError(
            f"Session {session_id!r} true_block schedule re-enters 'N' after leaving it: "
            f"{compressed!r}. M1_known_switch's log-odds recursion assumes Neutral is "
            "never re-entered, matching every other model in this family."
        )


def m1_known_switch_session_trials_from_frame(
    frame: pd.DataFrame,
    *,
    split_role: M1KnownSwitchSplitRole,
    device: torch.device | str | None = None,
) -> M1KnownSwitchSessionTrials:
    r"""Convert a canonical session table to M1_known_switch's oracle timeline.

    Purpose
    -------
    Derive the boolean switch signal this model is allowed to see from the
    canonical ``true_block`` column, validate the full real-block-schedule
    structure the recursion depends on, and then discard block identity
    entirely -- only ``is_switch`` reaches the session tensors below.

    Mathematical definition
    -----------------------
    For trial-ordered labels :math:`(\ell_1,\ldots,\ell_T)\in\{N,S,L\}^T`,

    .. math::

       e_1=\text{False},\qquad e_t=(\ell_t\neq\ell_{t-1})\ \text{for}\ t>1.

    Parameter interpretation
    ------------------------
    ``e_t`` (``is_switch``) is exactly the oracle signal this model receives:
    "did a new block begin at trial t". It is never accompanied by which
    category that new block is.

    Input units and shapes
    ----------------------
    Durations are seconds and every output tensor is one-dimensional with one
    element per original task trial.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`M1KnownSwitchSessionTrials` using ``torch.float64``
    for continuous fields and ``torch.bool`` for ``feedback_valid`` and
    ``is_switch``.

    Constraints
    -----------
    The frame must contain exactly one mouse/session, every ``true_block``
    entry must be in ``{N,S,L}``, the first trial's label must be ``N``, and
    the full schedule must satisfy the alternation invariant (see
    :func:`_validate_true_block_schedule`).

    Numerical stability
    -------------------
    Pandas missing values in continuous columns become IEEE NaN without being
    coerced; missing/invalid ``true_block`` values are a hard error, not a
    silently-treated gap, because they cannot be safely interpreted as either
    "no switch" or "switch" for this model.

    Causal restrictions
    -------------------
    True category, reward, true run length, and block identity are never
    carried into the returned tensors -- only ``is_switch``.

    Important edge cases
    --------------------
    A session that never leaves Neutral (``is_switch`` all False) is valid
    and simply predicts a constant 0.5 prior throughout.

    Related tests
    -------------
    ``test_m1_known_switch_frame_adapter_derives_switch_flags_and_rejects_gaps``.
    """

    required = {
        "mouse_id",
        "session_id",
        "trial_index",
        "duration_s",
        "choice_long",
        "category_long",
        "feedback_valid",
        "true_block",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Canonical M1_known_switch frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot construct M1KnownSwitchSessionTrials from an empty frame")
    identifiers = frame[["mouse_id", "session_id"]].drop_duplicates()
    if len(identifiers) != 1:
        raise ValueError("M1_known_switch frame must contain exactly one mouse/session pair")
    identifier = identifiers.iloc[0]
    session_id = str(identifier["session_id"])

    numeric_trial_index = pd.to_numeric(frame["trial_index"], errors="coerce")
    if numeric_trial_index.isna().any() or not numeric_trial_index.astype(float).apply(
        float.is_integer
    ).all():
        raise ValueError(f"Session {session_id!r} has non-integer or missing trial_index values")
    frame = frame.assign(trial_index=numeric_trial_index.astype(int))
    ordered = frame.sort_values("trial_index")
    trial_index = ordered["trial_index"]
    if trial_index.duplicated().any():
        raise ValueError(f"Session {session_id!r} has duplicate trial_index values")
    if int(trial_index.min()) != 1:
        raise ValueError(
            f"Session {session_id!r} trial_index must start at 1 (the session's true first "
            f"trial); observed minimum {int(trial_index.min())}. A dropped leading trial "
            "would silently discard the session's real Neutral start."
        )
    if int(trial_index.max()) - int(trial_index.min()) + 1 != len(trial_index):
        raise ValueError(
            f"Session {session_id!r} has a gap in trial_index: switch timing cannot be "
            "safely derived across a missing task trial. Pass the complete session timeline."
        )
    true_block_raw = ordered["true_block"]
    if true_block_raw.isna().any():
        raise ValueError(
            f"Session {session_id!r} has missing true_block entries; M1_known_switch "
            "cannot safely derive switch timing across a gap in the oracle schedule."
        )
    labels = [str(value) for value in true_block_raw.tolist()]
    invalid = sorted(set(labels) - set(_VALID_TRUE_BLOCK_LABELS))
    if invalid:
        raise ValueError(
            f"Session {session_id!r} has true_block values outside {{'N','S','L'}}: {invalid}"
        )
    _validate_true_block_schedule(labels, session_id=session_id)

    is_switch = [False] + [labels[index] != labels[index - 1] for index in range(1, len(labels))]

    def numeric(column: str) -> np.ndarray:
        return pd.to_numeric(ordered[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        ).copy()

    feedback_numeric = numeric("feedback_valid")
    if np.any(~np.isin(feedback_numeric[np.isfinite(feedback_numeric)], [0.0, 1.0])):
        raise ValueError("feedback_valid must be binary or missing")
    feedback = np.isfinite(feedback_numeric) & (feedback_numeric == 1.0)

    return M1KnownSwitchSessionTrials(
        mouse_id=str(identifier["mouse_id"]),
        session_id=session_id,
        duration_s=torch.as_tensor(numeric("duration_s"), dtype=torch.float64, device=device),
        choice_long=torch.as_tensor(numeric("choice_long"), dtype=torch.float64, device=device),
        category_long=torch.as_tensor(numeric("category_long"), dtype=torch.float64, device=device),
        feedback_valid=torch.as_tensor(feedback, dtype=torch.bool, device=device),
        is_switch=torch.as_tensor(np.array(is_switch, dtype=bool), dtype=torch.bool, device=device),
        split_role=split_role,
    )


def evaluate_m1_known_switch_session(
    session: M1KnownSwitchSessionTrials,
    parameters: DecodedParameters,
    moments: TaskMoments,
    *,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M1KnownSwitchSessionEvaluation:
    r"""Evaluate one M1_known_switch session under the oracle switch schedule.

    Purpose
    -------
    Produce the causal choice likelihood by tracking a scalar biased-subspace
    log-odds belief that is updated deterministically from the given switch
    flags and probabilistically from category feedback -- no hazard is
    inferred or fitted.

    Mathematical definition
    -----------------------
    With fixed prototypes :math:`\theta=(0.5,0.2,0.8)` and biased-subspace
    log-odds :math:`r_t=\log\frac{P(Z_t=L)}{P(Z_t=S)}`:

    .. math::

       r_t^-=\begin{cases}
       0, & \text{first switch (leaving Neutral)},\\
       -r_{t-1}^+, & \text{later switch},\\
       r_{t-1}^+, & \text{no switch},
       \end{cases}
       \qquad
       \pi_t=\begin{cases}
       0.5, & \text{still Neutral},\\
       \theta_S+(\theta_L-\theta_S)\sigma(r_t^-), & \text{biased}.
       \end{cases}

    After a valid category observation :math:`C_t\in\{0,1\}` while biased,

    .. math::

       r_t^+=r_t^- + (2C_t-1)\log 4.

    Parameter interpretation
    ------------------------
    ``log 4 = log(theta_L/theta_S)`` is the fixed per-observation
    log-likelihood-ratio implied by M1's true rates. There is no hazard: the
    switch timing is supplied exactly, so it never needs to be inferred.

    Input units and shapes
    ----------------------
    Session tensors have length ``T``; durations and ``s0`` are seconds.

    Returned quantities and shapes
    ------------------------------
    Returns scalar choice NLL, full-timeline probabilities, and the
    reconstructed ``(T,3)`` pre-/post-feedback ``(N,S,L)`` belief (for export
    parity with M1/M2) alongside the raw scalar log-odds trajectory.

    Constraints
    -----------
    ``session.is_switch`` must already satisfy the alternation invariant
    (enforced at frame-adapter time, not re-checked here).

    Numerical stability
    -------------------
    Only probabilities entering logs are clamped; saved predictions remain
    unclamped. The log-odds recursion is a plain sum with no risk of
    underflow that a raw probability recursion would have.

    Causal restrictions
    -------------------
    Current category and feedback cannot affect current ``prior_long`` or
    ``probability_long``. Block identity (which category the switch entered)
    is never available to the recursion, only that a switch occurred.

    Important edge cases
    --------------------
    Feedback observed while still Neutral has no effect on belief, exactly
    matching real M1 (a one-hot Neutral belief is invariant to any emission
    reweighting). Invalid choices add no likelihood but the switch/feedback
    state update still applies.

    Related tests
    -------------
    ``test_m1_known_switch_matches_reference_switch_filter`` and
    ``test_m1_known_switch_zero_switches_gives_constant_neutral_prior``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    theta_n, theta_s, theta_l = M1_KNOWN_SWITCH_FIXED_PROTOTYPES
    log_likelihood_ratio = parameters.w.new_tensor(M1_KNOWN_SWITCH_LOG_LIKELIHOOD_RATIO)
    theta_s_t = parameters.w.new_tensor(theta_s)
    theta_l_t = parameters.w.new_tensor(theta_l)

    priors: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    non_lapse: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    lower_boundaries: list[torch.Tensor] = []
    upper_boundaries: list[torch.Tensor] = []
    sensory_sd: list[torch.Tensor] = []
    decision_regimes: list[str] = []
    pre_states: list[torch.Tensor] = []
    post_states: list[torch.Tensor] = []
    log_odds_pre: list[torch.Tensor] = []
    log_odds_post: list[torch.Tensor] = []
    in_neutral_flags: list[bool] = []
    valid_nll: list[torch.Tensor] = []

    valid_mask = (
        torch.isfinite(session.duration_s)
        & (session.duration_s > 0.0)
        & torch.isfinite(session.choice_long)
        & ((session.choice_long == 0.0) | (session.choice_long == 1.0))
    )

    switches_seen = 0
    r_post_prev = parameters.w.new_tensor(0.0)
    zero = parameters.w.new_tensor(0.0)
    for index in range(session.duration_s.numel()):
        was_switch = bool(session.is_switch[index].detach().cpu())
        if was_switch:
            switches_seen += 1
        in_neutral = switches_seen == 0
        in_neutral_flags.append(in_neutral)

        if in_neutral:
            r_pre = zero
            prior = parameters.w.new_tensor(0.5)
            pre_state = parameters.w.new_tensor([1.0, 0.0, 0.0])
        else:
            first_switch = was_switch and switches_seen == 1
            if first_switch:
                r_pre = zero
            elif was_switch:
                r_pre = -r_post_prev
            else:
                r_pre = r_post_prev
            prior = theta_s_t + (theta_l_t - theta_s_t) * torch.sigmoid(r_pre)
            p_long = torch.sigmoid(r_pre)
            pre_state = torch.stack((zero, 1.0 - p_long, p_long))
        log_odds_pre.append(r_pre)
        pre_states.append(pre_state)
        priors.append(prior)

        if bool(valid_mask[index].detach().cpu()):
            result = m1_bayesian_choice_probability(
                session.duration_s[index],
                parameters,
                moments,
                prior,
                prior_probability_epsilon=prior_probability_epsilon,
                quadratic_tolerance=quadratic_tolerance,
                discriminant_tolerance=discriminant_tolerance,
            )
            probability = lapse_mixture(
                result.probability_long,
                session.category_long[index],
                parameters,
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

        if in_neutral:
            r_post = r_pre
            post_state = pre_state
        elif bool(session.feedback_valid[index].detach().cpu()):
            category = session.category_long[index]
            r_post = r_pre + (2.0 * category - 1.0) * log_likelihood_ratio
            p_long = torch.sigmoid(r_post)
            post_state = torch.stack((zero, 1.0 - p_long, p_long))
        else:
            r_post = r_pre
            post_state = pre_state
        log_odds_post.append(r_post)
        post_states.append(post_state)
        r_post_prev = r_post

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

    return M1KnownSwitchSessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=M1KnownSwitchPredictions(
            probability_long=probability_tensor,
            non_lapse_probability_long=non_lapse_tensor,
            prior_long=prior_tensor,
            positive_threshold_s=threshold_tensor,
            decision_lower_boundary_s=lower_boundary_tensor,
            decision_upper_boundary_s=upper_boundary_tensor,
            decision_regime=tuple(decision_regimes),
            sensory_sd_s=sensory_tensor,
            pre_choice_block_probability=torch.stack(pre_states),
            post_feedback_block_probability=torch.stack(post_states),
            biased_log_odds_pre=torch.stack(log_odds_pre),
            biased_log_odds_post=torch.stack(log_odds_post),
            in_neutral=torch.tensor(
                in_neutral_flags, dtype=torch.bool, device=session.duration_s.device
            ),
            valid_choice_mask=valid_mask,
            feedback_update_mask=session.feedback_valid,
            is_switch=session.is_switch,
            parameters=parameters,
        ),
        duration_s=session.duration_s,
        choice_long=session.choice_long,
    )


def evaluate_m1_known_switch_mouse_objective(
    sessions: Iterable[M1KnownSwitchSessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M1KnownSwitchObjectiveWeighting = "pooled_trials",
    required_split_role: M1KnownSwitchSplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M1KnownSwitchMouseObjectiveResult:
    r"""Evaluate M1_known_switch's full-batch pooled choice objective.

    Purpose
    -------
    Decode one five-parameter mouse vector and independently rebuild the
    oracle-switch trajectory for every supplied session before returning one
    loss.

    Mathematical definition
    -----------------------
    .. math::

       J=\frac{\sum_d J_d}{\sum_d N_d},

    with the same optional equal-session sensitivity objective as M0/M1.

    Parameter interpretation
    ------------------------
    Stable parameters are pooled across sessions; the log-odds belief resets
    to Neutral (undefined, prior 0.5) at the start of every session.

    Input units and shapes
    ----------------------
    ``raw_parameters`` has shape ``(5,)`` and session durations use seconds.

    Returned quantities and shapes
    ------------------------------
    Returns a scalar loss, total NLL/count, decoded parameters, and all
    session evaluations.

    Constraints
    -----------
    Sessions belong to one mouse and the requested non-test split. Test input
    is always rejected.

    Numerical stability
    -------------------
    State normalizers and final loss are checked for finiteness in float64.

    Causal restrictions
    -------------------
    The objective includes choice likelihood only; category feedback updates
    future belief and is never added as an auxiliary fitting target.

    Important edge cases
    --------------------
    Pooled weighting permits an omission-only session when another session
    has valid choices; equal-session weighting does not.

    Related tests
    -------------
    ``test_m1_known_switch_pooled_objective_autograd_and_session_order_invariance``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one M1_known_switch session is required")
    if raw_parameters.shape != (5,):
        raise ValueError("M1_known_switch raw_parameters must have shape (5,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown M1_known_switch objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("An M1_known_switch pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter M1_known_switch objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M1_known_switch session must have role {required_split_role!r}")

    parameters = decode_m1_known_switch_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_m1_known_switch_session(
            session,
            parameters,
            moments,
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
        raise ValueError("M1_known_switch objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(record.valid_trial_count == 0 for record in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M1_known_switch session")
        counts = session_nll.new_tensor([record.valid_trial_count for record in evaluations])
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M1_known_switch mouse objective is nonfinite")
    return M1KnownSwitchMouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
