"""Causal, oracle-switch-timing evaluation for M2_known_switch_v1.

This is an independent implementation, not a runtime dependency on
``m1_known_switch_v1``: it reuses only genuinely shared, pre-existing
primitives from ``m0_sensory_v1`` (the sensory readout, lapse mixture, and
parameter decoder), matching the pattern already used by
``m1_dynamic_block_length_v1``/``m4_hsmm_delta_v1`` in this codebase. The two
``known_switch`` models are cross-verified against each other (exact nesting
at ``delta0=log(4)``) only in the test suite -- see
``tests/test_m2_known_switch_model.py``.

M2_known_switch_v1 keeps M2's fitted, symmetric, static subjective rates
``(0.5, sigmoid(-delta0), sigmoid(delta0))`` and its unchanged Bayesian
sensory readout, but the model is told the true (oracle) trial-by-trial
block-switch index instead of inferring switch timing from a fitted hazard.
See ``docs/models/M2_known_switch_v1.md`` for the full derivation and
``m1_known_switch_v1/session.py`` for the sibling model this mirrors.
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

from .parameters import M2KnownSwitchParameters, decode_m2_known_switch_raw_parameters


M2KnownSwitchSplitRole = Literal["train", "validation", "test"]
M2KnownSwitchObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]

_VALID_TRUE_BLOCK_LABELS = ("N", "S", "L")


@dataclass(frozen=True)
class M2KnownSwitchSessionTrials:
    """One full task timeline plus the oracle block-switch indicator.

    Identical in shape and meaning to ``m1_known_switch_v1``'s session
    contract (``is_switch[t]`` True exactly at the first trial of a new
    block, ``is_switch[0]`` always False); duplicated here rather than
    imported so this model has no runtime dependency on its sibling.
    """

    mouse_id: str
    session_id: str
    duration_s: torch.Tensor
    choice_long: torch.Tensor
    category_long: torch.Tensor
    feedback_valid: torch.Tensor
    is_switch: torch.Tensor
    split_role: M2KnownSwitchSplitRole

    def __post_init__(self) -> None:
        """Validate timeline shape, registered precision, and binary fields."""

        if not self.mouse_id or not self.session_id:
            raise ValueError("mouse_id and session_id must be nonempty")
        if self.split_role not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split role: {self.split_role!r}")
        floating = (self.duration_s, self.choice_long, self.category_long)
        boolean = (self.feedback_valid, self.is_switch)
        if any(value.ndim != 1 for value in (*floating, *boolean)):
            raise ValueError("M2_known_switch session tensors must be one-dimensional")
        shapes = {tuple(value.shape) for value in (*floating, *boolean)}
        if len(shapes) != 1:
            raise ValueError("M2_known_switch session tensors must share one timeline shape")
        if self.duration_s.numel() == 0:
            raise ValueError("M2_known_switch sessions must contain at least one task trial")
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("M2_known_switch duration, choice, and category tensors must be floating")
        if any(value.dtype != torch.float64 for value in floating):
            raise TypeError("Registered M2_known_switch scientific tensors must use torch.float64")
        if any(value.dtype != torch.bool for value in boolean):
            raise TypeError("M2_known_switch feedback_valid and is_switch must use torch.bool")
        devices = {value.device for value in (*floating, *boolean)}
        if len(devices) != 1:
            raise ValueError("M2_known_switch session tensors must share a device")
        if bool(self.is_switch[0].detach().cpu()):
            raise ValueError(
                "M2_known_switch is_switch[0] must be False: trial 0 is always the "
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
            raise ValueError("Every feedback-valid M2_known_switch trial requires a binary category")


@dataclass(frozen=True)
class M2KnownSwitchPredictions:
    """Full-timeline M2_known_switch predictions and belief diagnostics."""

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
    parameters: M2KnownSwitchParameters


@dataclass(frozen=True)
class M2KnownSwitchSessionEvaluation:
    """Differentiable M2_known_switch session choice NLL and trajectories."""

    mouse_id: str
    session_id: str
    split_role: M2KnownSwitchSplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M2KnownSwitchPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M2KnownSwitchMouseObjectiveResult:
    """One pooled M2_known_switch objective built from reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: M2KnownSwitchObjectiveWeighting
    session_evaluations: tuple[M2KnownSwitchSessionEvaluation, ...]
    parameters: M2KnownSwitchParameters


def _validate_true_block_schedule(labels: list[str], *, session_id: str) -> None:
    r"""Reject any real-block-schedule violation of the alternation invariant.

    Purpose
    -------
    Identical requirement to ``m1_known_switch_v1``'s adapter check
    (duplicated rather than imported -- see module docstring): the recursion
    is only correct for a schedule that starts Neutral and never re-enters
    it. Given only two non-Neutral labels and run-length compression's
    guaranteed no-adjacent-duplicates property, those two conditions already
    force strict Short/Long alternation with no separate check needed.

    Mathematical definition
    -----------------------
    Let ``(b_1,...,b_k)`` be the run-length-compressed sequence of distinct
    consecutive ``true_block`` labels. This function requires :math:`b_1=N`
    and :math:`b_i\in\{S,L\}` for :math:`i>1`.

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
    A session that never leaves Neutral is valid and degenerates to a
    constant 0.5 prior throughout.

    Related tests
    -------------
    ``test_m2_known_switch_frame_adapter_derives_switch_flags_and_rejects_gaps``.
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
            f"{compressed!r}. M2_known_switch's log-odds recursion assumes Neutral is "
            "never re-entered, matching every other model in this family."
        )


def m2_known_switch_session_trials_from_frame(
    frame: pd.DataFrame,
    *,
    split_role: M2KnownSwitchSplitRole,
    device: torch.device | str | None = None,
) -> M2KnownSwitchSessionTrials:
    r"""Convert a canonical session table to M2_known_switch's oracle timeline.

    Purpose
    -------
    Derive the boolean switch signal from the canonical ``true_block``
    column, validate the real-block-schedule structure, and discard block
    identity entirely before returning session tensors.

    Mathematical definition
    -----------------------
    For trial-ordered labels :math:`(\ell_1,\ldots,\ell_T)\in\{N,S,L\}^T`,

    .. math::

       e_1=\text{False},\qquad e_t=(\ell_t\neq\ell_{t-1})\ \text{for}\ t>1.

    Parameter interpretation
    ------------------------
    ``e_t`` (``is_switch``) is the only oracle signal this model receives.

    Input units and shapes
    ----------------------
    Durations are seconds and every output tensor is one-dimensional with one
    element per original task trial.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`M2KnownSwitchSessionTrials` using ``torch.float64``
    for continuous fields and ``torch.bool`` for ``feedback_valid`` and
    ``is_switch``.

    Constraints
    -----------
    The frame must contain exactly one mouse/session, every ``true_block``
    entry must be in ``{N,S,L}``, the first trial's label must be ``N``, and
    the full schedule must satisfy the alternation invariant.

    Numerical stability
    -------------------
    Missing/invalid ``true_block`` values are a hard error, not a silently
    treated gap.

    Causal restrictions
    -------------------
    True category, reward, true run length, and block identity are never
    carried into the returned tensors -- only ``is_switch``.

    Important edge cases
    --------------------
    A session that never leaves Neutral is valid.

    Related tests
    -------------
    ``test_m2_known_switch_frame_adapter_derives_switch_flags_and_rejects_gaps``.
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
        raise KeyError(f"Canonical M2_known_switch frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot construct M2KnownSwitchSessionTrials from an empty frame")
    identifiers = frame[["mouse_id", "session_id"]].drop_duplicates()
    if len(identifiers) != 1:
        raise ValueError("M2_known_switch frame must contain exactly one mouse/session pair")
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
            f"Session {session_id!r} has missing true_block entries; M2_known_switch "
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

    return M2KnownSwitchSessionTrials(
        mouse_id=str(identifier["mouse_id"]),
        session_id=session_id,
        duration_s=torch.as_tensor(numeric("duration_s"), dtype=torch.float64, device=device),
        choice_long=torch.as_tensor(numeric("choice_long"), dtype=torch.float64, device=device),
        category_long=torch.as_tensor(numeric("category_long"), dtype=torch.float64, device=device),
        feedback_valid=torch.as_tensor(feedback, dtype=torch.bool, device=device),
        is_switch=torch.as_tensor(np.array(is_switch, dtype=bool), dtype=torch.bool, device=device),
        split_role=split_role,
    )


def evaluate_m2_known_switch_session(
    session: M2KnownSwitchSessionTrials,
    parameters: M2KnownSwitchParameters,
    moments: TaskMoments,
    *,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M2KnownSwitchSessionEvaluation:
    r"""Evaluate one M2_known_switch session under the oracle switch schedule.

    Purpose
    -------
    Produce the causal choice likelihood by tracking a scalar biased-subspace
    log-odds belief updated deterministically from the given switch flags and
    probabilistically from category feedback, using M2's fitted static
    prototypes in place of M1_known_switch_v1's fixed rates.

    Mathematical definition
    -----------------------
    With prototypes :math:`\theta=(0.5,\sigma(-\delta_0),\sigma(\delta_0))`
    and biased-subspace log-odds :math:`r_t=\log\frac{P(Z_t=L)}{P(Z_t=S)}`:

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

       r_t^+=r_t^- + (2C_t-1)\delta_0.

    Parameter interpretation
    ------------------------
    ``delta0`` is both the static contrast defining the prototypes and,
    equivalently, ``log(theta_L/theta_S)``, the fitted per-observation
    log-likelihood-ratio. There is no hazard.

    Input units and shapes
    ----------------------
    Session tensors have length ``T``; durations and ``s0`` are seconds.

    Returned quantities and shapes
    ------------------------------
    Returns scalar choice NLL, full-timeline probabilities, and the
    reconstructed ``(T,3)`` pre-/post-feedback ``(N,S,L)`` belief alongside
    the raw scalar log-odds trajectory.

    Constraints
    -----------
    ``session.is_switch`` must already satisfy the alternation invariant
    (enforced at frame-adapter time, not re-checked here).

    Numerical stability
    -------------------
    Only probabilities entering logs are clamped; saved predictions remain
    unclamped.

    Causal restrictions
    -------------------
    Current category and feedback cannot affect current ``prior_long`` or
    ``probability_long``. Block identity is never available to the
    recursion, only that a switch occurred.

    Important edge cases
    --------------------
    At ``delta0=0`` (via ``with_explicit_delta0``), ``theta_S=theta_L=0.5``
    and the prior is constant 0.5 regardless of switches, matching M0's own
    fixed neutral prior. At ``delta0=log(4)`` this model matches
    ``m1_known_switch_v1`` trial-by-trial (verified in the test suite, not
    at runtime -- see module docstring).

    Related tests
    -------------
    ``test_m2_known_switch_matches_reference_switch_filter`` and
    ``test_m2_known_switch_nests_m1_known_switch_at_log4``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    theta_n, theta_s_t, theta_l_t = parameters.prototypes().unbind(-1)
    log_likelihood_ratio = parameters.delta0

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
            pre_state = torch.stack((parameters.w.new_tensor(1.0), zero, zero))
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

    return M2KnownSwitchSessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=M2KnownSwitchPredictions(
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


def evaluate_m2_known_switch_mouse_objective(
    sessions: Iterable[M2KnownSwitchSessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M2KnownSwitchObjectiveWeighting = "pooled_trials",
    required_split_role: M2KnownSwitchSplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M2KnownSwitchMouseObjectiveResult:
    r"""Evaluate M2_known_switch's full-batch pooled choice objective.

    Purpose
    -------
    Decode one six-parameter mouse vector and independently rebuild the
    oracle-switch trajectory for every supplied session before returning one
    loss.

    Mathematical definition
    -----------------------
    .. math::

       J=\frac{\sum_d J_d}{\sum_d N_d},

    with the same optional equal-session sensitivity objective as M0/M1/M2.

    Parameter interpretation
    ------------------------
    Stable parameters are pooled across sessions; the log-odds belief resets
    to Neutral (undefined, prior 0.5) at the start of every session.

    Input units and shapes
    ----------------------
    ``raw_parameters`` has shape ``(6,)`` and session durations use seconds.

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
    ``test_m2_known_switch_pooled_objective_autograd_and_session_order_invariance``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one M2_known_switch session is required")
    if raw_parameters.shape != (6,):
        raise ValueError("M2_known_switch raw_parameters must have shape (6,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown M2_known_switch objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("An M2_known_switch pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter M2_known_switch objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M2_known_switch session must have role {required_split_role!r}")

    parameters = decode_m2_known_switch_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_m2_known_switch_session(
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
        raise ValueError("M2_known_switch objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(record.valid_trial_count == 0 for record in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M2_known_switch session")
        counts = session_nll.new_tensor([record.valid_trial_count for record in evaluations])
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M2_known_switch mouse objective is nonfinite")
    return M2KnownSwitchMouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
