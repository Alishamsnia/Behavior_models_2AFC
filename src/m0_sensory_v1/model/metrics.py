"""Frozen Part IX predictive metrics and diagnostics for M0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from .session import SessionEvaluation, SessionTrials, evaluate_mouse_objective

if TYPE_CHECKING:
    from ..registry import ModelRegistry


@dataclass(frozen=True)
class CalibrationBin:
    """One fixed-width calibration bin; empty bins retain null summaries."""

    bin_index: int
    lower_probability: float
    upper_probability: float
    valid_trial_count: int
    mean_predicted_long_probability: float | None
    observed_long_rate: float | None


@dataclass(frozen=True)
class PsychometricPoint:
    """Observed and predicted long-choice rates at one rounded duration."""

    duration_s: float
    valid_trial_count: int
    observed_long_rate: float
    predicted_long_rate: float


@dataclass(frozen=True)
class BlockPsychometricPoint:
    """Diagnostic psychometric point stratified by a supplied block label."""

    block_label: str
    duration_s: float
    valid_trial_count: int
    observed_long_rate: float
    predicted_long_rate: float


@dataclass(frozen=True)
class PerSessionMetrics:
    """Predictive summaries for one independently evaluated session."""

    mouse_id: str
    session_id: str
    split_role: str
    valid_trial_count: int
    elpd: float | None
    mean_nll_per_valid_trial: float | None
    brier_score: float | None
    classification_accuracy: float | None
    observed_long_rate: float | None
    predicted_long_rate: float | None


@dataclass(frozen=True)
class PredictionDistribution:
    """Compact distribution summary of valid long-choice probabilities."""

    valid_trial_count: int
    minimum: float
    quantile_05: float
    quantile_25: float
    median: float
    quantile_75: float
    quantile_95: float
    maximum: float
    mean: float
    population_standard_deviation: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Complete serializable Part IX metric bundle for a non-test split."""

    model_id: str
    mouse_id: str
    split_role: str
    valid_trial_count: int
    elpd: float
    mean_nll_per_valid_trial: float
    brier_score: float
    classification_accuracy: float
    observed_long_rate: float
    predicted_long_rate: float
    calibration_curve: tuple[CalibrationBin, ...]
    overall_psychometric_curve: tuple[PsychometricPoint, ...]
    per_session: tuple[PerSessionMetrics, ...]
    prediction_distribution: PredictionDistribution
    block_psychometric_curve: tuple[BlockPsychometricPoint, ...]
    block_diagnostic_status: str

    def as_dict(self) -> dict:
        """Return the complete nested metric bundle as JSON-safe primitives."""

        return asdict(self)


def _valid_numpy(
    evaluation: SessionEvaluation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract duration, observation, probability, and NLL on the valid mask."""

    valid = evaluation.predictions.valid_choice_mask
    arrays = (
        evaluation.duration_s[valid],
        evaluation.choice_long[valid],
        evaluation.predictions.probability_long[valid],
        evaluation.per_trial_negative_log_likelihood[valid],
    )
    return tuple(
        value.detach().cpu().numpy().astype(float, copy=False) for value in arrays
    )  # type: ignore[return-value]


def _summaries(
    observed: np.ndarray,
    probability: np.ndarray,
    nll: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float, float, float]:
    """Calculate ELPD, mean NLL, Brier, accuracy, and observed/predicted rates."""

    elpd = -float(np.sum(nll))
    return (
        elpd,
        float(np.mean(nll)),
        float(np.mean(np.square(probability - observed))),
        float(np.mean((probability > threshold) == observed)),
        float(np.mean(observed)),
        float(np.mean(probability)),
    )


def calculate_evaluation_metrics(
    session_evaluations: Iterable[SessionEvaluation],
    *,
    model_id: str = "M0_sensory_v1",
    calibration_bins: int = 10,
    psychometric_duration_decimals: int = 3,
    classification_threshold: float = 0.5,
    block_labels_by_session: Mapping[str, Sequence[object]] | None = None,
) -> EvaluationMetrics:
    r"""Calculate the complete registered Part IX metric set.

    Purpose
    -------
    Summarize predictive fit on training or validation decisions while keeping
    session, calibration, psychometric, and optional block diagnostics.

    Mathematical definition
    -----------------------
    For valid trials :math:`i=1,\ldots,N`, this function returns

    .. math::

       \mathrm{ELPD}=\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)],
       \quad \overline{\mathrm{NLL}}=-\mathrm{ELPD}/N,

    .. math::

       \mathrm{Brier}=N^{-1}\sum_i(p_i-y_i)^2,
       \quad \mathrm{Accuracy}=N^{-1}\sum_i
       \mathbf 1\{\mathbf 1[p_i>0.5]=y_i\}.

    Parameter interpretation
    ------------------------
    Probabilities are M0's lapse-mixed long-choice predictions. Classification
    ties at the registered threshold are classified as short.

    Input units and shapes
    ----------------------
    Each session evaluation holds equal-length one-dimensional full-timeline
    tensors. Durations are seconds; observations are 0/1 or NaN. Optional block
    sequences must match the corresponding complete session length.

    Returned quantities and shapes
    ------------------------------
    Returns one immutable bundle containing scalar metrics, fixed-width
    calibration bins, duration psychometrics, per-session values, a prediction
    distribution, and optional block-stratified psychometrics.

    Constraints
    -----------
    All evaluations must belong to one mouse and one non-test split and contain
    at least one valid decision in total. This function never evaluates a model.

    Numerical stability
    -------------------
    ELPD reuses the likelihood's already log-safe per-trial NLL. Unclamped
    diagnostic probabilities are required to be finite and inside ``[0,1]``.

    Causal restrictions
    -------------------
    Locked-test evaluations are rejected. Block labels are grouped only after
    prediction and can never affect probabilities, likelihood, or fitting.

    Important edge cases
    --------------------
    Empty calibration bins are retained with null rates. Sessions with no valid
    choices receive null per-session metrics. Missing block labels skip only the
    block diagnostic. Omissions never enter any denominator.

    Related tests
    -------------
    ``test_part9_metrics_match_manual_values_and_skip_omissions`` and
    ``test_part9_rejects_locked_test_evaluations``.
    """

    records = tuple(session_evaluations)
    if not records:
        raise ValueError("At least one session evaluation is required")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    if psychometric_duration_decimals < 0:
        raise ValueError("psychometric_duration_decimals must be nonnegative")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie in (0, 1)")
    mouse_ids = {record.mouse_id for record in records}
    roles = {record.split_role for record in records}
    if len(mouse_ids) != 1 or len(roles) != 1:
        raise ValueError("Metric evaluations must belong to one mouse and split")
    role = next(iter(roles))
    if role == "test":
        raise ValueError("Locked test evaluations cannot be summarized")

    extracted = tuple(_valid_numpy(record) for record in records)
    total_valid = sum(values[0].size for values in extracted)
    if total_valid == 0:
        raise ValueError("Metrics are undefined without valid completed choices")
    duration = np.concatenate([values[0] for values in extracted])
    observed = np.concatenate([values[1] for values in extracted])
    probability = np.concatenate([values[2] for values in extracted])
    nll = np.concatenate([values[3] for values in extracted])
    if not (
        np.all(np.isfinite(duration))
        and np.all(np.isfinite(observed))
        and np.all(np.isfinite(probability))
        and np.all(np.isfinite(nll))
        and np.all((probability >= 0.0) & (probability <= 1.0))
    ):
        raise FloatingPointError("Valid metric inputs must be finite probabilities")

    elpd, mean_nll, brier, accuracy, observed_rate, predicted_rate = _summaries(
        observed,
        probability,
        nll,
        classification_threshold,
    )

    calibration: list[CalibrationBin] = []
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    assignments = np.minimum(
        np.floor(probability * calibration_bins).astype(int),
        calibration_bins - 1,
    )
    for index in range(calibration_bins):
        selected = assignments == index
        count = int(np.sum(selected))
        calibration.append(
            CalibrationBin(
                bin_index=index,
                lower_probability=float(edges[index]),
                upper_probability=float(edges[index + 1]),
                valid_trial_count=count,
                mean_predicted_long_probability=(
                    float(np.mean(probability[selected])) if count else None
                ),
                observed_long_rate=(
                    float(np.mean(observed[selected])) if count else None
                ),
            )
        )

    rounded_duration = np.round(duration, psychometric_duration_decimals)
    psychometric: list[PsychometricPoint] = []
    for value in np.unique(rounded_duration):
        selected = rounded_duration == value
        psychometric.append(
            PsychometricPoint(
                duration_s=float(value),
                valid_trial_count=int(np.sum(selected)),
                observed_long_rate=float(np.mean(observed[selected])),
                predicted_long_rate=float(np.mean(probability[selected])),
            )
        )

    per_session: list[PerSessionMetrics] = []
    for record, (_, session_observed, session_probability, session_nll) in zip(
        records,
        extracted,
        strict=True,
    ):
        count = session_observed.size
        values = (
            _summaries(
                session_observed,
                session_probability,
                session_nll,
                classification_threshold,
            )
            if count
            else (None, None, None, None, None, None)
        )
        per_session.append(
            PerSessionMetrics(
                mouse_id=record.mouse_id,
                session_id=record.session_id,
                split_role=record.split_role,
                valid_trial_count=int(count),
                elpd=values[0],
                mean_nll_per_valid_trial=values[1],
                brier_score=values[2],
                classification_accuracy=values[3],
                observed_long_rate=values[4],
                predicted_long_rate=values[5],
            )
        )

    block_points: list[BlockPsychometricPoint] = []
    if block_labels_by_session is None:
        block_status = "not_computed_no_block_labels_supplied"
    else:
        block_rows: list[tuple[str, float, float, float]] = []
        for record, (session_duration, session_observed, session_probability, _) in zip(
            records,
            extracted,
            strict=True,
        ):
            if record.session_id not in block_labels_by_session:
                raise KeyError(f"Missing block labels for session {record.session_id}")
            labels = np.asarray(block_labels_by_session[record.session_id], dtype=object)
            if labels.shape != record.choice_long.shape:
                raise ValueError(
                    f"Block labels for {record.session_id} must match full timeline"
                )
            valid_labels = labels[
                record.predictions.valid_choice_mask.detach().cpu().numpy()
            ]
            for label, d_value, y_value, p_value in zip(
                valid_labels,
                session_duration,
                session_observed,
                session_probability,
                strict=True,
            ):
                block_rows.append(
                    (
                        str(label),
                        float(np.round(d_value, psychometric_duration_decimals)),
                        float(y_value),
                        float(p_value),
                    )
                )
        for label, d_value in sorted({(row[0], row[1]) for row in block_rows}):
            selected_rows = [
                row for row in block_rows if row[0] == label and row[1] == d_value
            ]
            block_points.append(
                BlockPsychometricPoint(
                    block_label=label,
                    duration_s=d_value,
                    valid_trial_count=len(selected_rows),
                    observed_long_rate=float(np.mean([row[2] for row in selected_rows])),
                    predicted_long_rate=float(np.mean([row[3] for row in selected_rows])),
                )
            )
        block_status = "computed_diagnostic_only_not_used_by_model"

    quantiles = np.quantile(probability, [0.05, 0.25, 0.5, 0.75, 0.95])
    distribution = PredictionDistribution(
        valid_trial_count=total_valid,
        minimum=float(np.min(probability)),
        quantile_05=float(quantiles[0]),
        quantile_25=float(quantiles[1]),
        median=float(quantiles[2]),
        quantile_75=float(quantiles[3]),
        quantile_95=float(quantiles[4]),
        maximum=float(np.max(probability)),
        mean=float(np.mean(probability)),
        population_standard_deviation=float(np.std(probability, ddof=0)),
    )
    return EvaluationMetrics(
        model_id=model_id,
        mouse_id=next(iter(mouse_ids)),
        split_role=role,
        valid_trial_count=total_valid,
        elpd=elpd,
        mean_nll_per_valid_trial=mean_nll,
        brier_score=brier,
        classification_accuracy=accuracy,
        observed_long_rate=observed_rate,
        predicted_long_rate=predicted_rate,
        calibration_curve=tuple(calibration),
        overall_psychometric_curve=tuple(psychometric),
        per_session=tuple(per_session),
        prediction_distribution=distribution,
        block_psychometric_curve=tuple(block_points),
        block_diagnostic_status=block_status,
    )


def evaluate_split_metrics(
    sessions: Iterable[SessionTrials],
    raw_parameters: torch.Tensor,
    registry: ModelRegistry,
    *,
    required_split_role: str,
    block_labels_by_session: Mapping[str, Sequence[object]] | None = None,
) -> EvaluationMetrics:
    r"""Evaluate M0 without gradients and calculate registered Part IX metrics.

    Purpose
    -------
    Join the registered session evaluator to the frozen metric configuration for
    a training or validation split.

    Mathematical definition
    -----------------------
    The model first produces :math:`p_i(\widetilde\Theta)` and

    .. math::

       \mathrm{NLL}_i=-[y_i\log p_i+(1-y_i)\log(1-p_i)],

    :func:`calculate_evaluation_metrics` then computes ELPD, mean NLL, Brier,
    accuracy, calibration, and psychometric summaries.

    Parameter interpretation
    ------------------------
    ``raw_parameters`` is one restored pooled M0 checkpoint. Block labels are
    diagnostic annotations and not fitted parameters.

    Input units and shapes
    ----------------------
    ``raw_parameters`` has shape ``(5,)``. Session durations use seconds and all
    model tensors use registered ``torch.float64`` precision.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`EvaluationMetrics` bundle for the requested split.

    Constraints
    -----------
    ``required_split_role`` must be ``train`` or ``validation``. Every supplied
    session must carry that exact role.

    Numerical stability
    -------------------
    Prediction and NLL evaluation occurs under ``torch.no_grad`` and reuses the
    registered probability-log epsilon.

    Causal restrictions
    -------------------
    Test evaluation is rejected. Metrics cannot change the checkpoint or model
    selection because no optimizer or gradient is available here.

    Important edge cases
    --------------------
    An empty split or a split with no completed decisions raises. Omission-only
    individual sessions are retained with null per-session summaries.

    Related tests
    -------------
    ``test_evaluate_split_metrics_uses_registry_and_disables_gradients``.
    """

    if required_split_role not in {"train", "validation"}:
        raise ValueError("Part IX metrics may evaluate only train or validation")
    session_tuple = tuple(sessions)
    with torch.no_grad():
        if registry.model_id == "M2_hmm_static_v1":
            from .m2_session import evaluate_m2_mouse_objective

            objective = evaluate_m2_mouse_objective(
                session_tuple,
                raw_parameters,
                registry.task_moments,
                objective_weighting=registry.objective_weighting,
                required_split_role=required_split_role,
                lapse_max=registry.lapse_max,
                parameter_epsilon=registry.parameter_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
                first_long_probability=registry.first_long_probability,
            )
        elif registry.model_id == "M1_hmm_true_v1":
            from .m1_session import evaluate_m1_mouse_objective

            objective = evaluate_m1_mouse_objective(
                session_tuple,
                raw_parameters,
                registry.task_moments,
                objective_weighting=registry.objective_weighting,
                required_split_role=required_split_role,
                lapse_max=registry.lapse_max,
                parameter_epsilon=registry.parameter_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
                first_long_probability=registry.first_long_probability,
            )
        else:
            objective = evaluate_mouse_objective(
                session_tuple,
                raw_parameters,
                registry.task_moments,
                objective_weighting=registry.objective_weighting,
                required_split_role=required_split_role,  # type: ignore[arg-type]
                lapse_max=registry.lapse_max,
                parameter_epsilon=registry.parameter_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
            )
    return calculate_evaluation_metrics(
        objective.session_evaluations,
        model_id=registry.model_id,
        calibration_bins=registry.metrics.calibration_bins,
        psychometric_duration_decimals=(
            registry.metrics.psychometric_duration_decimals
        ),
        classification_threshold=registry.metrics.classification_threshold,
        block_labels_by_session=block_labels_by_session,
    )


def save_evaluation_metrics(result: EvaluationMetrics, path: str | Path) -> None:
    """Atomically serialize all Part IX scalar metrics and diagnostics to JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
