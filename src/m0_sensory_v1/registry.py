"""Typed, integrity-checked registry for the complete M0 software contract."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .data.splits import validate_chronological_splits
from .data.task_moments import TaskMoments
from .model.bayesian import (
    category_variances,
    log_odds_coefficients,
    positive_decision_threshold,
)
from .model.initialization import InitializationSpec, validate_initialization_table
from .model.parameters import decode_raw_parameters


@dataclass(frozen=True)
class OptimizerRegistry:
    """Frozen settings consumed by the implemented Part VII optimizer."""

    name: str
    initial_learning_rate: float
    maximum_epochs: int
    gradient_clipping: float | None
    implementation_status: str


@dataclass(frozen=True)
class SchedulerRegistry:
    """Frozen settings consumed by the implemented Part VII scheduler."""

    name: str
    factor: float
    patience_epochs: int
    minimum_learning_rate: float
    monitored_value: str
    implementation_status: str


@dataclass(frozen=True)
class ConvergenceRegistry:
    """Frozen Part VII convergence tolerances and acceptance checks."""

    maximum_epochs: int
    minimum_learning_rate: float
    gradient_norm_tolerance: float
    meaningful_improvement_tolerance: float
    no_improvement_patience_epochs: int
    stable_training_window_epochs: int
    required_checks: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class MetricsRegistry:
    """Names of frozen primary, secondary, and diagnostic metrics."""

    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    diagnostics: tuple[str, ...]
    calibration_bins: int
    psychometric_duration_decimals: int
    classification_threshold: float


@dataclass(frozen=True)
class SeedRegistry:
    """Explicit random seeds reserved for fitting and simulation."""

    global_seed: int
    simulation_seed: int
    run_seed_rule: str


@dataclass(frozen=True)
class MultiStartRegistry:
    """Frozen Part VIII winner and identifiability-ridge policy."""

    winner_criterion: str
    near_equal_training_nll_tolerance: float
    identifiability_parameter_distance_threshold: float
    parameter_distance_scale: tuple[float, float, float, float, float]
    additional_starts_policy: str
    result_schema_version: str


@dataclass(frozen=True)
class SplitRegistry:
    """Frozen chronological split policy and assignment artifact."""

    version: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    rounding: str
    tie_priority: tuple[str, ...]
    assignment_path: Path
    assignment_sha256: str
    locked_test: bool


@dataclass(frozen=True)
class ModelRegistry:
    """Immutable aggregate of every Parts I-IX M0 registry responsibility."""

    model_id: str
    registry_schema_version: str
    software_version: str
    implemented_scope: str
    precision: torch.dtype
    duration_unit: str
    parameter_order: tuple[str, ...]
    prior_type: str
    prior_long: float
    lapse_max: float
    positive_root_only: bool
    objective_weighting: str
    supported_objective_weightings: tuple[str, ...]
    session_policy: str
    parameter_epsilon: float
    quadratic_tolerance: float
    discriminant_tolerance: float
    log_probability_epsilon: float
    task_moments: TaskMoments
    initializations: tuple[InitializationSpec, ...]
    initialization_table_version: str
    initialization_table_path: Path
    initialization_table_sha256: str
    split: SplitRegistry
    optimizer: OptimizerRegistry
    scheduler: SchedulerRegistry
    regularization_name: str
    regularization_coefficient: float
    seeds: SeedRegistry
    convergence: ConvergenceRegistry
    multi_start_selection: MultiStartRegistry
    metrics: MetricsRegistry
    config_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject non-object top-level values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _resolve_artifact(repository_root: Path, value: str) -> Path:
    """Resolve a registry artifact path relative to the repository root."""

    path = (repository_root / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered artifact does not exist: {path}")
    return path


def _verify_sha256(path: Path, expected: str) -> None:
    """Raise when an immutable registry artifact checksum has changed."""

    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, observed {observed}"
        )


def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    r"""Load and validate the authoritative M0 registry and frozen artifacts.

    Purpose
    -------
    Provide one immutable source of truth for model identity, parameters, task
    moments, numerical policy, initializations, split, seeds, optimizer plan,
    convergence plan, multi-start selection, metrics, and software version.

    Mathematical definition
    -----------------------
    Registry validation confirms the parameter domain

    .. math::

       w>0,\quad s_0>0,\quad 0<\ell_short<1,\quad 0<ell_long<1,

    verifies every start decodes and yields the registered unique positive
    decision threshold under :math:`\pi=0.5`, and checks the 70/15/15 split.

    Parameter interpretation
    ------------------------
    The registry supplies the frozen settings consumed by the Part VII and VIII
    implementations. Loading it never starts an optimizer.

    Input units and shapes
    ----------------------
    ``path`` points to the main JSON registry. If omitted, the repository's
    ``config/m0_sensory_v1.json`` is used. Durations and ``s0`` use seconds.

    Returned quantities and shapes
    ------------------------------
    Returns one frozen :class:`ModelRegistry` containing typed scalar records,
    an eight-element initialization tuple, and verified artifact paths.

    Constraints
    -----------
    Artifact SHA-256 values, model ID, parameter order, prior, precision,
    objective, no-regularization policy, split policy, and component names must
    match the registered M0 contract.

    Numerical stability
    -------------------
    Every initialization must map to finite ``torch.float64`` raw coordinates
    and a finite positive threshold using real task moments.

    Causal restrictions
    -------------------
    Loading the registry reads configuration and session metadata assignments
    only. It never loads behavioral trials or evaluates train/validation/test
    likelihoods.

    Important edge cases
    --------------------
    Any modified frozen table or split CSV fails its checksum. Convergence and
    winner-selection tolerances must be finite and positive.

    Related tests
    -------------
    ``test_registry_loads_all_required_part_five_fields``,
    ``test_registry_rejects_modified_initialization_artifact``, and
    ``test_every_start_has_a_valid_real_data_positive_threshold``.
    """

    if path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "m0_sensory_v1.json"
    else:
        config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Model registry does not exist: {config_path}")
    repository_root = config_path.parent.parent
    config = _read_json(config_path)

    if config.get("model_id") != "M0_sensory_v1":
        raise ValueError("Registry model_id must be M0_sensory_v1")
    if config.get("implemented_scope") != "Parts I-X":
        raise ValueError("Registry implemented_scope must be Parts I-X")
    parameter_order = tuple(config.get("parameter_order", ()))
    if parameter_order != ("w", "s0", "ell_short", "ell_long", "kappa"):
        raise ValueError(
            "Registry parameter order must be (w,s0,ell_short,ell_long,kappa)"
        )
    if config.get("precision") != "float64":
        raise ValueError("M0 registry precision must be float64")
    if config.get("duration_unit") != "seconds":
        raise ValueError("M0 duration unit must be seconds")
    if config.get("prior_type") != "fixed_neutral" or config.get("prior_long") != 0.5:
        raise ValueError("M0 requires the exact fixed-neutral prior 0.5")
    if config.get("objective_weighting") != "pooled_trials":
        raise ValueError("Primary M0 objective must be pooled_trials")
    if tuple(config.get("supported_objective_weightings", ())) != (
        "pooled_trials",
        "equal_sessions",
    ):
        raise ValueError("Supported objective weightings disagree with M0")
    if config.get("session_policy") != "independent_no_state":
        raise ValueError("M0 sessions must use the independent no-state policy")
    if config.get("positive_root_only") is not True:
        raise ValueError("M0 registry must enable the positive-root-only rule")
    for tolerance_name in (
        "epsilon",
        "quadratic_tolerance",
        "discriminant_tolerance",
        "log_probability_epsilon",
    ):
        if float(config[tolerance_name]) <= 0.0:
            raise ValueError(f"Registry tolerance {tolerance_name} must be positive")

    lapse_max = float(config["lapse_max"])
    if lapse_max != 1.0:
        raise ValueError("Registered M0 lapse domain must use lapse_max=1.0")
    task_moments = TaskMoments(**config["task_moments"])

    table_reference = config["initialization_table"]
    table_path = _resolve_artifact(repository_root, table_reference["file"])
    _verify_sha256(table_path, table_reference["sha256"])
    table = _read_json(table_path)
    if table.get("model_id") != config["model_id"]:
        raise ValueError("Initialization table model_id disagrees with registry")
    if table.get("frozen") is not True or table.get(
        "frozen_before_validation_comparisons"
    ) is not True:
        raise ValueError("Initialization table must be frozen before validation")
    if tuple(table.get("parameter_order", ())) != parameter_order:
        raise ValueError("Initialization parameter order disagrees with registry")
    if table.get("table_version") != table_reference.get("version"):
        raise ValueError("Initialization table version disagrees with registry")
    if table.get("generation_policy") != table_reference.get("generation_policy"):
        raise ValueError("Initialization generation policy disagrees with registry")
    if table.get("duration_unit") != "seconds" or float(table["lapse_max"]) != lapse_max:
        raise ValueError("Initialization units/bounds disagree with registry")
    initializations = validate_initialization_table(
        (InitializationSpec(**record) for record in table["starts"]),
        lapse_max=lapse_max,
        expected_count=int(table_reference["count"]),
    )

    split_config = config["split"]
    if split_config.get("rounding") != "largest_remainder":
        raise ValueError("M0 split rounding must be largest_remainder")
    if tuple(split_config.get("tie_priority", ())) != (
        "test",
        "validation",
        "train",
    ):
        raise ValueError("M0 split tie priority must be test, validation, train")
    if split_config.get("locked_test") is not True:
        raise ValueError("M0 test split must remain locked")
    split_path = _resolve_artifact(repository_root, split_config["assignment_file"])
    split_manifest = _read_json(split_path.with_suffix(".json"))
    split_checksum = str(split_manifest["assignment_sha256"])
    _verify_sha256(split_path, split_checksum)
    split_assignments = pd.read_csv(split_path)
    validate_chronological_splits(
        split_assignments,
        split_version=split_config["version"],
        proportions=split_config["proportions"],
    )

    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]
    convergence_config = config["convergence"]
    multi_start_config = config["multi_start_selection"]
    regularization = config["regularization"]
    if optimizer_config["name"] != "adam":
        raise ValueError("Registered optimizer must be adam")
    if float(optimizer_config["initial_learning_rate"]) <= 0.0:
        raise ValueError("Adam initial learning rate must be positive")
    if int(optimizer_config["maximum_epochs"]) <= 0:
        raise ValueError("Adam maximum epochs must be positive")
    if optimizer_config["gradient_clipping"] is not None:
        raise ValueError("Registered M0 gradient clipping must remain disabled")
    if scheduler_config["name"] != "reduce_on_plateau":
        raise ValueError("Registered scheduler must be reduce_on_plateau")
    if not 0.0 < float(scheduler_config["factor"]) < 1.0:
        raise ValueError("Scheduler factor must lie in (0, 1)")
    if int(scheduler_config["patience_epochs"]) < 0:
        raise ValueError("Scheduler patience must be nonnegative")
    if scheduler_config["monitored_value"] != "fresh_training_mean_nll":
        raise ValueError("Scheduler must monitor fresh training mean NLL")
    if regularization["name"] != "none" or float(regularization["coefficient"]) != 0.0:
        raise ValueError("Registered M0 Parts I-IX require zero regularization")
    if int(optimizer_config["maximum_epochs"]) != int(
        convergence_config["maximum_epochs"]
    ):
        raise ValueError("Optimizer and convergence maximum epochs disagree")
    if float(scheduler_config["minimum_learning_rate"]) != float(
        convergence_config["minimum_learning_rate"]
    ):
        raise ValueError("Scheduler and convergence minimum learning rates disagree")
    for name in (
        "gradient_norm_tolerance",
        "meaningful_improvement_tolerance",
    ):
        if float(convergence_config[name]) <= 0.0:
            raise ValueError(f"Convergence tolerance {name} must be positive")
    if int(convergence_config["no_improvement_patience_epochs"]) <= 0:
        raise ValueError("Convergence patience must be positive")
    if int(convergence_config["stable_training_window_epochs"]) <= 1:
        raise ValueError("Stable training window must exceed one epoch")
    expected_convergence_checks = (
        "finite_objective",
        "finite_parameters",
        "finite_gradients",
        "small_gradient_norm",
        "stable_fresh_training_nll",
        "no_parameter_transform_errors",
        "no_probability_normalization_errors",
    )
    if tuple(convergence_config["required_checks"]) != expected_convergence_checks:
        raise ValueError("Convergence check list disagrees with Part VII")
    if multi_start_config["winner_criterion"] != (
        "lowest_recalculated_best_training_mean_nll_among_converged_runs"
    ):
        raise ValueError("Multi-start winner must use recalculated training NLL")
    if multi_start_config["additional_starts_policy"] != (
        "report_need_only_never_generate_automatically"
    ):
        raise ValueError("Additional starts must never be generated automatically")
    if len(multi_start_config["parameter_distance_scale"]) != 5 or any(
        float(value) <= 0.0
        for value in multi_start_config["parameter_distance_scale"]
    ):
        raise ValueError("Multi-start parameter distance scale must have 5 positives")
    if float(multi_start_config["near_equal_training_nll_tolerance"]) <= 0.0:
        raise ValueError("Near-equal NLL tolerance must be positive")
    if float(
        multi_start_config["identifiability_parameter_distance_threshold"]
    ) <= 0.0:
        raise ValueError("Identifiability distance threshold must be positive")

    with torch.no_grad():
        prior = torch.tensor(0.5, dtype=torch.float64)
        for start in initializations:
            parameters = decode_raw_parameters(
                start.raw_tensor(lapse_max=lapse_max),
                lapse_max=lapse_max,
                epsilon=float(config["epsilon"]),
            )
            variances = category_variances(parameters, task_moments)
            coefficients = log_odds_coefficients(
                variances, task_moments, prior, parameters.log_kappa
            )
            threshold = positive_decision_threshold(
                coefficients,
                quadratic_tolerance=float(config["quadratic_tolerance"]),
                discriminant_tolerance=float(config["discriminant_tolerance"]),
            )
            if not bool(torch.isfinite(threshold) & (threshold > 0.0)):
                raise ValueError(f"Initialization {start.start_id} has invalid threshold")

    seed_config = config["random_seeds"]
    metrics_config = config["metrics"]
    if int(seed_config["global"]) < 0 or int(seed_config["simulation"]) < 0:
        raise ValueError("Registry random seeds must be nonnegative integers")
    if seed_config["run_seed_rule"] != "global_plus_start_id":
        raise ValueError("Run seed rule must be global_plus_start_id")
    if not all(metrics_config.get(section) for section in ("primary", "secondary", "diagnostics")):
        raise ValueError("Every registry metric section must be nonempty")
    if int(metrics_config["calibration_bins"]) <= 0:
        raise ValueError("Calibration bin count must be positive")
    if int(metrics_config["psychometric_duration_decimals"]) < 0:
        raise ValueError("Psychometric duration decimals must be nonnegative")
    if not 0.0 < float(metrics_config["classification_threshold"]) < 1.0:
        raise ValueError("Classification threshold must lie in (0, 1)")
    return ModelRegistry(
        model_id=config["model_id"],
        registry_schema_version=str(config["registry_schema_version"]),
        software_version=str(config["software_version"]),
        implemented_scope=str(config["implemented_scope"]),
        precision=torch.float64,
        duration_unit=str(config["duration_unit"]),
        parameter_order=parameter_order,
        prior_type=str(config["prior_type"]),
        prior_long=float(config["prior_long"]),
        lapse_max=lapse_max,
        positive_root_only=bool(config["positive_root_only"]),
        objective_weighting=str(config["objective_weighting"]),
        supported_objective_weightings=tuple(config["supported_objective_weightings"]),
        session_policy=str(config["session_policy"]),
        parameter_epsilon=float(config["epsilon"]),
        quadratic_tolerance=float(config["quadratic_tolerance"]),
        discriminant_tolerance=float(config["discriminant_tolerance"]),
        log_probability_epsilon=float(config["log_probability_epsilon"]),
        task_moments=task_moments,
        initializations=initializations,
        initialization_table_version=str(table["table_version"]),
        initialization_table_path=table_path,
        initialization_table_sha256=str(table_reference["sha256"]),
        split=SplitRegistry(
            version=str(split_config["version"]),
            train_fraction=float(split_config["proportions"]["train"]),
            validation_fraction=float(split_config["proportions"]["validation"]),
            test_fraction=float(split_config["proportions"]["test"]),
            rounding=str(split_config["rounding"]),
            tie_priority=tuple(split_config["tie_priority"]),
            assignment_path=split_path,
            assignment_sha256=split_checksum,
            locked_test=bool(split_config["locked_test"]),
        ),
        optimizer=OptimizerRegistry(
            name=str(optimizer_config["name"]),
            initial_learning_rate=float(optimizer_config["initial_learning_rate"]),
            maximum_epochs=int(optimizer_config["maximum_epochs"]),
            gradient_clipping=optimizer_config["gradient_clipping"],
            implementation_status=str(optimizer_config["implementation_status"]),
        ),
        scheduler=SchedulerRegistry(
            name=str(scheduler_config["name"]),
            factor=float(scheduler_config["factor"]),
            patience_epochs=int(scheduler_config["patience_epochs"]),
            minimum_learning_rate=float(scheduler_config["minimum_learning_rate"]),
            monitored_value=str(scheduler_config["monitored_value"]),
            implementation_status=str(scheduler_config["implementation_status"]),
        ),
        regularization_name=str(regularization["name"]),
        regularization_coefficient=float(regularization["coefficient"]),
        seeds=SeedRegistry(
            global_seed=int(seed_config["global"]),
            simulation_seed=int(seed_config["simulation"]),
            run_seed_rule=str(seed_config["run_seed_rule"]),
        ),
        convergence=ConvergenceRegistry(
            maximum_epochs=int(convergence_config["maximum_epochs"]),
            minimum_learning_rate=float(convergence_config["minimum_learning_rate"]),
            gradient_norm_tolerance=float(
                convergence_config["gradient_norm_tolerance"]
            ),
            meaningful_improvement_tolerance=float(
                convergence_config["meaningful_improvement_tolerance"]
            ),
            no_improvement_patience_epochs=int(
                convergence_config["no_improvement_patience_epochs"]
            ),
            stable_training_window_epochs=int(
                convergence_config["stable_training_window_epochs"]
            ),
            required_checks=tuple(convergence_config["required_checks"]),
            note=str(convergence_config["note"]),
        ),
        multi_start_selection=MultiStartRegistry(
            winner_criterion=str(multi_start_config["winner_criterion"]),
            near_equal_training_nll_tolerance=float(
                multi_start_config["near_equal_training_nll_tolerance"]
            ),
            identifiability_parameter_distance_threshold=float(
                multi_start_config[
                    "identifiability_parameter_distance_threshold"
                ]
            ),
            parameter_distance_scale=tuple(
                float(value)
                for value in multi_start_config["parameter_distance_scale"]
            ),
            additional_starts_policy=str(
                multi_start_config["additional_starts_policy"]
            ),
            result_schema_version=str(multi_start_config["result_schema_version"]),
        ),
        metrics=MetricsRegistry(
            primary=tuple(metrics_config["primary"]),
            secondary=tuple(metrics_config["secondary"]),
            diagnostics=tuple(metrics_config["diagnostics"]),
            calibration_bins=int(metrics_config["calibration_bins"]),
            psychometric_duration_decimals=int(
                metrics_config["psychometric_duration_decimals"]
            ),
            classification_threshold=float(
                metrics_config["classification_threshold"]
            ),
        ),
        config_path=config_path,
    )
