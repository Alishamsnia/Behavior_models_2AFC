"""Typed, integrity-checked registry for the additive M3 delta learner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from m0_sensory_v1.data.splits import validate_chronological_splits
from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.registry import (
    ConvergenceRegistry,
    MetricsRegistry,
    OptimizerRegistry,
    SchedulerRegistry,
    SeedRegistry,
    SplitRegistry,
)

from .initialization import M3InitializationSpec, validate_m3_initialization_table


@dataclass(frozen=True)
class M3MultiStartRegistry:
    winner_criterion: str
    near_equal_training_nll_tolerance: float
    identifiability_parameter_distance_threshold: float
    parameter_distance_scale: tuple[float, ...]
    additional_starts_policy: str
    result_schema_version: str


@dataclass(frozen=True)
class M3FittingRegistry:
    default_optimizer: str
    supported_optimizers: tuple[str, ...]
    pdf_recommended_optimizer: str
    lbfgs_maximum_steps: int
    lbfgs_learning_rate: float
    lbfgs_history_size: int
    lbfgs_maximum_evaluations_per_step: int
    lbfgs_tolerance_grad: float
    lbfgs_tolerance_change: float
    lbfgs_line_search_fn: str
    optional_adam_warmup_steps: int
    optional_adam_learning_rate: float
    fresh_final_objective_required: bool
    training_only_winner_selection: bool


@dataclass(frozen=True)
class M3DeltaRegistry:
    """Complete immutable scientific and optimization contract for M3."""

    model_id: str
    scientific_alias: str
    registry_schema_version: str
    software_version: str
    implemented_scope: str
    precision: torch.dtype
    duration_unit: str
    parameter_order: tuple[str, ...]
    lapse_max: float
    objective_weighting: str
    supported_objective_weightings: tuple[str, ...]
    session_policy: str
    parameter_epsilon: float
    quadratic_tolerance: float
    discriminant_tolerance: float
    prior_probability_epsilon: float
    log_probability_epsilon: float
    state_order: tuple[str, str, str]
    first_long_probability: float
    neutral_hazard_scale: float
    delta_min: float
    maximum_accepted_projection_fraction: float
    empirical_median_block_length_trials: int
    empirical_inverse_median_hazard: float
    task_moments: TaskMoments
    initializations: tuple[M3InitializationSpec, ...]
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
    fitting: M3FittingRegistry
    multi_start_selection: M3MultiStartRegistry
    metrics: MetricsRegistry
    config_path: Path


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path, *, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve M3 -> M2 inheritance and one subject overlay without mutation."""

    resolved = path.resolve()
    if resolved in seen:
        raise ValueError("Circular model registry inheritance")
    value = _read_object(resolved)
    parent_name = value.get("extends")
    if parent_name is None:
        return value
    if not isinstance(parent_name, str) or Path(parent_name).name != parent_name:
        raise ValueError("Registry extends must name a sibling JSON file")
    parent = _read_config(resolved.parent / parent_name, seen=(*seen, resolved))
    return _deep_merge(parent, value)


def _artifact(repository_root: Path, relative: str, expected_sha256: str) -> Path:
    path = (repository_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered M3 artifact is missing: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"M3 artifact checksum mismatch for {path}: expected {expected_sha256}, "
            f"observed {observed}"
        )
    return path


def load_m3_model_registry(path: str | Path | None = None) -> M3DeltaRegistry:
    """Load and validate the complete M3 constant-step learner contract."""

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "m3_hmm_delta_v1.json"
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"M3 registry does not exist: {config_path}")
    repository_root = config_path.parent.parent
    config = _read_config(config_path)
    if config.get("model_id") != "M3_hmm_delta_v1":
        raise ValueError("M3 registry model_id must be M3_hmm_delta_v1")
    parameter_order = tuple(config.get("parameter_order", ()))
    if parameter_order != (
        "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "alpha_theta"
    ):
        raise ValueError("M3 parameter order is invalid")
    if config.get("precision") != "float64" or config.get("duration_unit") != "seconds":
        raise ValueError("M3 requires float64 durations in seconds")
    if float(config["lapse_max"]) != 1.0:
        raise ValueError("M3 must preserve the existing full lapse domain")
    common = config["common_readout"]
    if common.get("decision_region") != "full_quadratic_log_odds_region":
        raise ValueError("M3 must preserve M2's full quadratic readout")
    hmm = config["hmm"]
    if tuple(hmm["state_order"]) != ("N", "S", "L"):
        raise ValueError("M3 state order must be (N,S,L)")
    if hmm.get("prototype_learning") is not True:
        raise ValueError("M3 must enable prototype learning")
    if hmm.get("learning_rule") != "responsibility_weighted_constant_step_in_delta_coordinates":
        raise ValueError("M3 learning rule is invalid")
    if hmm.get("within_trial_order") != (
        "predict_then_emission_update_then_delta_update_then_transition"
    ):
        raise ValueError("M3 within-trial order is invalid")
    neutral_hazard_scale = float(hmm.get("neutral_hazard_scale", -1.0))
    if neutral_hazard_scale != 0.5:
        raise ValueError("M3 neutral hazard must be fixed at half the biased hazard")
    if hmm.get("prototype_use") != (
        "same_preupdate_subjective_prototypes_in_choice_prior_and_category_emission"
    ):
        raise ValueError("M3 must use one pre-update prototype tensor consistently")
    delta_min = float(hmm["delta_min"])
    projection_limit = float(hmm["maximum_accepted_projection_fraction"])
    if delta_min <= 0.0 or not 0.0 <= projection_limit <= 1.0:
        raise ValueError("M3 delta projection settings are invalid")
    block_reference = config["empirical_block_length_reference"]
    median_block_length = int(block_reference["median_trials"])
    inverse_median_hazard = float(block_reference["inverse_median_hazard"])
    if median_block_length != 23:
        raise ValueError("M3 empirical median block length must be 23 trials")
    if abs(inverse_median_hazard - 1.0 / median_block_length) > 1e-15:
        raise ValueError("M3 inverse-median hazard reference is inconsistent")
    if block_reference.get("fitted_hazard_remains_free") is not True:
        raise ValueError("M3 empirical block length must not fix the fitted hazard")

    table_ref = config["initialization_table"]
    table_path = _artifact(repository_root, table_ref["file"], table_ref["sha256"])
    table = _read_object(table_path)
    if table.get("model_id") != config["model_id"] or table.get("frozen") is not True:
        raise ValueError("M3 initialization identity/frozen flag is invalid")
    if table.get("table_version") != table_ref["version"]:
        raise ValueError("M3 initialization table version mismatch")
    initializations = validate_m3_initialization_table(
        (M3InitializationSpec(**item) for item in table["starts"]),
        lapse_max=float(config["lapse_max"]),
        expected_count=int(table_ref["count"]),
    )
    m2_ref = table["m2_source"]
    m2_path = _artifact(repository_root, m2_ref["file"], m2_ref["sha256"])
    m2_table = _read_object(m2_path)
    if m2_table.get("table_version") != m2_ref["table_version"]:
        raise ValueError("M3 source M2 table version mismatch")
    m2_by_id = {int(item["start_id"]): item for item in m2_table["starts"]}
    for start in initializations:
        source = m2_by_id[start.source_m2_start_id]
        expected = tuple(float(source[name]) for name in (
            "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0"
        ))
        observed = (
            start.w, start.s0, start.ell_short, start.ell_long, start.kappa,
            start.ah, start.delta0,
        )
        if observed != expected:
            raise ValueError("Every M3 start must preserve its declared M2 anchor")

    split_config = config["split"]
    split_unverified = (repository_root / split_config["assignment_file"]).resolve()
    split_manifest = _read_object(split_unverified.with_suffix(".json"))
    split_sha = str(split_manifest["assignment_sha256"])
    split_path = _artifact(repository_root, split_config["assignment_file"], split_sha)
    validate_chronological_splits(
        pd.read_csv(split_path),
        split_version=str(split_config["version"]),
        proportions=split_config["proportions"],
    )
    if split_config.get("locked_test") is not True:
        raise ValueError("M3 test split must remain locked")

    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    convergence = config["convergence"]
    multi = config["multi_start_selection"]
    fitting = config["m3_fitting"]
    metrics = config["metrics"]
    seeds = config["random_seeds"]
    required_checks = tuple(convergence["required_checks"])
    scale = tuple(float(value) for value in multi["parameter_distance_scale"])
    if len(scale) != 8 or any(value <= 0.0 for value in scale):
        raise ValueError("M3 parameter-distance scale needs eight positive values")
    if fitting["default_optimizer"] != "adam":
        raise ValueError("M3 default optimizer must match M2's registered Adam path")
    if tuple(fitting["supported_optimizers"]) != ("adam", "lbfgs", "hybrid"):
        raise ValueError("M3 must expose Adam, pure L-BFGS, and hybrid strategies")
    if fitting["pdf_recommended_optimizer"] != "full_batch_lbfgs":
        raise ValueError("M3 must record the PDF's full-batch L-BFGS recommendation")
    if fitting["lbfgs_line_search_fn"] != "strong_wolfe":
        raise ValueError("M3 L-BFGS must use strong-Wolfe line search")
    if int(fitting["lbfgs_maximum_steps"]) <= 0:
        raise ValueError("M3 L-BFGS step budget must be positive")
    moments = TaskMoments(**config["task_moments"])
    return M3DeltaRegistry(
        model_id=str(config["model_id"]),
        scientific_alias=str(config["scientific_alias"]),
        registry_schema_version=str(config["registry_schema_version"]),
        software_version=str(config["software_version"]),
        implemented_scope=str(config["implemented_scope"]),
        precision=torch.float64,
        duration_unit=str(config["duration_unit"]),
        parameter_order=parameter_order,
        lapse_max=float(config["lapse_max"]),
        objective_weighting=str(config["objective_weighting"]),
        supported_objective_weightings=tuple(config["supported_objective_weightings"]),
        session_policy=str(config["session_policy"]),
        parameter_epsilon=float(config["epsilon"]),
        quadratic_tolerance=float(config["quadratic_tolerance"]),
        discriminant_tolerance=float(config["discriminant_tolerance"]),
        prior_probability_epsilon=float(config["prior_probability_epsilon"]),
        log_probability_epsilon=float(config["log_probability_epsilon"]),
        state_order=("N", "S", "L"),
        first_long_probability=float(hmm["first_long_probability"]),
        neutral_hazard_scale=neutral_hazard_scale,
        delta_min=delta_min,
        maximum_accepted_projection_fraction=projection_limit,
        empirical_median_block_length_trials=median_block_length,
        empirical_inverse_median_hazard=inverse_median_hazard,
        task_moments=moments,
        initializations=initializations,
        initialization_table_version=str(table["table_version"]),
        initialization_table_path=table_path,
        initialization_table_sha256=str(table_ref["sha256"]),
        split=SplitRegistry(
            version=str(split_config["version"]),
            train_fraction=float(split_config["proportions"]["train"]),
            validation_fraction=float(split_config["proportions"]["validation"]),
            test_fraction=float(split_config["proportions"]["test"]),
            rounding=str(split_config["rounding"]),
            tie_priority=tuple(split_config["tie_priority"]),
            assignment_path=split_path,
            assignment_sha256=split_sha,
            locked_test=True,
        ),
        optimizer=OptimizerRegistry(
            name=str(optimizer["name"]),
            initial_learning_rate=float(optimizer["initial_learning_rate"]),
            maximum_epochs=int(optimizer["maximum_epochs"]),
            gradient_clipping=optimizer["gradient_clipping"],
            implementation_status=str(optimizer["implementation_status"]),
        ),
        scheduler=SchedulerRegistry(
            name=str(scheduler["name"]),
            factor=float(scheduler["factor"]),
            patience_epochs=int(scheduler["patience_epochs"]),
            minimum_learning_rate=float(scheduler["minimum_learning_rate"]),
            monitored_value=str(scheduler["monitored_value"]),
            implementation_status=str(scheduler["implementation_status"]),
        ),
        regularization_name=str(config["regularization"]["name"]),
        regularization_coefficient=float(config["regularization"]["coefficient"]),
        seeds=SeedRegistry(
            global_seed=int(seeds["global"]),
            simulation_seed=int(seeds["simulation"]),
            run_seed_rule=str(seeds["run_seed_rule"]),
        ),
        convergence=ConvergenceRegistry(
            maximum_epochs=int(convergence["maximum_epochs"]),
            minimum_learning_rate=float(convergence["minimum_learning_rate"]),
            gradient_norm_tolerance=float(convergence["gradient_norm_tolerance"]),
            meaningful_improvement_tolerance=float(convergence["meaningful_improvement_tolerance"]),
            no_improvement_patience_epochs=int(convergence["no_improvement_patience_epochs"]),
            stable_training_window_epochs=int(convergence["stable_training_window_epochs"]),
            required_checks=required_checks,
            note=str(convergence["note"]),
        ),
        fitting=M3FittingRegistry(
            default_optimizer=str(fitting["default_optimizer"]),
            supported_optimizers=tuple(fitting["supported_optimizers"]),
            pdf_recommended_optimizer=str(fitting["pdf_recommended_optimizer"]),
            lbfgs_maximum_steps=int(fitting["lbfgs_maximum_steps"]),
            lbfgs_learning_rate=float(fitting["lbfgs_learning_rate"]),
            lbfgs_history_size=int(fitting["lbfgs_history_size"]),
            lbfgs_maximum_evaluations_per_step=int(
                fitting["lbfgs_maximum_evaluations_per_step"]
            ),
            lbfgs_tolerance_grad=float(fitting["lbfgs_tolerance_grad"]),
            lbfgs_tolerance_change=float(fitting["lbfgs_tolerance_change"]),
            lbfgs_line_search_fn=str(fitting["lbfgs_line_search_fn"]),
            optional_adam_warmup_steps=int(fitting["optional_adam_warmup_steps"]),
            optional_adam_learning_rate=float(fitting["optional_adam_learning_rate"]),
            fresh_final_objective_required=bool(
                fitting["fresh_final_objective_required"]
            ),
            training_only_winner_selection=bool(
                fitting["training_only_winner_selection"]
            ),
        ),
        multi_start_selection=M3MultiStartRegistry(
            winner_criterion=str(multi["winner_criterion"]),
            near_equal_training_nll_tolerance=float(multi["near_equal_training_nll_tolerance"]),
            identifiability_parameter_distance_threshold=float(multi["identifiability_parameter_distance_threshold"]),
            parameter_distance_scale=scale,
            additional_starts_policy=str(multi["additional_starts_policy"]),
            result_schema_version=str(multi["result_schema_version"]),
        ),
        metrics=MetricsRegistry(
            primary=tuple(metrics["primary"]),
            secondary=tuple(metrics["secondary"]),
            diagnostics=tuple(metrics["diagnostics"]),
            calibration_bins=int(metrics["calibration_bins"]),
            psychometric_duration_decimals=int(metrics["psychometric_duration_decimals"]),
            classification_threshold=float(metrics["classification_threshold"]),
        ),
        config_path=config_path,
    )
