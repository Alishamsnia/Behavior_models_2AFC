"""Typed, integrity-checked registry for the M1 true-rate constant-hazard HMM."""

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
from .model.m1_bayesian import m1_bayesian_choice_probability
from .model.m1_initialization import (
    M1InitializationSpec,
    validate_m1_initialization_table,
)
from .model.m1_parameters import decode_m1_raw_parameters
from .registry import (
    ConvergenceRegistry,
    MetricsRegistry,
    MultiStartRegistry,
    OptimizerRegistry,
    SchedulerRegistry,
    SeedRegistry,
    SplitRegistry,
)


@dataclass(frozen=True)
class M1HMMRegistry:
    """Immutable aggregate of M1's common and hidden-state contracts."""

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
    true_rate_prototypes: tuple[float, float, float]
    first_long_probability: float
    neutral_hazard_scale: float
    task_moments: TaskMoments
    initializations: tuple[M1InitializationSpec, ...]
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


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a small subject overlay into the frozen M1 contract."""

    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path) -> dict[str, Any]:
    """Read an M1 registry, resolving at most one local subject overlay."""

    override = _read_object(path)
    parent_name = override.get("extends")
    if parent_name is None:
        return override
    if not isinstance(parent_name, str) or Path(parent_name).name != parent_name:
        raise ValueError("M1 registry extends must name a sibling JSON file")
    parent_path = path.parent / parent_name
    parent = _read_object(parent_path)
    if "extends" in parent:
        raise ValueError("Nested M1 registry overlays are not supported")
    return _deep_merge(parent, override)


def _artifact(repository_root: Path, relative: str, expected_sha256: str) -> Path:
    path = (repository_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered M1 artifact is missing: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"M1 artifact checksum mismatch for {path}: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return path


def load_m1_model_registry(path: str | Path | None = None) -> M1HMMRegistry:
    r"""Load and validate M1's complete pooled HMM software contract.

    Purpose
    -------
    Provide one immutable source of truth for the M1 model, fixed true-rate
    prototypes, constant-hazard state machine, starts, split, and fitting rules.

    Mathematical definition
    -----------------------
    Registry validation fixes

    .. math::

       \theta=(0.5,0.2,0.8),\quad \xi_1^-=(1,0,0),\quad
       h=\operatorname{sigmoid}(a_h).

    Parameter interpretation
    ------------------------
    ``first_long_probability`` is the task-design probability ``omega`` and is
    never replaced by a session's realized first biased block.

    Input units and shapes
    ----------------------
    ``path`` is an M1 JSON registry; the MC08 registry is used by default.

    Returned quantities and shapes
    ------------------------------
    Returns one frozen :class:`M1HMMRegistry` with eight initialization records.

    Constraints
    -----------
    Checksums, split policy, parameter order, state order, prototypes, causal
    flags, numerical tolerances, and every start must match M1's contract.

    Numerical stability
    -------------------
    Every start must decode in float64 and yield finite probabilities at each
    of the three fixed prototype priors under the full decision-region solver.

    Causal restrictions
    -------------------
    Loading reads configuration and session assignments only; no behavioral
    trial or true block sequence is opened.

    Important edge cases
    --------------------
    Modified frozen starts or split assignments fail before fitting.

    Related tests
    -------------
    ``test_m1_registry_loads_and_validates_all_starts``.
    """

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "m1_hmm_true_v1.json"
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"M1 registry does not exist: {config_path}")
    repository_root = config_path.parent.parent
    config = _read_config(config_path)
    if config.get("model_id") != "M1_hmm_true_v1":
        raise ValueError("M1 registry model_id must be M1_hmm_true_v1")
    if tuple(config.get("parameter_order", ())) != (
        "w", "s0", "ell_short", "ell_long", "kappa", "ah"
    ):
        raise ValueError("M1 parameter order must include kappa before ah")
    if config.get("precision") != "float64" or config.get("duration_unit") != "seconds":
        raise ValueError("M1 requires float64 durations in seconds")
    if float(config["lapse_max"]) != 1.0:
        raise ValueError("M1 must share M0's full natural lapse domain")
    common = config["common_readout"]
    if common.get("decision_region") != "full_quadratic_log_odds_region":
        raise ValueError("M1 must use the full quadratic log-odds region")
    if common.get("negative_root_policy") != "integrate_all_real_measurement_regions":
        raise ValueError("M1 must integrate every long-choice measurement region")
    hmm = config["hmm"]
    if tuple(hmm["state_order"]) != ("N", "S", "L"):
        raise ValueError("M1 HMM state order must be (N,S,L)")
    prototypes = tuple(float(value) for value in hmm["true_rate_prototypes"])
    if prototypes != (0.5, 0.2, 0.8):
        raise ValueError("M1 true-rate prototypes must be exactly (0.5,0.2,0.8)")
    if tuple(float(x) for x in hmm["initial_pre_choice_belief"]) != (1.0, 0.0, 0.0):
        raise ValueError("M1 sessions must reset to a one-hot neutral belief")
    if any(
        hmm.get(name) is not False
        for name in (
            "neutral_reentry",
            "category_emission_in_outer_objective",
            "true_block_predictive_input",
            "true_run_length_predictive_input",
        )
    ):
        raise ValueError("M1 HMM causal/reentry flags disagree with the model")
    omega = float(hmm["first_long_probability"])
    if not 0.0 <= omega <= 1.0:
        raise ValueError("M1 first-long design probability must lie in [0,1]")
    neutral_hazard_scale = float(hmm.get("neutral_hazard_scale", -1.0))
    if neutral_hazard_scale != 0.5:
        raise ValueError("M1 neutral hazard must be fixed at half the biased hazard")

    table_ref = config["initialization_table"]
    table_path = _artifact(
        repository_root, table_ref["file"], str(table_ref["sha256"])
    )
    table = _read_object(table_path)
    if table.get("model_id") != config["model_id"] or table.get("frozen") is not True:
        raise ValueError("M1 initialization table identity/frozen flag is invalid")
    if table.get("table_version") != table_ref["version"]:
        raise ValueError("M1 initialization table version mismatch")
    if table.get("generation_policy") != table_ref["generation_policy"]:
        raise ValueError("M1 initialization generation policy mismatch")
    initializations = validate_m1_initialization_table(
        (M1InitializationSpec(**record) for record in table["starts"]),
        lapse_max=float(config["lapse_max"]),
        expected_count=int(table_ref["count"]),
    )
    m0_source = table.get("m0_source")
    if not isinstance(m0_source, dict):
        raise ValueError("M1 initialization table must declare its shared M0 source")
    m0_table_path = _artifact(
        repository_root, str(m0_source["file"]), str(m0_source["sha256"])
    )
    m0_table = _read_object(m0_table_path)
    if m0_table.get("table_version") != m0_source.get("table_version"):
        raise ValueError("M1 initialization source version disagrees with M0")
    m0_starts = sorted(m0_table.get("starts", []), key=lambda item: item["start_id"])
    if len(m0_starts) != len(initializations):
        raise ValueError("M1 and shared M0 initialization counts must match")
    for m1_start, m0_start in zip(initializations, m0_starts, strict=True):
        m1_common = (
            m1_start.w, m1_start.s0, m1_start.ell_short, m1_start.ell_long,
            m1_start.kappa,
        )
        m0_common = tuple(
            float(m0_start[name])
            for name in ("w", "s0", "ell_short", "ell_long", "kappa")
        )
        if m1_start.start_id != int(m0_start["start_id"]) or m1_common != m0_common:
            raise ValueError(
                "Every M1 start must preserve the same-ID shared M0 "
                "(w,s0,ell_short,ell_long,kappa) initialization"
            )

    split_config = config["split"]
    split_path_unverified = (repository_root / split_config["assignment_file"]).resolve()
    split_manifest = _read_object(split_path_unverified.with_suffix(".json"))
    split_sha = str(split_manifest["assignment_sha256"])
    split_path = _artifact(
        repository_root, split_config["assignment_file"], split_sha
    )
    assignments = pd.read_csv(split_path)
    validate_chronological_splits(
        assignments,
        split_version=str(split_config["version"]),
        proportions=split_config["proportions"],
    )
    if split_config.get("locked_test") is not True:
        raise ValueError("M1 test split must remain locked")

    moments = TaskMoments(**config["task_moments"])
    with torch.no_grad():
        for start in initializations:
            decoded = decode_m1_raw_parameters(
                start.raw_tensor(
                    lapse_max=float(config["lapse_max"]),
                    epsilon=float(config["epsilon"]),
                ),
                lapse_max=float(config["lapse_max"]),
                epsilon=float(config["epsilon"]),
            )
            for prior_value in prototypes:
                result = m1_bayesian_choice_probability(
                    torch.tensor(1.0, dtype=torch.float64),
                    decoded.common(),
                    moments,
                    torch.tensor(prior_value, dtype=torch.float64),
                    prior_probability_epsilon=float(config["prior_probability_epsilon"]),
                    quadratic_tolerance=float(config["quadratic_tolerance"]),
                    discriminant_tolerance=float(config["discriminant_tolerance"]),
                )
                if not bool(
                    torch.isfinite(result.probability_long)
                    & (result.probability_long >= 0.0)
                    & (result.probability_long <= 1.0)
                ):
                    raise ValueError(f"M1 start {start.start_id} has invalid readout")

    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    convergence = config["convergence"]
    multi = config["multi_start_selection"]
    metrics = config["metrics"]
    regularization = config["regularization"]
    seeds = config["random_seeds"]
    if optimizer["name"] != "adam" or optimizer["gradient_clipping"] is not None:
        raise ValueError("M1 registered base optimizer must be unclipped Adam")
    if scheduler["monitored_value"] != "fresh_training_mean_nll":
        raise ValueError("M1 scheduler must monitor fresh training NLL")
    if regularization != {"name": "none", "coefficient": 0.0}:
        raise ValueError("M1 primary registry currently requires no regularization")
    required_checks = (
        "finite_objective",
        "finite_parameters",
        "finite_gradients",
        "small_gradient_norm",
        "stable_fresh_training_nll",
        "no_parameter_transform_errors",
        "no_probability_normalization_errors",
    )
    if tuple(convergence["required_checks"]) != required_checks:
        raise ValueError("M1 convergence checks are incomplete")
    scale = tuple(float(value) for value in multi["parameter_distance_scale"])
    if len(scale) != 6 or any(value <= 0.0 for value in scale):
        raise ValueError("M1 parameter-distance scale needs six positive values")

    return M1HMMRegistry(
        model_id=str(config["model_id"]),
        scientific_alias=str(config["scientific_alias"]),
        registry_schema_version=str(config["registry_schema_version"]),
        software_version=str(config["software_version"]),
        implemented_scope=str(config["implemented_scope"]),
        precision=torch.float64,
        duration_unit=str(config["duration_unit"]),
        parameter_order=tuple(config["parameter_order"]),
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
        true_rate_prototypes=prototypes,
        first_long_probability=omega,
        neutral_hazard_scale=neutral_hazard_scale,
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
        regularization_name="none",
        regularization_coefficient=0.0,
        seeds=SeedRegistry(
            global_seed=int(seeds["global"]),
            simulation_seed=int(seeds["simulation"]),
            run_seed_rule=str(seeds["run_seed_rule"]),
        ),
        convergence=ConvergenceRegistry(
            maximum_epochs=int(convergence["maximum_epochs"]),
            minimum_learning_rate=float(convergence["minimum_learning_rate"]),
            gradient_norm_tolerance=float(convergence["gradient_norm_tolerance"]),
            meaningful_improvement_tolerance=float(
                convergence["meaningful_improvement_tolerance"]
            ),
            no_improvement_patience_epochs=int(
                convergence["no_improvement_patience_epochs"]
            ),
            stable_training_window_epochs=int(
                convergence["stable_training_window_epochs"]
            ),
            required_checks=required_checks,
            note=str(convergence["note"]),
        ),
        multi_start_selection=MultiStartRegistry(
            winner_criterion=str(multi["winner_criterion"]),
            near_equal_training_nll_tolerance=float(
                multi["near_equal_training_nll_tolerance"]
            ),
            identifiability_parameter_distance_threshold=float(
                multi["identifiability_parameter_distance_threshold"]
            ),
            parameter_distance_scale=scale,
            additional_starts_policy=str(multi["additional_starts_policy"]),
            result_schema_version=str(multi["result_schema_version"]),
        ),
        metrics=MetricsRegistry(
            primary=tuple(metrics["primary"]),
            secondary=tuple(metrics["secondary"]),
            diagnostics=tuple(metrics["diagnostics"]),
            calibration_bins=int(metrics["calibration_bins"]),
            psychometric_duration_decimals=int(
                metrics["psychometric_duration_decimals"]
            ),
            classification_threshold=float(metrics["classification_threshold"]),
        ),
        config_path=config_path,
    )
