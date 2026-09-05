"""Typed registry for M2_dynamic_block_length_v1.

Proxies genuinely shared fields from the TRUE PARENT's own registry object
(``load_m2_model_registry``, unmodified) via read-only forwarding properties
-- the same pattern ``m4_hsmm_delta_v1/registry.py`` uses for its
``source_m3_registry``, applied here to M2 instead. There is no reference to
``m4_hsmm_delta_v1`` anywhere in this file. ``convergence`` is NOT forwarded:
this registry declares its own required-checks contract (see below), since
this model's convergence surface differs from M2's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from m0_sensory_v1.m2_registry import M2StaticRegistry, load_m2_model_registry
from m0_sensory_v1.registry import ConvergenceRegistry, MetricsRegistry

from .initialization import (
    M2DynamicBlockLengthInitializationSpec,
    validate_m2_dynamic_block_length_initialization_table,
)


@dataclass(frozen=True)
class M2DynamicBlockLengthFittingRegistry:
    """Own (not proxied) L-BFGS/hybrid defaults, matching M3/M4's ``m3_fitting`` shape."""

    lbfgs_maximum_steps: int
    lbfgs_learning_rate: float
    lbfgs_history_size: int
    lbfgs_maximum_evaluations_per_step: int
    lbfgs_tolerance_grad: float
    lbfgs_tolerance_change: float
    lbfgs_line_search_fn: str
    optional_adam_warmup_steps: int
    optional_adam_learning_rate: float


@dataclass(frozen=True)
class M2DynamicBlockLengthMultiStartRegistry:
    winner_criterion: str
    near_equal_training_nll_tolerance: float
    identifiability_parameter_distance_threshold: float
    parameter_distance_scale: tuple[float, ...]
    additional_starts_policy: str
    result_schema_version: str


@dataclass(frozen=True)
class M2DynamicBlockLengthRegistry:
    """M2_dynamic_block_length's own contract plus the immutable parent M2 contract."""

    model_id: str
    scientific_alias: str
    registry_schema_version: str
    software_version: str
    implemented_scope: str
    parameter_order: tuple[str, ...]
    session_policy: str
    run_length_center: int
    maximum_represented_run_length: int
    tail_state_semantics: str
    tail_hazard_policy: str
    tail_occupancy_statistic: str
    maximum_accepted_mean_tail_probability: float
    source_m2_registry: M2StaticRegistry
    initializations: tuple[M2DynamicBlockLengthInitializationSpec, ...]
    initialization_table_version: str
    initialization_table_path: Path
    initialization_table_sha256: str
    convergence: ConvergenceRegistry
    fitting: M2DynamicBlockLengthFittingRegistry
    multi_start_selection: M2DynamicBlockLengthMultiStartRegistry
    metrics: MetricsRegistry
    config_path: Path

    @property
    def precision(self): return self.source_m2_registry.precision
    @property
    def duration_unit(self): return self.source_m2_registry.duration_unit
    @property
    def lapse_max(self): return self.source_m2_registry.lapse_max
    @property
    def objective_weighting(self): return self.source_m2_registry.objective_weighting
    @property
    def parameter_epsilon(self): return self.source_m2_registry.parameter_epsilon
    @property
    def quadratic_tolerance(self): return self.source_m2_registry.quadratic_tolerance
    @property
    def discriminant_tolerance(self): return self.source_m2_registry.discriminant_tolerance
    @property
    def prior_probability_epsilon(self): return self.source_m2_registry.prior_probability_epsilon
    @property
    def log_probability_epsilon(self): return self.source_m2_registry.log_probability_epsilon
    @property
    def state_order(self): return self.source_m2_registry.state_order
    @property
    def prototype_parameterization(self): return self.source_m2_registry.prototype_parameterization
    @property
    def first_long_probability(self): return self.source_m2_registry.first_long_probability
    @property
    def neutral_hazard_scale(self): return self.source_m2_registry.neutral_hazard_scale
    @property
    def empirical_median_block_length_trials(self): return self.run_length_center
    @property
    def task_moments(self): return self.source_m2_registry.task_moments
    @property
    def split(self): return self.source_m2_registry.split
    @property
    def optimizer(self): return self.source_m2_registry.optimizer
    @property
    def scheduler(self): return self.source_m2_registry.scheduler
    @property
    def seeds(self): return self.source_m2_registry.seeds


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_config(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError("Circular M2_dynamic_block_length registry inheritance")
    value = _read(path)
    parent = value.get("extends")
    if parent is None:
        return value
    if not isinstance(parent, str) or Path(parent).name != parent:
        raise ValueError("M2_dynamic_block_length extends must name a sibling registry")
    return _merge(_read_config(path.parent / parent, (*seen, path)), value)


def _artifact(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered M2_dynamic_block_length artifact is missing: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(
            f"M2_dynamic_block_length artifact checksum mismatch: expected {expected}, observed {observed}"
        )
    return path


def load_m2_dynamic_block_length_model_registry(
    path: str | Path | None = None,
) -> M2DynamicBlockLengthRegistry:
    r"""Load and validate M2_dynamic_block_length's complete software contract.

    Purpose
    -------
    Provide one immutable source of truth for M2's fitted static prototypes
    plus a duration-dependent hazard slope, proxying only genuinely shared
    fields from the real M2 registry.

    Mathematical definition
    -----------------------
    .. math::

       \theta(\delta_0)=(0.5,\sigma(-\delta_0),\sigma(\delta_0)),\qquad
       h_B(r)=\sigma(a_h+b_h(r-23)).

    Parameter interpretation
    ------------------------
    ``source_m2_registry`` supplies every genuinely shared field (task
    moments, split, optimizer, scheduler, seeds, epsilons); this registry
    declares its own ``hsmm`` contract, ``parameter_order``, initializations,
    and convergence checks. ``delta0`` keeps M2's own domain -- no artificial
    floor is imposed, since there is no delta-update rule here to protect.

    Input units and shapes
    ----------------------
    ``path`` names a base or one-level subject-overlay JSON registry.

    Returned quantities and shapes
    ------------------------------
    Returns a frozen registry with eight starts and a proxied task/split
    contract.

    Constraints
    -----------
    Identity, parameter order, HSMM contract, checksums, and starts are
    validated before use; every start must preserve its M2 source row exactly.

    Numerical stability
    -------------------
    Every start is decoded at load time and checked for finiteness.

    Causal restrictions
    -------------------
    Registry loading opens configuration and split assignments, never behavior.

    Important edge cases
    --------------------
    Altered frozen artifacts, nested overlays, or a mismatched M2 source row raise.

    Related tests
    -------------
    ``test_m2_dynamic_block_length_registry_preserves_m2_starts_and_adds_bh``.
    """

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "m2_dynamic_block_length_v1.json"
        if path is None else Path(path).expanduser().resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"M2_dynamic_block_length registry does not exist: {config_path}")
    config = _read_config(config_path)
    if config.get("model_id") != "M2_dynamic_block_length_v1":
        raise ValueError("M2_dynamic_block_length registry model_id is invalid")
    order = tuple(config.get("parameter_order", ()))
    expected_order = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh")
    if order != expected_order:
        raise ValueError("M2_dynamic_block_length parameter order must append bh to the M2 order")

    root = config_path.parent.parent
    source_reference = config["source_m2_registry"]
    if not isinstance(source_reference, dict):
        raise TypeError("M2_dynamic_block_length source_m2_registry must be a checksummed artifact reference")
    source_path = _artifact(root, str(source_reference["file"]), str(source_reference["sha256"]))
    source = load_m2_model_registry(source_path)

    hsmm = config["hsmm"]
    center = int(hsmm["run_length_center"])
    cap = int(hsmm["maximum_represented_run_length"])
    if center != 23 or cap != 65:
        raise ValueError("Registered M2_dynamic_block_length requires run-length center 23 and cap 65")
    if hsmm.get("tail_state_semantics") != "r65_represents_R_greater_than_or_equal_to_65":
        raise ValueError("M2_dynamic_block_length requires the registered capped-tail interpretation")
    if hsmm.get("tail_hazard_policy") != "hazard_saturates_at_h65_and_survival_remains_in_tail":
        raise ValueError("M2_dynamic_block_length tail hazard policy is invalid")
    if hsmm.get("common_hazard_across_neutral_and_biased_states") is not False:
        raise ValueError("M2_dynamic_block_length must use state-specific tied neutral/biased hazards")
    if float(hsmm.get("neutral_hazard_scale", -1.0)) != 0.5:
        raise ValueError("M2_dynamic_block_length neutral hazard must be fixed at half the biased hazard")
    if hsmm.get("true_block_predictive_input") is not False or hsmm.get("true_run_length_predictive_input") is not False:
        raise ValueError("True block/run length cannot enter M2_dynamic_block_length predictions")
    if hsmm.get("prototype_learning") is not False:
        raise ValueError("M2_dynamic_block_length prototypes must remain static (no learning)")

    tail_acceptance = config["tail_acceptance"]
    tail_statistic = str(tail_acceptance["statistic"])
    maximum_tail = float(tail_acceptance["maximum_accepted_value"])
    if tail_statistic != "mean_pre_choice_tail_probability_across_all_training_timeline_trials":
        raise ValueError("M2_dynamic_block_length tail acceptance statistic is invalid")
    if not 0.0 <= maximum_tail <= 1.0:
        raise ValueError("M2_dynamic_block_length maximum accepted tail probability must lie in [0,1]")
    if tail_acceptance.get("required_for_convergence_and_winner_eligibility") is not True:
        raise ValueError("M2_dynamic_block_length tail occupancy must gate convergence and winner eligibility")

    reference = config["initialization_table"]
    table_path = _artifact(root, reference["file"], reference["sha256"])
    table = _read(table_path)
    if table.get("table_version") != reference["version"] or table.get("frozen") is not True:
        raise ValueError("M2_dynamic_block_length initialization version/frozen flag is invalid")
    starts = validate_m2_dynamic_block_length_initialization_table(
        (M2DynamicBlockLengthInitializationSpec(**item) for item in table["starts"]),
        lapse_max=source.lapse_max,
        expected_count=int(reference["count"]),
    )
    m2_reference = table["m2_source"]
    m2_table_path = _artifact(root, m2_reference["file"], m2_reference["sha256"])
    m2_table = _read(m2_table_path)
    if m2_table.get("table_version") != m2_reference["table_version"]:
        raise ValueError("M2_dynamic_block_length source M2 table version mismatch")
    m2_by_id = {int(item["start_id"]): item for item in m2_table["starts"]}
    names = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0")
    for start in starts:
        source_row = m2_by_id[start.source_m2_start_id]
        if tuple(getattr(start, name) for name in names) != tuple(float(source_row[name]) for name in names):
            raise ValueError("Every M2_dynamic_block_length start must preserve its M2 source row exactly")

    convergence_config = config["convergence"]
    required_checks = (
        "finite_objective", "finite_parameters", "finite_gradients", "small_gradient_norm",
        "stable_fresh_training_nll", "no_parameter_transform_errors", "no_probability_normalization_errors",
        "no_optimizer_runtime_errors", "tail_occupancy_acceptable",
    )
    if tuple(convergence_config["required_checks"]) != required_checks:
        raise ValueError("M2_dynamic_block_length convergence checks are incomplete")

    multi = config["multi_start_selection"]
    metrics_config = config["metrics"]
    inherited_metrics = source.metrics
    scale = tuple(float(value) for value in multi["parameter_distance_scale"])
    if len(scale) != 8 or any(value <= 0.0 for value in scale):
        raise ValueError("M2_dynamic_block_length parameter-distance scale requires eight positive values")

    fitting_config = config["fitting"]
    if fitting_config["lbfgs_line_search_fn"] != "strong_wolfe":
        raise ValueError("M2_dynamic_block_length L-BFGS must use strong_wolfe line search")
    if int(fitting_config["lbfgs_maximum_steps"]) <= 0:
        raise ValueError("M2_dynamic_block_length L-BFGS maximum_steps must be positive")

    return M2DynamicBlockLengthRegistry(
        model_id="M2_dynamic_block_length_v1",
        scientific_alias=str(config["scientific_alias"]),
        registry_schema_version=str(config["registry_schema_version"]),
        software_version=str(config["software_version"]),
        implemented_scope=str(config["implemented_scope"]),
        parameter_order=order,
        session_policy=str(config["session_policy"]),
        run_length_center=center,
        maximum_represented_run_length=cap,
        tail_state_semantics=str(hsmm["tail_state_semantics"]),
        tail_hazard_policy=str(hsmm["tail_hazard_policy"]),
        tail_occupancy_statistic=tail_statistic,
        maximum_accepted_mean_tail_probability=maximum_tail,
        source_m2_registry=source,
        initializations=starts,
        initialization_table_version=str(table["table_version"]),
        initialization_table_path=table_path,
        initialization_table_sha256=str(reference["sha256"]),
        convergence=ConvergenceRegistry(
            maximum_epochs=int(convergence_config["maximum_epochs"]),
            minimum_learning_rate=float(convergence_config["minimum_learning_rate"]),
            gradient_norm_tolerance=float(convergence_config["gradient_norm_tolerance"]),
            meaningful_improvement_tolerance=float(convergence_config["meaningful_improvement_tolerance"]),
            no_improvement_patience_epochs=int(convergence_config["no_improvement_patience_epochs"]),
            stable_training_window_epochs=int(convergence_config["stable_training_window_epochs"]),
            required_checks=required_checks,
            note=str(convergence_config["note"]),
        ),
        fitting=M2DynamicBlockLengthFittingRegistry(
            lbfgs_maximum_steps=int(fitting_config["lbfgs_maximum_steps"]),
            lbfgs_learning_rate=float(fitting_config["lbfgs_learning_rate"]),
            lbfgs_history_size=int(fitting_config["lbfgs_history_size"]),
            lbfgs_maximum_evaluations_per_step=int(fitting_config["lbfgs_maximum_evaluations_per_step"]),
            lbfgs_tolerance_grad=float(fitting_config["lbfgs_tolerance_grad"]),
            lbfgs_tolerance_change=float(fitting_config["lbfgs_tolerance_change"]),
            lbfgs_line_search_fn=str(fitting_config["lbfgs_line_search_fn"]),
            optional_adam_warmup_steps=int(fitting_config["optional_adam_warmup_steps"]),
            optional_adam_learning_rate=float(fitting_config["optional_adam_learning_rate"]),
        ),
        multi_start_selection=M2DynamicBlockLengthMultiStartRegistry(
            winner_criterion=str(multi["winner_criterion"]),
            near_equal_training_nll_tolerance=float(multi["near_equal_training_nll_tolerance"]),
            identifiability_parameter_distance_threshold=float(multi["identifiability_parameter_distance_threshold"]),
            parameter_distance_scale=scale,
            additional_starts_policy=str(multi["additional_starts_policy"]),
            result_schema_version=str(multi["result_schema_version"]),
        ),
        metrics=MetricsRegistry(
            primary=tuple(metrics_config["primary"]),
            secondary=tuple(metrics_config.get("secondary", inherited_metrics.secondary)),
            diagnostics=tuple(metrics_config["diagnostics"]),
            calibration_bins=int(metrics_config.get("calibration_bins", inherited_metrics.calibration_bins)),
            psychometric_duration_decimals=int(metrics_config.get(
                "psychometric_duration_decimals", inherited_metrics.psychometric_duration_decimals,
            )),
            classification_threshold=float(metrics_config.get(
                "classification_threshold", inherited_metrics.classification_threshold,
            )),
        ),
        config_path=config_path,
    )
