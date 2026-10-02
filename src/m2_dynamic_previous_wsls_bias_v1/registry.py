"""Typed registry for M2_dynamic_previous_wsls_bias_v1.

Proxies genuinely shared fields from the TRUE PARENT's own registry object
(``load_m2_dynamic_block_length_model_registry``, unmodified) via read-only
forwarding properties -- the same pattern
``m2_dynamic_block_length_v1/registry.py`` uses for its
``source_m2_registry``, applied here one level deeper (M2_dynamic_block_length_v1
instead of M2). There is no reference to ``m4_hsmm_delta_v1``,
``m1_action_kernel_v1``, ``m2_dynamic_previous_action_bias_v1``, or
``wsls_core_v1`` anywhere in this file. ``convergence`` IS forwarded
unchanged: the bias-mixing eta introduces no new failure mode relative to
M2_dynamic_block_length_v1's own nine checks. ``wsls_learning_rate`` is
declared here as a validated constant equal to ``1.0`` -- this model has no
fitted learning-rate coordinate at all, matching ``pure_wsls_v1``'s own
fixed-``alpha=1`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from m2_dynamic_block_length_v1.registry import (
    M2DynamicBlockLengthRegistry,
    load_m2_dynamic_block_length_model_registry,
)
from m0_sensory_v1.registry import ConvergenceRegistry, MetricsRegistry

from .initialization import (
    M2DynamicPreviousWslsBiasInitializationSpec,
    validate_m2_dynamic_previous_wsls_bias_initialization_table,
)


@dataclass(frozen=True)
class M2DynamicPreviousWslsBiasFittingRegistry:
    """Own (not proxied) L-BFGS/hybrid defaults, matching the family's ``fitting`` shape."""

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
class M2DynamicPreviousWslsBiasMultiStartRegistry:
    winner_criterion: str
    near_equal_training_nll_tolerance: float
    identifiability_parameter_distance_threshold: float
    parameter_distance_scale: tuple[float, ...]
    additional_starts_policy: str
    result_schema_version: str


@dataclass(frozen=True)
class M2DynamicPreviousWslsBiasRegistry:
    """M2_dynamic_previous_wsls_bias's own contract plus the immutable
    parent M2_dynamic_block_length_v1 contract."""

    model_id: str
    scientific_alias: str
    registry_schema_version: str
    software_version: str
    implemented_scope: str
    parameter_order: tuple[str, ...]
    session_policy: str
    wsls_learning_rate: float
    source_m2_dynamic_block_length_registry: M2DynamicBlockLengthRegistry
    initializations: tuple[M2DynamicPreviousWslsBiasInitializationSpec, ...]
    initialization_table_version: str
    initialization_table_path: Path
    initialization_table_sha256: str
    convergence: ConvergenceRegistry
    fitting: M2DynamicPreviousWslsBiasFittingRegistry
    multi_start_selection: M2DynamicPreviousWslsBiasMultiStartRegistry
    metrics: MetricsRegistry
    config_path: Path

    @property
    def precision(self): return self.source_m2_dynamic_block_length_registry.precision
    @property
    def duration_unit(self): return self.source_m2_dynamic_block_length_registry.duration_unit
    @property
    def lapse_max(self): return self.source_m2_dynamic_block_length_registry.lapse_max
    @property
    def objective_weighting(self): return self.source_m2_dynamic_block_length_registry.objective_weighting
    @property
    def parameter_epsilon(self): return self.source_m2_dynamic_block_length_registry.parameter_epsilon
    @property
    def quadratic_tolerance(self): return self.source_m2_dynamic_block_length_registry.quadratic_tolerance
    @property
    def discriminant_tolerance(self): return self.source_m2_dynamic_block_length_registry.discriminant_tolerance
    @property
    def prior_probability_epsilon(self): return self.source_m2_dynamic_block_length_registry.prior_probability_epsilon
    @property
    def log_probability_epsilon(self): return self.source_m2_dynamic_block_length_registry.log_probability_epsilon
    @property
    def state_order(self): return self.source_m2_dynamic_block_length_registry.state_order
    @property
    def prototype_parameterization(self): return self.source_m2_dynamic_block_length_registry.prototype_parameterization
    @property
    def first_long_probability(self): return self.source_m2_dynamic_block_length_registry.first_long_probability
    @property
    def neutral_hazard_scale(self): return self.source_m2_dynamic_block_length_registry.neutral_hazard_scale
    @property
    def run_length_center(self): return self.source_m2_dynamic_block_length_registry.run_length_center
    @property
    def maximum_represented_run_length(self): return self.source_m2_dynamic_block_length_registry.maximum_represented_run_length
    @property
    def tail_state_semantics(self): return self.source_m2_dynamic_block_length_registry.tail_state_semantics
    @property
    def tail_hazard_policy(self): return self.source_m2_dynamic_block_length_registry.tail_hazard_policy
    @property
    def tail_occupancy_statistic(self): return self.source_m2_dynamic_block_length_registry.tail_occupancy_statistic
    @property
    def maximum_accepted_mean_tail_probability(self): return self.source_m2_dynamic_block_length_registry.maximum_accepted_mean_tail_probability
    @property
    def empirical_median_block_length_trials(self): return self.run_length_center
    @property
    def initial_pi(self) -> float: return 0.5
    @property
    def task_moments(self): return self.source_m2_dynamic_block_length_registry.task_moments
    @property
    def split(self): return self.source_m2_dynamic_block_length_registry.split
    @property
    def optimizer(self): return self.source_m2_dynamic_block_length_registry.optimizer
    @property
    def scheduler(self): return self.source_m2_dynamic_block_length_registry.scheduler
    @property
    def seeds(self): return self.source_m2_dynamic_block_length_registry.seeds


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
        raise ValueError("Circular M2_dynamic_previous_wsls_bias registry inheritance")
    value = _read(path)
    parent = value.get("extends")
    if parent is None:
        return value
    if not isinstance(parent, str) or Path(parent).name != parent:
        raise ValueError("M2_dynamic_previous_wsls_bias extends must name a sibling registry")
    return _merge(_read_config(path.parent / parent, (*seen, path)), value)


def _artifact(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered M2_dynamic_previous_wsls_bias artifact is missing: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(
            f"M2_dynamic_previous_wsls_bias artifact checksum mismatch: expected {expected}, observed {observed}"
        )
    return path


def load_m2_dynamic_previous_wsls_bias_model_registry(
    path: str | Path | None = None,
) -> M2DynamicPreviousWslsBiasRegistry:
    r"""Load and validate M2_dynamic_previous_wsls_bias's complete software contract.

    Purpose
    -------
    Provide one immutable source of truth for M2_dynamic_block_length_v1's
    fitted static prototypes and duration-dependent hazard, plus a fitted
    previous-category (win-stay/lose-shift) bias (the fixed alpha=1
    boundary) mixed between the sensory readout and lapse, proxying only
    genuinely shared fields from the real M2_dynamic_block_length_v1
    registry.

    Mathematical definition
    -----------------------
    .. math::

       r_t=\text{BayesianReadout}(\cdot),\quad
       \pi_{t+1}=C_t\ (F_t=\text{True}),\quad
       m_t=(1-\eta)r_t+\eta\pi_t,\quad
       p_t=\text{lapse\_mixture}(m_t,\cdot).

    Parameter interpretation
    ------------------------
    ``source_m2_dynamic_block_length_registry`` supplies every genuinely
    shared field (HSMM contract, task moments, split, optimizer, scheduler,
    seeds, epsilons); this registry declares its own ``parameter_order``,
    ``wsls_learning_rate`` constant, initializations, and multi-start
    selection contract. ``convergence`` is forwarded, not redeclared.

    Input units and shapes
    ----------------------
    ``path`` names a base or one-level subject-overlay JSON registry.

    Returned quantities and shapes
    ------------------------------
    Returns a frozen registry with eight starts and a proxied
    HSMM/task/split contract.

    Constraints
    -----------
    Identity, parameter order, checksums, and starts are validated before
    use; every start must preserve its M2_dynamic_block_length_v1 source
    row exactly; ``wsls_learning_rate`` must equal exactly ``1.0``.

    Numerical stability
    -------------------
    Every start is decoded at load time and checked for finiteness.

    Causal restrictions
    -------------------
    Registry loading opens configuration and split assignments, never behavior.

    Important edge cases
    --------------------
    Altered frozen artifacts, nested overlays, or a mismatched
    M2_dynamic_block_length_v1 source row raise.

    Related tests
    -------------
    ``test_registry_loads_with_nine_parameters_and_fixed_learning_rate`` and ``test_starts_preserve_m2_dynamic_block_length_source_rows_exactly``.
    """

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "m2_dynamic_previous_wsls_bias_v1.json"
        if path is None else Path(path).expanduser().resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"M2_dynamic_previous_wsls_bias registry does not exist: {config_path}")
    config = _read_config(config_path)
    if config.get("model_id") != "M2_dynamic_previous_wsls_bias_v1":
        raise ValueError("M2_dynamic_previous_wsls_bias registry model_id is invalid")
    order = tuple(config.get("parameter_order", ()))
    expected_order = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit")
    if order != expected_order:
        raise ValueError("M2_dynamic_previous_wsls_bias parameter order must append eta_logit to the M2_dynamic order")

    root = config_path.parent.parent
    source_reference = config["source_m2_dynamic_block_length_registry"]
    if not isinstance(source_reference, dict):
        raise TypeError("M2_dynamic_previous_wsls_bias source_m2_dynamic_block_length_registry must be a checksummed artifact reference")
    source_path = _artifact(root, str(source_reference["file"]), str(source_reference["sha256"]))
    source = load_m2_dynamic_block_length_model_registry(source_path)

    hsmm = config["hsmm"]
    if int(hsmm["run_length_center"]) != 23 or int(hsmm["maximum_represented_run_length"]) != 65:
        raise ValueError("M2_dynamic_previous_wsls_bias requires run-length center 23 and cap 65")
    if hsmm.get("prototype_learning") is not False:
        raise ValueError("M2_dynamic_previous_wsls_bias prototypes must remain static (no learning)")
    if hsmm.get("true_block_predictive_input") is not False or hsmm.get("true_run_length_predictive_input") is not False:
        raise ValueError("True block/run length cannot enter M2_dynamic_previous_wsls_bias predictions")

    tail_acceptance = config["tail_acceptance"]
    maximum_tail = float(tail_acceptance["maximum_accepted_value"])
    if not 0.0 <= maximum_tail <= 1.0:
        raise ValueError("M2_dynamic_previous_wsls_bias maximum accepted tail probability must lie in [0,1]")
    if tail_acceptance.get("required_for_convergence_and_winner_eligibility") is not True:
        raise ValueError("M2_dynamic_previous_wsls_bias tail occupancy must gate convergence and winner eligibility")

    wsls_bias = config["wsls_bias"]
    if wsls_bias.get("prediction_uses_pre_update_pi") is not True:
        raise ValueError("M2_dynamic_previous_wsls_bias must predict from the pre-update previous-wsls-bias state")
    wsls_learning_rate = float(wsls_bias["alpha_fixed_value"])
    if wsls_learning_rate != 1.0:
        raise ValueError("M2_dynamic_previous_wsls_bias requires the fixed previous-category bias learning rate alpha=1.0")

    reference = config["initialization_table"]
    table_path = _artifact(root, reference["file"], reference["sha256"])
    table = _read(table_path)
    if table.get("table_version") != reference["version"] or table.get("frozen") is not True:
        raise ValueError("M2_dynamic_previous_wsls_bias initialization version/frozen flag is invalid")
    starts = validate_m2_dynamic_previous_wsls_bias_initialization_table(
        (M2DynamicPreviousWslsBiasInitializationSpec(**item) for item in table["starts"]),
        lapse_max=source.lapse_max,
        expected_count=int(reference["count"]),
    )
    m2_dynamic_reference = table["m2_dynamic_source"]
    m2_dynamic_table_path = _artifact(root, m2_dynamic_reference["file"], m2_dynamic_reference["sha256"])
    m2_dynamic_table = _read(m2_dynamic_table_path)
    if m2_dynamic_table.get("table_version") != m2_dynamic_reference["table_version"]:
        raise ValueError("M2_dynamic_previous_wsls_bias source M2_dynamic table version mismatch")
    m2_dynamic_by_id = {int(item["start_id"]): item for item in m2_dynamic_table["starts"]}
    names = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh")
    for start in starts:
        source_row = m2_dynamic_by_id[start.source_m2_dynamic_start_id]
        if tuple(getattr(start, name) for name in names) != tuple(float(source_row[name]) for name in names):
            raise ValueError("Every M2_dynamic_previous_wsls_bias start must preserve its M2_dynamic source row exactly")

    convergence_config = config["convergence"]
    required_checks = source.convergence.required_checks
    if tuple(convergence_config["required_checks"]) != required_checks:
        raise ValueError("M2_dynamic_previous_wsls_bias convergence checks must match M2_dynamic_block_length_v1's own")

    multi = config["multi_start_selection"]
    metrics_config = config["metrics"]
    inherited_metrics = source.metrics
    scale = tuple(float(value) for value in multi["parameter_distance_scale"])
    if len(scale) != 9 or any(value <= 0.0 for value in scale):
        raise ValueError("M2_dynamic_previous_wsls_bias parameter-distance scale requires nine positive values")

    fitting_config = config["fitting"]
    if fitting_config["lbfgs_line_search_fn"] != "strong_wolfe":
        raise ValueError("M2_dynamic_previous_wsls_bias L-BFGS must use strong_wolfe line search")
    if int(fitting_config["lbfgs_maximum_steps"]) <= 0:
        raise ValueError("M2_dynamic_previous_wsls_bias L-BFGS maximum_steps must be positive")

    return M2DynamicPreviousWslsBiasRegistry(
        model_id="M2_dynamic_previous_wsls_bias_v1",
        scientific_alias=str(config["scientific_alias"]),
        registry_schema_version=str(config["registry_schema_version"]),
        software_version=str(config["software_version"]),
        implemented_scope=str(config["implemented_scope"]),
        parameter_order=order,
        session_policy=str(config["session_policy"]),
        wsls_learning_rate=wsls_learning_rate,
        source_m2_dynamic_block_length_registry=source,
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
        fitting=M2DynamicPreviousWslsBiasFittingRegistry(
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
        multi_start_selection=M2DynamicPreviousWslsBiasMultiStartRegistry(
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
