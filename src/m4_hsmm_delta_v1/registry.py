"""Typed registry for the additive capped-tail M4 HSMM."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from m3_hmm_delta_v1.registry import M3DeltaRegistry, load_m3_model_registry
from m0_sensory_v1.registry import MetricsRegistry

from .initialization import M4InitializationSpec, validate_m4_initialization_table


@dataclass(frozen=True)
class M4MultiStartRegistry:
    winner_criterion: str
    near_equal_training_nll_tolerance: float
    identifiability_parameter_distance_threshold: float
    parameter_distance_scale: tuple[float, ...]
    additional_starts_policy: str
    result_schema_version: str


@dataclass(frozen=True)
class M4HSMMRegistry:
    """M4-specific contract plus the immutable shared M3 contract."""

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
    source_m3_registry: M3DeltaRegistry
    initializations: tuple[M4InitializationSpec, ...]
    initialization_table_version: str
    initialization_table_path: Path
    initialization_table_sha256: str
    multi_start_selection: M4MultiStartRegistry
    metrics: MetricsRegistry
    config_path: Path

    @property
    def precision(self): return self.source_m3_registry.precision
    @property
    def duration_unit(self): return self.source_m3_registry.duration_unit
    @property
    def lapse_max(self): return self.source_m3_registry.lapse_max
    @property
    def objective_weighting(self): return self.source_m3_registry.objective_weighting
    @property
    def parameter_epsilon(self): return self.source_m3_registry.parameter_epsilon
    @property
    def quadratic_tolerance(self): return self.source_m3_registry.quadratic_tolerance
    @property
    def discriminant_tolerance(self): return self.source_m3_registry.discriminant_tolerance
    @property
    def prior_probability_epsilon(self): return self.source_m3_registry.prior_probability_epsilon
    @property
    def log_probability_epsilon(self): return self.source_m3_registry.log_probability_epsilon
    @property
    def state_order(self): return self.source_m3_registry.state_order
    @property
    def first_long_probability(self): return self.source_m3_registry.first_long_probability
    @property
    def neutral_hazard_scale(self): return self.source_m3_registry.neutral_hazard_scale
    @property
    def delta_min(self): return self.source_m3_registry.delta_min
    @property
    def maximum_accepted_projection_fraction(self):
        return self.source_m3_registry.maximum_accepted_projection_fraction
    @property
    def empirical_median_block_length_trials(self): return self.run_length_center
    @property
    def task_moments(self): return self.source_m3_registry.task_moments
    @property
    def split(self): return self.source_m3_registry.split
    @property
    def optimizer(self): return self.source_m3_registry.optimizer
    @property
    def scheduler(self): return self.source_m3_registry.scheduler
    @property
    def seeds(self): return self.source_m3_registry.seeds
    @property
    def convergence(self): return self.source_m3_registry.convergence
    @property
    def fitting(self): return self.source_m3_registry.fitting
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
        raise ValueError("Circular M4 registry inheritance")
    value = _read(path)
    parent = value.get("extends")
    if parent is None:
        return value
    if not isinstance(parent, str) or Path(parent).name != parent:
        raise ValueError("M4 extends must name a sibling registry")
    return _merge(_read_config(path.parent / parent, (*seen, path)), value)


def _artifact(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered M4 artifact is missing: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"M4 artifact checksum mismatch: expected {expected}, observed {observed}")
    return path


def load_m4_model_registry(path: str | Path | None = None) -> M4HSMMRegistry:
    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "m4_hsmm_delta_v1.json"
        if path is None else Path(path).expanduser().resolve()
    )
    config = _read_config(config_path)
    if config.get("model_id") != "M4_hsmm_delta_v1":
        raise ValueError("M4 registry model_id must be M4_hsmm_delta_v1")
    order = tuple(config.get("parameter_order", ()))
    expected_order = (
        "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0",
        "alpha_theta", "bh",
    )
    if order != expected_order:
        raise ValueError("M4 parameter order must append bh to the M3 order")
    root = config_path.parent.parent
    source_reference = config["source_m3_registry"]
    if not isinstance(source_reference, dict):
        raise TypeError("M4 source_m3_registry must be a checksummed artifact reference")
    source_path = _artifact(
        root, str(source_reference["file"]), str(source_reference["sha256"])
    )
    source = load_m3_model_registry(source_path)
    hsmm = config["hsmm"]
    center = int(hsmm["run_length_center"])
    cap = int(hsmm["maximum_represented_run_length"])
    if center != 23 or cap != 65:
        raise ValueError("Registered M4 requires run-length center 23 and cap 65")
    if hsmm.get("tail_state_semantics") != "r65_represents_R_greater_than_or_equal_to_65":
        raise ValueError("M4 requires the registered capped-tail interpretation")
    if hsmm.get("tail_hazard_policy") != "hazard_saturates_at_h65_and_survival_remains_in_tail":
        raise ValueError("M4 tail hazard policy is invalid")
    if hsmm.get("common_hazard_across_neutral_and_biased_states") is not False:
        raise ValueError("M4 must use state-specific tied neutral/biased hazards")
    if float(hsmm.get("neutral_hazard_scale", -1.0)) != 0.5:
        raise ValueError("M4 neutral hazard must be fixed at half the biased hazard")
    if hsmm.get("true_block_predictive_input") is not False or hsmm.get("true_run_length_predictive_input") is not False:
        raise ValueError("True block/run length cannot enter M4 predictions")
    tail_acceptance = config["tail_acceptance"]
    tail_statistic = str(tail_acceptance["statistic"])
    maximum_tail = float(tail_acceptance["maximum_accepted_value"])
    if tail_statistic != "mean_pre_choice_tail_probability_across_all_training_timeline_trials":
        raise ValueError("M4 tail acceptance statistic is invalid")
    if not 0.0 <= maximum_tail <= 1.0:
        raise ValueError("M4 maximum accepted tail probability must lie in [0,1]")
    if tail_acceptance.get("required_for_convergence_and_winner_eligibility") is not True:
        raise ValueError("M4 tail occupancy must gate convergence and winner eligibility")
    reference = config["initialization_table"]
    table_path = _artifact(root, reference["file"], reference["sha256"])
    table = _read(table_path)
    if table.get("table_version") != reference["version"] or table.get("frozen") is not True:
        raise ValueError("M4 initialization version/frozen flag is invalid")
    starts = validate_m4_initialization_table(
        (M4InitializationSpec(**item) for item in table["starts"]),
        lapse_max=source.lapse_max,
        expected_count=int(reference["count"]),
    )
    m3_reference = table["m3_source"]
    m3_table_path = _artifact(root, m3_reference["file"], m3_reference["sha256"])
    m3_table = _read(m3_table_path)
    if m3_table.get("table_version") != m3_reference["table_version"]:
        raise ValueError("M4 source M3 table version mismatch")
    m3_by_id = {int(item["start_id"]): item for item in m3_table["starts"]}
    names = (
        "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0",
        "alpha_theta",
    )
    for start in starts:
        source_row = m3_by_id[start.source_m3_start_id]
        if tuple(getattr(start, name) for name in names) != tuple(float(source_row[name]) for name in names):
            raise ValueError("Every M4 start must preserve its M3 source row exactly")
    inherited = source.multi_start_selection
    multi = config["multi_start_selection"]
    metrics = config["metrics"]
    inherited_metrics = source.metrics
    scale = tuple(float(value) for value in multi["parameter_distance_scale"])
    if len(scale) != 9 or any(value <= 0 for value in scale):
        raise ValueError("M4 parameter-distance scale requires nine positive values")
    return M4HSMMRegistry(
        model_id="M4_hsmm_delta_v1",
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
        source_m3_registry=source,
        initializations=starts,
        initialization_table_version=str(table["table_version"]),
        initialization_table_path=table_path,
        initialization_table_sha256=str(reference["sha256"]),
        multi_start_selection=M4MultiStartRegistry(
            winner_criterion=inherited.winner_criterion,
            near_equal_training_nll_tolerance=inherited.near_equal_training_nll_tolerance,
            identifiability_parameter_distance_threshold=inherited.identifiability_parameter_distance_threshold,
            parameter_distance_scale=scale,
            additional_starts_policy=inherited.additional_starts_policy,
            result_schema_version=str(multi["result_schema_version"]),
        ),
        metrics=MetricsRegistry(
            primary=tuple(metrics["primary"]),
            secondary=tuple(metrics.get("secondary", inherited_metrics.secondary)),
            diagnostics=tuple(metrics["diagnostics"]),
            calibration_bins=int(metrics.get("calibration_bins", inherited_metrics.calibration_bins)),
            psychometric_duration_decimals=int(metrics.get(
                "psychometric_duration_decimals",
                inherited_metrics.psychometric_duration_decimals,
            )),
            classification_threshold=float(metrics.get(
                "classification_threshold", inherited_metrics.classification_threshold
            )),
        ),
        config_path=config_path,
    )
