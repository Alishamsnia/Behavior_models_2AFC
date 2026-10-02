"""Typed registry for Extended_WSLS_v1.

Proxies genuinely shared fields from the real M0 registry
(``load_model_registry``, unmodified) via read-only forwarding properties --
the same pattern ``m1_stimulus_kernel_v1/registry.py`` uses. Extended_WSLS's
initialization table IS chained to M0's own eight-row table, exactly like
``m1_stimulus_kernel_v1``'s: every start's ``(ell_short, ell_long)`` is
verified to match its M0 source row by ``start_id`` (only ``w``/``s0``/
``kappa`` are dropped, since there is no sensory coordinate here at all).
``convergence``, ``fitting``, and ``multi_start_selection`` are declared
here rather than forwarded, matching ``m1_stimulus_kernel_v1``'s reasoning:
this model has no hazard/HMM/hidden-state machinery, so its own convergence
contract is a strict subset of the sensory families' own.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from m0_sensory_v1.registry import ConvergenceRegistry, MetricsRegistry, ModelRegistry, load_model_registry

from .initialization import (
    ExtendedWslsInitializationSpec,
    validate_extended_wsls_initialization_table,
)


@dataclass(frozen=True)
class ExtendedWslsFittingRegistry:
    """Own L-BFGS/hybrid defaults, matching the family's ``fitting`` shape."""

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
class ExtendedWslsMultiStartRegistry:
    """Own winner/identifiability-ridge policy (two-parameter distance scale)."""

    winner_criterion: str
    near_equal_training_nll_tolerance: float
    identifiability_parameter_distance_threshold: float
    parameter_distance_scale: tuple[float, ...]
    additional_starts_policy: str
    result_schema_version: str


@dataclass(frozen=True)
class ExtendedWslsRegistry:
    """Extended_WSLS_v1's own contract plus the immutable parent M0 contract."""

    model_id: str
    scientific_alias: str
    registry_schema_version: str
    software_version: str
    implemented_scope: str
    parameter_order: tuple[str, ...]
    session_policy: str
    initial_pi: float
    source_m0_registry: ModelRegistry
    initializations: tuple[ExtendedWslsInitializationSpec, ...]
    initialization_table_version: str
    initialization_table_path: Path
    initialization_table_sha256: str
    convergence: ConvergenceRegistry
    fitting: ExtendedWslsFittingRegistry
    multi_start_selection: ExtendedWslsMultiStartRegistry
    metrics: MetricsRegistry
    config_path: Path

    @property
    def precision(self): return self.source_m0_registry.precision
    @property
    def duration_unit(self): return self.source_m0_registry.duration_unit
    @property
    def lapse_max(self): return self.source_m0_registry.lapse_max
    @property
    def objective_weighting(self): return self.source_m0_registry.objective_weighting
    @property
    def parameter_epsilon(self): return self.source_m0_registry.parameter_epsilon
    @property
    def log_probability_epsilon(self): return self.source_m0_registry.log_probability_epsilon
    @property
    def task_moments(self): return self.source_m0_registry.task_moments
    @property
    def split(self): return self.source_m0_registry.split
    @property
    def optimizer(self): return self.source_m0_registry.optimizer
    @property
    def scheduler(self): return self.source_m0_registry.scheduler
    @property
    def seeds(self): return self.source_m0_registry.seeds


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
        raise ValueError("Circular Extended_WSLS registry inheritance")
    value = _read(path)
    parent = value.get("extends")
    if parent is None:
        return value
    if not isinstance(parent, str) or Path(parent).name != parent:
        raise ValueError("Extended_WSLS extends must name a sibling registry")
    return _merge(_read_config(path.parent / parent, (*seen, path)), value)


def _artifact(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered Extended_WSLS artifact is missing: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(
            f"Extended_WSLS artifact checksum mismatch for {path}: "
            f"expected {expected}, observed {observed}"
        )
    return path


def load_extended_wsls_model_registry(
    path: str | Path | None = None,
) -> ExtendedWslsRegistry:
    r"""Load and validate Extended_WSLS's complete software contract.

    Purpose
    -------
    Provide one immutable source of truth for the fixed-alpha win-stay/
    lose-shift-on-category model, proxying only genuinely shared fields
    from the real M0 registry for the named subject.

    Mathematical definition
    -----------------------
    .. math::

       \pi_1=0.5,\qquad \pi_{t+1}=C_t\ \text{(exactly, whenever feedback is valid)}.

    Parameter interpretation
    ------------------------
    ``source_m0_registry`` supplies every genuinely shared field (task
    moments, split, optimizer, scheduler, seeds, precision, lapse_max) for
    the subject named by the registry file; this registry declares its own
    parameter order, initial prior, initializations, and convergence
    checks.

    Constraints
    -----------
    Identity, parameter order, initial prior, checksums, and starts are
    validated before use.

    Related tests
    -------------
    ``test_extended_wsls_registry_loads_and_validates``.
    """

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "extended_wsls_v1.json"
        if path is None else Path(path).expanduser().resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Extended_WSLS registry does not exist: {config_path}")
    config = _read_config(config_path)
    if config.get("model_id") != "Extended_WSLS_v1":
        raise ValueError("Extended_WSLS registry model_id is invalid")
    order = tuple(config.get("parameter_order", ()))
    expected_order = ("ell_short", "ell_long", "a")
    if order != expected_order:
        raise ValueError("Extended_WSLS parameter order must be (ell_short, ell_long, a)")

    root = config_path.parent.parent
    source_reference = config["source_m0_registry"]
    if not isinstance(source_reference, dict):
        raise TypeError("Extended_WSLS source_m0_registry must be a checksummed artifact reference")
    source_path = _artifact(root, str(source_reference["file"]), str(source_reference["sha256"]))
    source = load_model_registry(source_path)
    if source.model_id != "M0_sensory_v1":
        raise ValueError("Extended_WSLS source registry must be a real M0 registry")

    initial_pi = float(config["initial_pi"])
    if initial_pi != 0.5:
        raise ValueError("Extended_WSLS requires the fixed initial prior pi_1=0.5")

    reference = config["initialization_table"]
    table_path = _artifact(root, reference["file"], reference["sha256"])
    table = _read(table_path)
    if table.get("frozen") is not True:
        raise ValueError("Extended_WSLS initialization table must be frozen")
    starts = validate_extended_wsls_initialization_table(
        (
            ExtendedWslsInitializationSpec(
                start_id=item["start_id"], ell_short=item["ell_short"],
                ell_long=item["ell_long"], a=item["a"], purpose=item.get("purpose", ""),
            )
            for item in table["starts"]
        ),
        lapse_max=source.lapse_max,
        expected_count=int(reference["count"]),
    )
    m0_reference = table["m0_source"]
    m0_table_path = _artifact(root, m0_reference["file"], m0_reference["sha256"])
    m0_table = _read(m0_table_path)
    m0_by_id = {int(item["start_id"]): item for item in m0_table["starts"]}
    names = ("ell_short", "ell_long")
    for start in starts:
        source_row = m0_by_id[start.start_id]
        if tuple(getattr(start, name) for name in names) != tuple(float(source_row[name]) for name in names):
            raise ValueError("Every Extended_WSLS start must preserve its M0 source row exactly")

    convergence_config = config["convergence"]
    required_checks = (
        "finite_objective", "finite_parameters", "finite_gradients", "small_gradient_norm",
        "stable_fresh_training_nll", "no_parameter_transform_errors", "no_optimizer_runtime_errors",
    )
    if tuple(convergence_config["required_checks"]) != required_checks:
        raise ValueError("Extended_WSLS convergence checks are incomplete")

    multi = config["multi_start_selection"]
    metrics_config = config["metrics"]
    inherited_metrics = source.metrics
    scale = tuple(float(value) for value in multi["parameter_distance_scale"])
    if len(scale) != 3 or any(value <= 0.0 for value in scale):
        raise ValueError("Extended_WSLS parameter-distance scale requires three positive values")

    fitting_config = config["fitting"]
    if fitting_config["lbfgs_line_search_fn"] != "strong_wolfe":
        raise ValueError("Extended_WSLS L-BFGS must use strong_wolfe line search")
    if int(fitting_config["lbfgs_maximum_steps"]) <= 0:
        raise ValueError("Extended_WSLS L-BFGS maximum_steps must be positive")

    return ExtendedWslsRegistry(
        model_id="Extended_WSLS_v1",
        scientific_alias=str(config["scientific_alias"]),
        registry_schema_version=str(config["registry_schema_version"]),
        software_version=str(config["software_version"]),
        implemented_scope=str(config["implemented_scope"]),
        parameter_order=order,
        session_policy=str(config["session_policy"]),
        initial_pi=initial_pi,
        source_m0_registry=source,
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
        fitting=ExtendedWslsFittingRegistry(
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
        multi_start_selection=ExtendedWslsMultiStartRegistry(
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
