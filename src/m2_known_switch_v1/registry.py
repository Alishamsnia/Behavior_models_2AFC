"""Typed registry for M2_known_switch_v1.

Proxies genuinely shared fields from the real M0 registry
(``load_model_registry``, unmodified), exactly as
``m1_known_switch_v1/registry.py`` does -- this model's free parameter
vector is M0's five coordinates plus ``delta0``, with no hazard to fit. This
file has no runtime dependency on ``m1_known_switch_v1`` (see
``session.py``'s module docstring for why).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from m0_sensory_v1.registry import ConvergenceRegistry, MetricsRegistry, ModelRegistry, load_model_registry


@dataclass(frozen=True)
class M2KnownSwitchInitializationSpec:
    """One readable, non-Cartesian M2_known_switch optimizer starting point."""

    start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    delta0: float
    purpose: str = ""

    def as_dict(self) -> dict[str, int | float | str]:
        """Return the exact constrained record for manifests and result files."""

        return asdict(self)

    def constrained_tensor(
        self, *, dtype: torch.dtype = torch.float64, device=None
    ) -> torch.Tensor:
        """Return ``(w,s0,ell_short,ell_long,kappa,delta0)`` in fitted coordinates."""

        return torch.tensor(
            [self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.delta0],
            dtype=dtype, device=device,
        )

    def raw_tensor(
        self, *, lapse_max: float = 1.0, epsilon: float = 1e-10,
        dtype: torch.dtype = torch.float64, device=None,
    ) -> torch.Tensor:
        """Convert this constrained start to differentiable raw coordinates."""

        from .parameters import encode_m2_known_switch_constrained_parameters

        return encode_m2_known_switch_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max, epsilon=epsilon,
        )


@dataclass(frozen=True)
class M2KnownSwitchFittingRegistry:
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
class M2KnownSwitchMultiStartRegistry:
    winner_criterion: str
    near_equal_training_nll_tolerance: float
    identifiability_parameter_distance_threshold: float
    parameter_distance_scale: tuple[float, ...]
    additional_starts_policy: str
    result_schema_version: str


@dataclass(frozen=True)
class M2KnownSwitchRegistry:
    """M2_known_switch_v1's own contract plus the immutable parent M0 contract."""

    model_id: str
    scientific_alias: str
    registry_schema_version: str
    software_version: str
    implemented_scope: str
    parameter_order: tuple[str, ...]
    session_policy: str
    prior_probability_epsilon: float
    source_m0_registry: ModelRegistry
    initializations: tuple[M2KnownSwitchInitializationSpec, ...]
    initialization_table_version: str
    initialization_table_path: Path
    initialization_table_sha256: str
    convergence: ConvergenceRegistry
    fitting: M2KnownSwitchFittingRegistry
    multi_start_selection: M2KnownSwitchMultiStartRegistry
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
    def quadratic_tolerance(self): return self.source_m0_registry.quadratic_tolerance
    @property
    def discriminant_tolerance(self): return self.source_m0_registry.discriminant_tolerance
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
        raise ValueError("Circular M2_known_switch registry inheritance")
    value = _read(path)
    parent = value.get("extends")
    if parent is None:
        return value
    if not isinstance(parent, str) or Path(parent).name != parent:
        raise ValueError("M2_known_switch extends must name a sibling registry")
    return _merge(_read_config(path.parent / parent, (*seen, path)), value)


def _artifact(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registered M2_known_switch artifact is missing: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(
            f"M2_known_switch artifact checksum mismatch for {path}: "
            f"expected {expected}, observed {observed}"
        )
    return path


def _validate_initializations(
    items: list[dict[str, Any]], *, lapse_max: float, expected_count: int,
) -> tuple[M2KnownSwitchInitializationSpec, ...]:
    if len(items) != expected_count:
        raise ValueError(f"M2_known_switch requires exactly {expected_count} starts")
    starts = tuple(
        M2KnownSwitchInitializationSpec(
            start_id=item["start_id"], w=item["w"], s0=item["s0"],
            ell_short=item["ell_short"], ell_long=item["ell_long"],
            kappa=item["kappa"], delta0=item["delta0"], purpose=item.get("purpose", ""),
        )
        for item in items
    )
    ids = [start.start_id for start in starts]
    if sorted(ids) != list(range(1, expected_count + 1)):
        raise ValueError("M2_known_switch start IDs must be exactly 1..N with no gaps")
    rows = [start.constrained_tensor() for start in starts]
    if len({tuple(row.tolist()) for row in rows}) != len(rows):
        raise ValueError("M2_known_switch starts must be unique")
    for start in starts:
        if start.w <= 0 or start.s0 <= 0:
            raise ValueError("M2_known_switch starts require positive w and s0")
        if not (0.0 < start.ell_short < lapse_max) or not (0.0 < start.ell_long < lapse_max):
            raise ValueError("M2_known_switch starts require open-interval lapse rates")
        if start.kappa <= 0:
            raise ValueError("M2_known_switch starts require positive kappa")
        if start.delta0 <= 0:
            raise ValueError("M2_known_switch starts require positive delta0")
    return starts


def load_m2_known_switch_model_registry(
    path: str | Path | None = None,
) -> M2KnownSwitchRegistry:
    r"""Load and validate M2_known_switch's complete software contract.

    Purpose
    -------
    Provide one immutable source of truth for M2's fitted static subjective
    prototypes evaluated under the oracle switch schedule, proxying only
    genuinely shared fields from the real M0 registry for the named subject.

    Mathematical definition
    -----------------------
    .. math::

       \theta=(0.5,\sigma(-\delta_0),\sigma(\delta_0)).

    Parameter interpretation
    ------------------------
    ``source_m0_registry`` supplies every genuinely shared field for the
    subject named by the registry file; this registry declares its own
    parameter order, initializations, and convergence checks.

    Input units and shapes
    ----------------------
    ``path`` names a subject-specific registry JSON (defaults to the base
    MC08 registry).

    Returned quantities and shapes
    ------------------------------
    Returns a frozen registry with eight starts and a proxied task/split
    contract for one subject.

    Constraints
    -----------
    Identity, parameter order, checksums, and starts are validated before
    use; every start must preserve its M0 source row exactly and add
    ``delta0=log(4)``.

    Numerical stability
    -------------------
    Every start is decoded and evaluated for a finite readout at load time.

    Causal restrictions
    -------------------
    Registry loading opens configuration and split assignments, never behavior.

    Important edge cases
    --------------------
    Altered frozen artifacts or a mismatched M0 source row raise.

    Related tests
    -------------
    ``test_m2_known_switch_registry_preserves_m0_starts_and_adds_delta0``.
    """

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "m2_known_switch_v1.json"
        if path is None else Path(path).expanduser().resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"M2_known_switch registry does not exist: {config_path}")
    config = _read_config(config_path)
    if config.get("model_id") != "M2_known_switch_v1":
        raise ValueError("M2_known_switch registry model_id is invalid")
    order = tuple(config.get("parameter_order", ()))
    expected_order = ("w", "s0", "ell_short", "ell_long", "kappa", "delta0")
    if order != expected_order:
        raise ValueError("M2_known_switch parameter order must be M0's order plus delta0")

    root = config_path.parent.parent
    source_reference = config["source_m0_registry"]
    if not isinstance(source_reference, dict):
        raise TypeError("M2_known_switch source_m0_registry must be a checksummed artifact reference")
    source_path = _artifact(root, str(source_reference["file"]), str(source_reference["sha256"]))
    source = load_model_registry(source_path)
    if source.model_id != "M0_sensory_v1":
        raise ValueError("M2_known_switch source registry must be a real M0 registry")

    reference = config["initialization_table"]
    table_path = _artifact(root, reference["file"], reference["sha256"])
    table = _read(table_path)
    if table.get("frozen") is not True:
        raise ValueError("M2_known_switch initialization table must be frozen")
    starts = _validate_initializations(
        table["starts"], lapse_max=source.lapse_max, expected_count=int(reference["count"]),
    )
    m0_reference = table["m0_source"]
    m0_table_path = _artifact(root, m0_reference["file"], m0_reference["sha256"])
    m0_table = _read(m0_table_path)
    m0_by_id = {int(item["start_id"]): item for item in m0_table["starts"]}
    names = ("w", "s0", "ell_short", "ell_long", "kappa")
    for start in starts:
        source_row = m0_by_id[start.start_id]
        if tuple(getattr(start, name) for name in names) != tuple(float(source_row[name]) for name in names):
            raise ValueError("Every M2_known_switch start must preserve its M0 source row exactly")

    convergence_config = config["convergence"]
    required_checks = (
        "finite_objective", "finite_parameters", "finite_gradients", "small_gradient_norm",
        "stable_fresh_training_nll", "no_parameter_transform_errors", "no_optimizer_runtime_errors",
    )
    if tuple(convergence_config["required_checks"]) != required_checks:
        raise ValueError("M2_known_switch convergence checks are incomplete")

    multi = config["multi_start_selection"]
    metrics_config = config["metrics"]
    inherited_metrics = source.metrics
    scale = tuple(float(value) for value in multi["parameter_distance_scale"])
    if len(scale) != 6 or any(value <= 0.0 for value in scale):
        raise ValueError("M2_known_switch parameter-distance scale requires six positive values")

    fitting_config = config["fitting"]
    if fitting_config["lbfgs_line_search_fn"] != "strong_wolfe":
        raise ValueError("M2_known_switch L-BFGS must use strong_wolfe line search")
    if int(fitting_config["lbfgs_maximum_steps"]) <= 0:
        raise ValueError("M2_known_switch L-BFGS maximum_steps must be positive")

    return M2KnownSwitchRegistry(
        model_id="M2_known_switch_v1",
        scientific_alias=str(config["scientific_alias"]),
        registry_schema_version=str(config["registry_schema_version"]),
        software_version=str(config["software_version"]),
        implemented_scope=str(config["implemented_scope"]),
        parameter_order=order,
        session_policy=str(config["session_policy"]),
        prior_probability_epsilon=float(config["prior_probability_epsilon"]),
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
        fitting=M2KnownSwitchFittingRegistry(
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
        multi_start_selection=M2KnownSwitchMultiStartRegistry(
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
