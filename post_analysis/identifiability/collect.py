"""Data assembly for the identifiability report: multistart spread (pure
read of already-computed JSON) and Fisher information / Hessian curvature
(re-evaluates each model's differentiable NLL at an already-fitted point --
never re-fits; no optimizer is invoked here).

Model registry: reuses ``post_analysis.expanded_model_comparison.config``'s
frozen 13-model registry (``MODEL_ORDER``/``MODEL_BY_ID``) -- the same
model set, names, and colors as the expanded full-session-matched model
comparison -- rather than the older, coarser M0-M4 chain in
``post_analysis.common``. Results live under the same
``results_sliding_window/<directory>/<mouse>/full_session_matched_v1`` /
``results_single_session/<directory>/<mouse>/full_session_matched_v1``
layout that registry's ``directory`` field already points at.

``sys.path`` note
------------------
This machine also has an OLDER, DIVERGENT sibling checkout
(``Bayesian_model`` without ``_v2``) ``pip install -e``'d, whose ``src/`` is
on ``sys.path`` under the exact same top-level package names used here
(``m0_sensory_v1``, ``pure_wsls_v1``, ...). A bare ``import pure_wsls_v1``
would therefore silently import the WRONG codebase with no error. We insert
THIS repo's ``src/`` at the front of ``sys.path`` before importing any model
package so it always wins the lookup, and we validate this on every run
(see ``_recomputed_mean_nll`` cross-check in ``compute_fisher_information``):
the re-evaluated mean NLL at the loaded ``best_raw_parameters`` must
reproduce the fit's own recorded ``recalculated_best_training_mean_nll`` to
near machine precision, which would fail immediately if the wrong package
were ever picked up.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from post_analysis import common
from post_analysis.expanded_model_comparison import config as expanded_config
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, MODEL_ORDER

_SRC_ROOT = str(common.REPO_ROOT / "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
elif sys.path[0] != _SRC_ROOT:
    # Already present but shadowed by something earlier (e.g. the stale
    # sibling checkout) -- move it to the front so it wins.
    sys.path.remove(_SRC_ROOT)
    sys.path.insert(0, _SRC_ROOT)

import torch  # noqa: E402

from m0_sensory_v1.model.session import (  # noqa: E402
    evaluate_mouse_objective, session_trials_from_frame,
)
from m0_sensory_v1.model.m1_session import (  # noqa: E402
    evaluate_m1_mouse_objective, m1_session_trials_from_frame,
)
from m0_sensory_v1.model.m2_session import evaluate_m2_mouse_objective  # noqa: E402
from m0_sensory_v1.registry import load_model_registry  # noqa: E402
from m0_sensory_v1.m1_registry import load_m1_model_registry  # noqa: E402
from m0_sensory_v1.m2_registry import load_m2_model_registry  # noqa: E402

from pure_wsls_v1.session import evaluate_pure_wsls_mouse_objective  # noqa: E402
from pure_wsls_v1.registry import load_pure_wsls_model_registry  # noqa: E402
from extended_wsls_v1.session import evaluate_extended_wsls_mouse_objective  # noqa: E402
from extended_wsls_v1.registry import load_extended_wsls_model_registry  # noqa: E402

from m1_action_kernel_v1.session import evaluate_m1_action_kernel_mouse_objective  # noqa: E402
from m1_action_kernel_v1.registry import load_m1_action_kernel_model_registry  # noqa: E402
from m1_stimulus_kernel_v1.session import evaluate_m1_stimulus_kernel_mouse_objective  # noqa: E402
from m1_stimulus_kernel_v1.registry import load_m1_stimulus_kernel_model_registry  # noqa: E402
from m1_known_switch_v1.session import (  # noqa: E402
    evaluate_m1_known_switch_mouse_objective, m1_known_switch_session_trials_from_frame,
)
from m1_known_switch_v1.registry import load_m1_known_switch_model_registry  # noqa: E402
from m1_dynamic_block_length_v1.session import evaluate_m1_dynamic_block_length_mouse_objective  # noqa: E402
from m1_dynamic_block_length_v1.registry import load_m1_dynamic_block_length_model_registry  # noqa: E402

from m2_known_switch_v1.session import (  # noqa: E402
    evaluate_m2_known_switch_mouse_objective, m2_known_switch_session_trials_from_frame,
)
from m2_known_switch_v1.registry import load_m2_known_switch_model_registry  # noqa: E402
from m2_dynamic_block_length_v1.session import evaluate_m2_dynamic_block_length_mouse_objective  # noqa: E402
from m2_dynamic_block_length_v1.registry import load_m2_dynamic_block_length_model_registry  # noqa: E402
from m2_dynamic_previous_wsls_bias_v1.session import (  # noqa: E402
    evaluate_m2_dynamic_previous_wsls_bias_mouse_objective,
)
from m2_dynamic_previous_wsls_bias_v1.registry import (  # noqa: E402
    load_m2_dynamic_previous_wsls_bias_model_registry,
)
from m2_dynamic_previous_action_bias_v1.session import (  # noqa: E402
    evaluate_m2_dynamic_previous_action_bias_mouse_objective,
)
from m2_dynamic_previous_action_bias_v1.registry import (  # noqa: E402
    load_m2_dynamic_previous_action_bias_model_registry,
)


# ---------------------------------------------------------------------------
# Path helpers -- same on-disk layout as post_analysis.common, but keyed off
# the expanded-model-comparison registry's 13 model_ids/directories instead
# of common.MODEL_SPECS's coarser M0-M4 chain.
# ---------------------------------------------------------------------------

def _sliding_window_dir(model_id: str, mouse_id: str) -> Path:
    spec = MODEL_BY_ID[model_id]
    return common.RESULTS_SLIDING_ROOT / spec.directory / mouse_id / expanded_config.WORKFLOW_DIRECTORY


def _single_session_dir(model_id: str, mouse_id: str) -> Path:
    spec = MODEL_BY_ID[model_id]
    return common.RESULTS_SINGLE_ROOT / spec.directory / mouse_id / expanded_config.WORKFLOW_DIRECTORY


def _list_window_dirs(model_id: str, mouse_id: str) -> list[Path]:
    root = _sliding_window_dir(model_id, mouse_id)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("window_"))


def _list_session_dirs(model_id: str, mouse_id: str) -> list[Path]:
    root = _single_session_dir(model_id, mouse_id)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_"))


# ---------------------------------------------------------------------------
# Page 1: multistart parameter spread (no model evaluation -- pure JSON read)
# ---------------------------------------------------------------------------

# A run is treated as "near-best" (kept for spread/instability statistics)
# when its own recalculated training mean NLL is within this many nats/trial
# of the window's WINNING start's NLL. Motivated by a concrete failure found
# in MC08/Extended_WSLS_v1 window 30: 5 of 8 starts land within ~5e-9 of each
# other (a converged cluster, some flagged converged=True, two converged=False
# despite matching NLL almost exactly), while 1 start collapses to a
# chance-level fit (mean NLL 0.693 vs the cluster's 0.661, raw a=-19.386) --
# unfiltered, that one bad start dominates the SD and corrupts both the
# multistart-spread page and the restart-instability ranking derived from it.
# 0.01 nats/trial is ~30x tighter than that observed bad-start gap while
# staying loose enough to keep near-duplicate optima whose own convergence
# flag is a false negative (the two converged=False-but-matching starts above).
NEAR_BEST_NLL_TOLERANCE = 0.01


def _multistart_runs(window_dir: Path) -> list[dict]:
    """The per-start fit results for one window, as a list of dicts each
    carrying (at least) ``start_id``, ``best_constrained_parameters``,
    ``converged``, and ``recalculated_best_training_mean_nll``.

    Prefers the window-level ``fit_result.json`` (verified to carry the same
    ``runs``/``winner_start_id`` as ``starts/multistart_summary.json`` when
    both exist, and present for every model family including the few --
    M1/M2 known-switch, M1_stimulus_kernel -- whose ``starts/`` subdirectory
    never got a combined summary written); falls back to
    ``starts/multistart_summary.json`` and then the individual
    ``starts/start_NN/fit_result.json`` files for robustness.
    """
    top_level = common.read_json(window_dir / "fit_result.json")
    if top_level and top_level.get("runs"):
        return top_level["runs"]
    summary = common.read_json(window_dir / "starts" / "multistart_summary.json")
    if summary and summary.get("runs"):
        return summary["runs"]
    runs = []
    starts_dir = window_dir / "starts"
    if not starts_dir.is_dir():
        return runs
    for start_dir in sorted(p for p in starts_dir.iterdir() if p.is_dir() and p.name.startswith("start_")):
        run = common.read_json(start_dir / "fit_result.json")
        if run is not None:
            runs.append(run)
    return runs


def _window_winner_run(window_dir: Path) -> Optional[dict]:
    """The winning start's own run record for one window (has
    ``best_raw_parameters``/``best_constrained_parameters``/
    ``recalculated_best_training_mean_nll``), or ``None`` if no winner was
    selected."""
    top_level = common.read_json(window_dir / "fit_result.json")
    if not top_level:
        return None
    winner_id = top_level.get("winner_start_id")
    if winner_id is None:
        return None
    for run in top_level.get("runs", []):
        if run.get("start_id") == winner_id:
            return run
    return None


def _qualifying_multistart_runs(window_dir: Path) -> list[dict]:
    """``_multistart_runs`` filtered to runs worth including in a spread/
    instability statistic: finite constrained parameters, and a recalculated
    training mean NLL within ``NEAR_BEST_NLL_TOLERANCE`` of the window's own
    winning NLL (see that constant's docstring for why an unfiltered set is
    unsafe). Falls back to only the finite-parameter filter if the winner's
    own NLL isn't available, rather than dropping the window entirely.
    """
    runs = [
        run for run in _multistart_runs(window_dir)
        if run.get("best_constrained_parameters") is not None
    ]
    winner = _window_winner_run(window_dir)
    winner_nll = winner.get("recalculated_best_training_mean_nll") if winner else None
    if winner_nll is None or not np.isfinite(winner_nll):
        return runs
    qualifying = []
    for run in runs:
        nll = run.get("recalculated_best_training_mean_nll")
        if nll is not None and np.isfinite(nll) and abs(float(nll) - float(winner_nll)) <= NEAR_BEST_NLL_TOLERANCE:
            qualifying.append(run)
    return qualifying


def collect_multistart_spread(mouse_id: str) -> pd.DataFrame:
    """One row per (model, window, parameter): the sample SD of that
    parameter's CONSTRAINED value across the near-best-NLL, independently-
    initialized sliding-window restarts for that window (see
    ``NEAR_BEST_NLL_TOLERANCE``: a clearly-failed/outlier start no longer
    silently inflates this). Uses every window with at least 2 qualifying
    starts; reads only per-start fit result JSON already written during
    fitting -- no new model evaluation.
    """
    rows = []
    for model_id in MODEL_ORDER:
        param_names = MODEL_BY_ID[model_id].parameter_names
        for window_dir in _list_window_dirs(model_id, mouse_id):
            runs = _qualifying_multistart_runs(window_dir)
            if not runs:
                continue
            manifest = common.read_json(window_dir / "window_manifest.json") or {}
            window_id = manifest.get("window_id")
            constrained = np.array(
                [run["best_constrained_parameters"] for run in runs], dtype=float,
            )
            if constrained.shape[0] < 2:
                continue
            if constrained.shape[1] != len(param_names):
                raise ValueError(
                    f"{model_id}/{mouse_id} window {window_id}: multistart summary has "
                    f"{constrained.shape[1]} parameters, expected {len(param_names)}"
                )
            spread = constrained.std(axis=0, ddof=1)
            for param_index, param_name in enumerate(param_names):
                rows.append({
                    "model_id": model_id, "window_id": window_id,
                    "param_name": param_name, "n_starts": int(constrained.shape[0]),
                    "spread_std": float(spread[param_index]),
                })
    columns = ["model_id", "window_id", "param_name", "n_starts", "spread_std"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Page 6: fitted parameter VALUES across all windows (no model evaluation --
# pure JSON read of each window's own winning fit).
# ---------------------------------------------------------------------------

def collect_parameter_trajectories(mouse_id: str) -> pd.DataFrame:
    """One row per (model, window, parameter): the WINNING start's fitted
    constrained parameter value for that window (not a restart mean/spread --
    the actual value this window's fit reports downstream elsewhere in the
    pipeline). Skips windows with no selected winner; reads only JSON already
    on disk -- no new model evaluation.
    """
    rows = []
    for model_id in MODEL_ORDER:
        param_names = MODEL_BY_ID[model_id].parameter_names
        for window_dir in _list_window_dirs(model_id, mouse_id):
            winner = _window_winner_run(window_dir)
            if winner is None or winner.get("best_constrained_parameters") is None:
                continue
            manifest = common.read_json(window_dir / "window_manifest.json") or {}
            window_id = manifest.get("window_id")
            values = winner["best_constrained_parameters"]
            if len(values) != len(param_names):
                raise ValueError(
                    f"{model_id}/{mouse_id} window {window_id}: winner has {len(values)} "
                    f"parameters, expected {len(param_names)}"
                )
            for param_name, value in zip(param_names, values):
                rows.append({
                    "model_id": model_id, "window_id": window_id,
                    "param_name": param_name, "value": float(value),
                })
    columns = ["model_id", "window_id", "param_name", "value"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Page 7: per-window "restart-instability" rank -- NOT a Fisher-information/
# curvature rank. Ranks each model's named parameters, within a window, by
# how much the near-best-NLL restarts disagree on that parameter's value
# (``collect_multistart_spread``'s ``spread_std``) -- large disagreement is a
# PRACTICAL red flag (see figures.py's page 1 caption), not a measurement of
# local curvature the way a Hessian SE is. See
# ``validate_restart_instability_against_hessian`` for a small benchmarked
# cross-check of how well this proxy tracks real per-window Hessian ranks.
# ---------------------------------------------------------------------------

def _sloppiest_to_stiffest_percentile(values: np.ndarray) -> np.ndarray:
    """Percentile rank of each entry in ``values`` from 0 (largest value --
    "sloppiest"/least stable) to 1 (smallest value -- "stiffest"/most
    stable), matching page 3's eigenvalue-rank convention. Ties get the
    average percentile of the tied group (pandas ``rank(method="average")``).
    """
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5)
    ascending_rank = pd.Series(values).rank(method="average").to_numpy()  # 1=smallest ("stiffest")
    return 1.0 - (ascending_rank - 1.0) / (n - 1.0)


def collect_restart_instability_rank(mouse_id: str, spread: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (model, window, parameter): ``spread_std`` from
    ``collect_multistart_spread`` plus its within-window, within-model
    percentile rank (0 = most restart-unstable ("sloppiest"), 1 = most
    restart-stable ("stiffest")) -- comparable across models with different
    parameter counts because it's a rank, not a raw magnitude. Zero new
    model evaluations (built entirely from ``collect_multistart_spread``).
    """
    if spread is None:
        spread = collect_multistart_spread(mouse_id)
    if spread.empty:
        return spread.assign(percentile_rank=pd.Series(dtype=float))
    parts = []
    for (model_id, window_id), group in spread.groupby(["model_id", "window_id"], sort=False):
        group = group.copy()
        group["percentile_rank"] = _sloppiest_to_stiffest_percentile(group["spread_std"].to_numpy())
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Pages 2-3: Fisher information / Hessian curvature at one representative
# already-fitted point per model (re-evaluation only -- no re-fitting).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelAdapter:
    load_registry: Callable[[str], object]
    session_from_frame: Callable[..., object]
    objective_fn: Callable[..., object]
    kwargs_fn: Callable[[object, str], dict]


def _m0_kwargs(registry, role: str) -> dict:
    return dict(
        objective_weighting=registry.objective_weighting, required_split_role=role,
        lapse_max=registry.lapse_max, parameter_epsilon=registry.parameter_epsilon,
        log_probability_epsilon=registry.log_probability_epsilon,
    )


def _wsls_kwargs(registry, role: str) -> dict:
    """Pure_WSLS / Extended_WSLS: base kwargs + the scalar prior's reset value."""
    kwargs = _m0_kwargs(registry, role)
    kwargs["initial_pi"] = registry.initial_pi
    return kwargs


def _switch_kwargs(registry, role: str) -> dict:
    """M1/M2 known-switch (oracle block-switch signal, no hazard machinery)."""
    kwargs = _m0_kwargs(registry, role)
    kwargs.update(
        prior_probability_epsilon=registry.prior_probability_epsilon,
        quadratic_tolerance=registry.quadratic_tolerance,
        discriminant_tolerance=registry.discriminant_tolerance,
    )
    return kwargs


def _history_kwargs(registry, role: str) -> dict:
    """M1 action-kernel / stimulus-kernel: switch kwargs + scalar-prior reset."""
    kwargs = _switch_kwargs(registry, role)
    kwargs["initial_pi"] = registry.initial_pi
    return kwargs


def _hmm_kwargs(registry, role: str) -> dict:
    """Base M1 (hmm-true) / M2 (hmm-static): switch kwargs + first-long prior."""
    kwargs = _switch_kwargs(registry, role)
    kwargs["first_long_probability"] = registry.first_long_probability
    return kwargs


def _dynamic_kwargs(registry, role: str) -> dict:
    """M1/M2 dynamic block length: hmm kwargs + run-length hazard settings."""
    kwargs = _hmm_kwargs(registry, role)
    kwargs["run_length_center"] = registry.run_length_center
    kwargs["maximum_run_length"] = registry.maximum_represented_run_length
    return kwargs


def _dynamic_bias_kwargs(registry, role: str) -> dict:
    """M2 dynamic + WSLS/repetition bias: dynamic kwargs + scalar-prior reset."""
    kwargs = _dynamic_kwargs(registry, role)
    kwargs["initial_pi"] = registry.initial_pi
    return kwargs


_MODEL_ADAPTERS: dict[str, _ModelAdapter] = {
    "pure_wsls": _ModelAdapter(
        load_pure_wsls_model_registry, m1_session_trials_from_frame,
        evaluate_pure_wsls_mouse_objective, _wsls_kwargs,
    ),
    "extended_wsls": _ModelAdapter(
        load_extended_wsls_model_registry, m1_session_trials_from_frame,
        evaluate_extended_wsls_mouse_objective, _wsls_kwargs,
    ),
    "m0_sensory": _ModelAdapter(
        load_model_registry, session_trials_from_frame,
        evaluate_mouse_objective, _m0_kwargs,
    ),
    "m1_action_kernel": _ModelAdapter(
        load_m1_action_kernel_model_registry, m1_session_trials_from_frame,
        evaluate_m1_action_kernel_mouse_objective, _history_kwargs,
    ),
    "m1_stimulus_kernel": _ModelAdapter(
        load_m1_stimulus_kernel_model_registry, m1_session_trials_from_frame,
        evaluate_m1_stimulus_kernel_mouse_objective, _history_kwargs,
    ),
    "m1_known_switch": _ModelAdapter(
        load_m1_known_switch_model_registry, m1_known_switch_session_trials_from_frame,
        evaluate_m1_known_switch_mouse_objective, _switch_kwargs,
    ),
    "m1_hmm_true": _ModelAdapter(
        load_m1_model_registry, m1_session_trials_from_frame,
        evaluate_m1_mouse_objective, _hmm_kwargs,
    ),
    "m1_dynamic": _ModelAdapter(
        load_m1_dynamic_block_length_model_registry, m1_session_trials_from_frame,
        evaluate_m1_dynamic_block_length_mouse_objective, _dynamic_kwargs,
    ),
    "m2_known_switch": _ModelAdapter(
        load_m2_known_switch_model_registry, m2_known_switch_session_trials_from_frame,
        evaluate_m2_known_switch_mouse_objective, _switch_kwargs,
    ),
    "m2_static": _ModelAdapter(
        load_m2_model_registry, m1_session_trials_from_frame,
        evaluate_m2_mouse_objective, _hmm_kwargs,
    ),
    "m2_dynamic": _ModelAdapter(
        load_m2_dynamic_block_length_model_registry, m1_session_trials_from_frame,
        evaluate_m2_dynamic_block_length_mouse_objective, _dynamic_kwargs,
    ),
    "m2_dynamic_wsls_bias": _ModelAdapter(
        load_m2_dynamic_previous_wsls_bias_model_registry, m1_session_trials_from_frame,
        evaluate_m2_dynamic_previous_wsls_bias_mouse_objective, _dynamic_bias_kwargs,
    ),
    "m2_dynamic_repetition_bias": _ModelAdapter(
        load_m2_dynamic_previous_action_bias_model_registry, m1_session_trials_from_frame,
        evaluate_m2_dynamic_previous_action_bias_mouse_objective, _dynamic_bias_kwargs,
    ),
}

if set(_MODEL_ADAPTERS) != set(MODEL_ORDER):
    raise RuntimeError(
        "identifiability's model adapters disagree with the expanded model "
        "comparison's MODEL_ORDER -- update _MODEL_ADAPTERS to match."
    )


def _config_path(model_id: str, mouse_id: str) -> Path:
    """Mouse-specific config if one exists (verified elsewhere to match the
    ``registry_config_sha256`` actually recorded in that mouse's fits), else
    the shared base config (used as-is for mice without an override, e.g.
    MC08)."""
    base = MODEL_BY_ID[model_id].directory.lower()
    mouse_specific = common.REPO_ROOT / "config" / f"{base}_{mouse_id.lower()}.json"
    if mouse_specific.is_file():
        return mouse_specific
    return common.REPO_ROOT / "config" / f"{base}.json"


def _representative_session_dir(model_id: str, mouse_id: str) -> Optional[Path]:
    """The chronologically-first converged single-session fine-tune fit --
    single-session (not sliding-window) because it is ~15x fewer trials per
    forward pass, so cheap to differentiate through even for the heavier
    dynamic-block-length HSMM recursions."""
    for session_dir in _list_session_dirs(model_id, mouse_id):
        fit = common.read_json(session_dir / "fine_tune_fit_result.json")
        if fit and fit.get("converged") and (session_dir / "trial_predictions.csv").is_file():
            return session_dir
    return None


@dataclass(frozen=True)
class FisherInformationResult:
    model_id: str
    session_id: str
    param_names: tuple
    raw_point: np.ndarray
    hessian: np.ndarray             # Hessian of SUMMED (not mean) train NLL w.r.t. raw params
    eigenvalues: np.ndarray         # ascending (numpy.linalg.eigh convention)
    eigenvectors: np.ndarray        # columns correspond to eigenvalues
    raw_se: np.ndarray              # sqrt(diag(pseudo-inverse)); NaN-safe near-zero-curvature floor
    n_trials: int
    mean_nll: float
    hessian_seconds: float


def _pseudo_inverse_from_eigh(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, relative_floor: float = 1e-8,
) -> np.ndarray:
    """Inverse with near-zero/negative (numerical-noise) eigenvalues floored
    at ``max(eigenvalue) * relative_floor`` before inverting -- a sloppy
    direction gets a large but finite variance instead of NaN/blow-up from a
    tiny negative curvature value at a not-quite-exact optimum."""
    floor = max(float(eigenvalues.max()) * relative_floor, 1e-12)
    clipped = np.clip(eigenvalues, floor, None)
    return (eigenvectors * (1.0 / clipped)) @ eigenvectors.T


def _hessian_at_point(
    model_id: str, label: str, adapter: _ModelAdapter, sessions: tuple,
    raw_point: torch.Tensor, registry, *, recorded_mean_nll: Optional[float],
) -> FisherInformationResult:
    """Shared Hessian/eigendecomposition core used by both
    ``compute_fisher_information`` (one representative session per model)
    and ``compute_window_fisher_information`` (one sliding window's full
    9-session training set) -- re-evaluation only (``torch.autograd.
    functional.hessian`` on the model's own differentiable objective), never
    a re-fit. ``label`` is an opaque identifier (session dirname or window
    id) carried through into the result for reporting.
    """
    kwargs = adapter.kwargs_fn(registry, "train")

    def total_nll(raw: torch.Tensor) -> torch.Tensor:
        return adapter.objective_fn(sessions, raw, registry.task_moments, **kwargs).total_negative_log_likelihood

    with torch.no_grad():
        reference = adapter.objective_fn(sessions, raw_point, registry.task_moments, **kwargs)
    n_trials = int(reference.total_valid_trial_count)
    mean_nll = float(reference.loss)
    if recorded_mean_nll is not None and not np.isclose(mean_nll, recorded_mean_nll, rtol=1e-6, atol=1e-9):
        raise RuntimeError(
            f"{model_id}/{label}: re-evaluated training mean NLL ({mean_nll}) does not match "
            f"the fit's own recorded value ({recorded_mean_nll}) -- wrong model code, config, or "
            "data reloaded (see this module's sys.path note)."
        )

    start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(total_nll, raw_point).detach().numpy()
    elapsed = time.perf_counter() - start

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    pseudo_inverse = _pseudo_inverse_from_eigh(eigenvalues, eigenvectors)
    raw_se = np.sqrt(np.clip(np.diag(pseudo_inverse), 0.0, None))

    return FisherInformationResult(
        model_id=model_id, session_id=label,
        param_names=MODEL_BY_ID[model_id].parameter_names,
        raw_point=raw_point.numpy(), hessian=hessian,
        eigenvalues=eigenvalues, eigenvectors=eigenvectors,
        raw_se=raw_se, n_trials=n_trials, mean_nll=mean_nll,
        hessian_seconds=elapsed,
    )


def _decades_spanned(result: FisherInformationResult, floor: float = 1e-12) -> float:
    """log10(largest eigenvalue) - log10(smallest eigenvalue), floored --
    the "decades stiffest-to-sloppiest" span shown on page 3, factored out
    here so page 8's cross-model summary reuses the exact same definition."""
    floored_min = max(float(result.eigenvalues.min()), floor)
    return float(np.log10(float(result.eigenvalues.max())) - np.log10(floored_min))


def compute_fisher_information(mouse_id: str) -> dict[str, FisherInformationResult]:
    """Hessian (w.r.t. raw/unconstrained parameters) of the training-split
    negative log-likelihood at one representative already-fitted point per
    model, via ``torch.autograd.functional.hessian`` -- re-evaluation only,
    the exact differentiable objective each model's own fitting loop already
    uses (see e.g. ``src/pure_wsls_v1/fit.py``, ``src/m2_dynamic_block_length_v1/fit.py``),
    never a re-fit.
    """
    results: dict[str, FisherInformationResult] = {}
    for model_id in MODEL_ORDER:
        adapter = _MODEL_ADAPTERS[model_id]
        session_dir = _representative_session_dir(model_id, mouse_id)
        if session_dir is None:
            continue
        fit = common.read_json(session_dir / "fine_tune_fit_result.json")
        raw_point = torch.tensor(fit["best_raw_parameters"], dtype=torch.float64)
        registry = adapter.load_registry(str(_config_path(model_id, mouse_id)))

        frame = pd.read_csv(session_dir / "trial_predictions.csv").sort_values("trial_index")
        train_frame = frame.loc[frame["split_role"] == "train"]
        sessions = (adapter.session_from_frame(train_frame, split_role="train"),)

        results[model_id] = _hessian_at_point(
            model_id, session_dir.name, adapter, sessions, raw_point, registry,
            recorded_mean_nll=fit.get("recalculated_best_training_mean_nll"),
        )
    return results


# ---------------------------------------------------------------------------
# Window-level Hessians -- NOT part of the default report path (see module
# docstring / figures.py page 7 caption). Used only for
# ``validate_restart_instability_against_hessian``'s small benchmarked
# cross-check: each window's full 9-session training set is ~9x the trials
# of the single-session Hessians above, so this is deliberately run on a
# handful of (model, window) pairs, never all 13 models x 30 windows (that
# full sweep was estimated at ~8-9 hours and was ruled out during planning).
# ---------------------------------------------------------------------------

def _window_training_sessions(adapter: _ModelAdapter, window_dir: Path) -> tuple:
    """The window's 9 training sessions as separate Session objects (one per
    ``session_id``, sorted chronologically by each session's own first trial
    index) -- matches how the real sliding-window fitting pipeline pools
    multiple sessions into one objective, unlike the single merged-frame
    session used by the cheap representative-session Hessians above."""
    frame = pd.read_csv(window_dir / "trial_predictions.csv").sort_values("trial_index")
    train_frame = frame.loc[frame["split_role"] == "train"]
    order = train_frame.groupby("session_id")["trial_index"].min().sort_values()
    return tuple(
        adapter.session_from_frame(
            train_frame.loc[train_frame["session_id"] == session_id].sort_values("trial_index"),
            split_role="train",
        )
        for session_id in order.index
    )


def compute_window_fisher_information(
    mouse_id: str, model_ids: list[str], window_ids: list[int],
) -> dict[tuple[str, int], FisherInformationResult]:
    """Hessian at the WINNING fit of each requested (model, window) pair,
    evaluated on that window's own full 9-session training set -- same
    re-evaluation-only machinery as ``compute_fisher_information``, just at
    window scale instead of one representative session. Deliberately scoped
    to a caller-supplied small subset (see module note above)."""
    results: dict[tuple[str, int], FisherInformationResult] = {}
    for model_id in model_ids:
        adapter = _MODEL_ADAPTERS[model_id]
        registry = adapter.load_registry(str(_config_path(model_id, mouse_id)))
        for window_dir in _list_window_dirs(model_id, mouse_id):
            manifest = common.read_json(window_dir / "window_manifest.json") or {}
            window_id = manifest.get("window_id")
            if window_id not in window_ids:
                continue
            winner = _window_winner_run(window_dir)
            if winner is None or winner.get("best_raw_parameters") is None:
                continue
            raw_point = torch.tensor(winner["best_raw_parameters"], dtype=torch.float64)
            sessions = _window_training_sessions(adapter, window_dir)
            results[(model_id, window_id)] = _hessian_at_point(
                model_id, f"window_{window_id:03d}", adapter, sessions, raw_point, registry,
                recorded_mean_nll=winner.get("recalculated_best_training_mean_nll"),
            )
    return results


def validate_restart_instability_against_hessian(
    mouse_id: str, model_ids: list[str], window_ids: list[int],
    restart_rank: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """For a small (model, window) subset, compares the cheap restart-
    instability percentile rank (``collect_restart_instability_rank``,
    constrained-space, zero new model evaluations) against a real Hessian's
    raw-SE-based percentile rank (raw/logit space) computed fresh for that
    same window, via Spearman rank correlation. Ordinal (Spearman, not the
    raw values) correlation is the right comparison across the two
    coordinate systems: for the component-wise-monotonic raw<->constrained
    transforms used throughout this codebase, a parameter's rank position is
    preserved by the transform even though its raw SE and constrained spread
    are not on the same scale.
    """
    if restart_rank is None:
        restart_rank = collect_restart_instability_rank(mouse_id)
    window_fisher = compute_window_fisher_information(mouse_id, model_ids, window_ids)

    rows = []
    for (model_id, window_id), result in window_fisher.items():
        raw_se_percentile = _sloppiest_to_stiffest_percentile(result.raw_se)
        hessian_rank = dict(zip(result.param_names, raw_se_percentile))

        restart_subset = restart_rank.loc[
            (restart_rank["model_id"] == model_id) & (restart_rank["window_id"] == window_id)
        ]
        restart_lookup = dict(zip(restart_subset["param_name"], restart_subset["percentile_rank"]))

        shared_params = [p for p in result.param_names if p in restart_lookup]
        if len(shared_params) < 2:
            continue
        hessian_values = pd.Series([hessian_rank[p] for p in shared_params])
        restart_values = pd.Series([restart_lookup[p] for p in shared_params])
        spearman_rho = hessian_values.corr(restart_values, method="spearman")

        rows.append({
            "model_id": model_id, "window_id": window_id, "n_params": len(shared_params),
            "spearman_rho": float(spearman_rho) if spearman_rho is not None else float("nan"),
            "hessian_seconds": result.hessian_seconds, "n_trials": result.n_trials,
        })
    columns = ["model_id", "window_id", "n_params", "spearman_rho", "hessian_seconds", "n_trials"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Page 8: cross-model parameter-identifiability comparison (two independent
# panels, deliberately not collapsed into one "trustworthiness" score -- see
# figures.py page 8 caption for why).
# ---------------------------------------------------------------------------

def collect_model_identifiability_summary(
    restart_rank: pd.DataFrame, trajectories: pd.DataFrame, fisher: dict[str, FisherInformationResult],
) -> pd.DataFrame:
    """One row per model: (a) median relative restart instability --
    ``spread_std / (|constrained value| + epsilon)`` across all (parameter,
    window) cells, an approximate scale-normalized restart-instability
    summary (epsilon-floored per Codex-consult finding: a raw SE/value ratio
    can explode for a parameter whose fitted value sits near zero -- this
    bounds but does not eliminate that inflation, so treat it as approximate,
    not exact); (b) "decades spanned" from the SAME single representative-
    session Hessian used on pages 2-3 (curvature ANISOTROPY at one session,
    not an all-windows aggregate -- computing per-window Hessians for all 13
    models x 30 windows was ruled out during planning on cost grounds, see
    ``compute_window_fisher_information``'s module note).
    """
    epsilon = 1e-3
    merged = restart_rank.merge(trajectories, on=["model_id", "window_id", "param_name"], how="inner")
    merged["relative_spread"] = merged["spread_std"] / (merged["value"].abs() + epsilon)

    rows = []
    for model_id in MODEL_ORDER:
        subset = merged.loc[merged["model_id"] == model_id]
        result = fisher.get(model_id)
        rows.append({
            "model_id": model_id,
            "median_relative_restart_instability": float(subset["relative_spread"].median()) if not subset.empty else float("nan"),
            "n_windows_restart": int(subset["window_id"].nunique()) if not subset.empty else 0,
            "decades_spanned": _decades_spanned(result) if result is not None else float("nan"),
            "hessian_session_id": result.session_id if result is not None else None,
        })
    return pd.DataFrame(rows)
