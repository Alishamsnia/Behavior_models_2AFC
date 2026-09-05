"""Shared paths, model registry metadata, and small IO helpers for post-analysis.

Model parameter names are hard-coded from the frozen scientific contracts in
``docs/models/M{0..4}_*.md`` and ``README.md`` (each model's pooled parameter
vector order is stated explicitly there and protected by the project's own
tests/audits). The result JSON files themselves only store parameter values as
plain arrays, so the names are attached here rather than re-derived per run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_SLIDING_ROOT = REPO_ROOT / "results_sliding_window"
RESULTS_SINGLE_ROOT = REPO_ROOT / "results_single_session"
POST_ANALYSIS_ROOT = REPO_ROOT / "results_post_analysis"

WORKFLOW = "full_session_matched_v1"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    dir_name: str
    label: str
    param_names: tuple
    color: str


# Colorblind-safe qualitative palette (Okabe-Ito), one fixed color per model
# reused across every figure in this package.
MODEL_SPECS: dict[str, ModelSpec] = {
    spec.model_id: spec
    for spec in [
        ModelSpec(
            "M0", "M0_sensory_v1", "M0 sensory",
            ("w", "s0", "ell_short", "ell_long", "kappa"),
            "#0072B2",
        ),
        ModelSpec(
            "M1", "M1_hmm_true_v1", "M1 HMM-true",
            ("w", "s0", "ell_short", "ell_long", "kappa", "ah"),
            "#009E73",
        ),
        ModelSpec(
            "M2", "M2_hmm_static_v1", "M2 HMM-static",
            ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0"),
            "#E69F00",
        ),
        ModelSpec(
            "M3", "M3_hmm_delta_v1", "M3 HMM-delta",
            (
                "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0",
                "alpha_theta",
            ),
            "#D55E00",
        ),
        ModelSpec(
            "M4", "M4_hsmm_delta_v1", "M4 HSMM-delta",
            (
                "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0",
                "alpha_theta", "bh",
            ),
            "#CC79A7",
        ),
    ]
}

MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4"]

# Parameters shared by every model in the nested family, in a fixed plotting
# order. Every model fits these five coordinates with the same meaning.
SHARED_PARAM_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa")

# Filenames used for the per-split/per-session summary table. Differs by
# model because the two model families (M0/M1/M2 vs M3/M4) named the file
# differently even though the schema is identical.
_SPLIT_SESSION_METRIC_FILENAMES = (
    "split_and_session_metrics.csv",
    "prefix_and_heldout_metrics.csv",
)


def sliding_window_dir(model_id: str, mouse_id: str) -> Path:
    spec = MODEL_SPECS[model_id]
    return RESULTS_SLIDING_ROOT / spec.dir_name / mouse_id / WORKFLOW


def single_session_dir(model_id: str, mouse_id: str) -> Path:
    spec = MODEL_SPECS[model_id]
    return RESULTS_SINGLE_ROOT / spec.dir_name / mouse_id / WORKFLOW


def available_mice() -> list[str]:
    """Mice that have at least one model's sliding-window results on disk."""
    mice: set[str] = set()
    for spec in MODEL_SPECS.values():
        model_dir = RESULTS_SLIDING_ROOT / spec.dir_name
        if not model_dir.is_dir():
            continue
        for child in model_dir.iterdir():
            if (child / WORKFLOW).is_dir():
                mice.add(child.name)
    return sorted(mice)


def list_window_dirs(model_id: str, mouse_id: str) -> list[Path]:
    root = sliding_window_dir(model_id, mouse_id)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("window_"))


def list_session_dirs(model_id: str, mouse_id: str) -> list[Path]:
    root = single_session_dir(model_id, mouse_id)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_"))


def read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)


def read_split_session_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load the per-split/per-session metrics table for one fitted run.

    Tries both filenames used across the model family (see module docstring).
    """
    for filename in _SPLIT_SESSION_METRIC_FILENAMES:
        candidate = run_dir / filename
        if candidate.is_file():
            return pd.read_csv(candidate)
    return None


def find_multistart_winner(window_dir: Path) -> Optional[dict]:
    summary = read_json(window_dir / "starts" / "multistart_summary.json")
    if summary is None:
        return None
    winner_id = summary.get("winner_start_id")
    if winner_id is None:
        return None
    for run in summary.get("runs", []):
        if run.get("start_id") == winner_id:
            return run
    return None


def ensure_output_dirs(analysis_name: str, mouse_id: str) -> tuple[Path, Path]:
    base = POST_ANALYSIS_ROOT / analysis_name / mouse_id
    tables_dir = base / "tables"
    figures_dir = base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir
