#!/usr/bin/env python3
"""Per-window diagnostics for M2_dynamic_action_kernel_bias_v1 (model E).

Read-only: inspects saved sliding-window results and prints/writes one row
per window. It answers the questions specific to E's tenth coordinate:

- convergence: how many of the eight starts converged, and which checks failed;
- the fitted learning rate alpha = sigmoid(a) and eta = sigmoid(eta_logit),
  and how far alpha moves away from the parent's fixed alpha = 1;
- weak identification: the spread of alpha / eta among converged starts whose
  training NLL is within ``--ridge-tolerance`` of the winner;
- nesting: E contains the parent M2_dynamic_previous_action_bias_v1 at
  alpha = 1, so a correctly optimized E has training NLL <= the parent's in the
  same window (up to the optimizer tolerance). A positive gap flags an
  optimization shortfall, not a model property. Also reported against
  M2_dynamic_block_length_v1 (eta = 0 boundary) and, out of sample, the
  validation/test NLL differences.

Usage:
    python scripts/report_action_kernel_bias_diagnostics.py --mouse MC08
    python scripts/report_action_kernel_bias_diagnostics.py --mouse MC08 \
        --e-root <dir containing M2_dynamic_action_kernel_bias_v1/> --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SLIDING_ROOT = REPOSITORY_ROOT / "results_sliding_window"
WORKFLOW = "full_session_matched_v1"

E_MODEL = "M2_dynamic_action_kernel_bias_v1"
COMPARATORS = {
    "parent": "M2_dynamic_previous_action_bias_v1",  # alpha = 1 boundary
    "block": "M2_dynamic_block_length_v1",  # eta = 0 boundary
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _windows(root: Path, model: str, mouse: str) -> dict[int, Path]:
    base = root / model / mouse / WORKFLOW
    found: dict[int, Path] = {}
    for path in sorted(base.glob("window_*")) if base.is_dir() else []:
        found[int(path.name.split("_")[1])] = path
    return found


def _split_nll(window: Path) -> dict[str, float]:
    path = window / "split_and_session_metrics.csv"
    if not path.is_file():
        return {}
    with path.open() as handle:
        return {
            row["split_role"]: float(row["mean_nll"])
            for row in csv.DictReader(handle)
            if row.get("session_id") == "ALL_SESSIONS"
        }


def _fit(window: Path) -> dict[str, Any] | None:
    path = window / "fit_result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _status(window: Path) -> str:
    path = window / "window_manifest.json"
    return json.loads(path.read_text()).get("status", "?") if path.is_file() else "missing"


def _winner_train_nll(fit: dict[str, Any] | None) -> float | None:
    if not fit or fit.get("winner_start_id") is None:
        return None
    for run in fit["runs"]:
        if run["start_id"] == fit["winner_start_id"]:
            return float(run["recalculated_best_training_mean_nll"])
    return None


def _e_row(window_id: int, window: Path, tolerance: float) -> dict[str, Any]:
    fit = _fit(window)
    row: dict[str, Any] = {"window": window_id, "status": _status(window)}
    if fit is None:
        return row
    runs = [r for r in fit["runs"] if r.get("exception_message") is None]
    converged = [r for r in runs if r.get("converged")]
    failed_checks: dict[str, int] = {}
    for run in runs:
        for name, ok in run.get("convergence_checks") or []:
            if not ok:
                failed_checks[name] = failed_checks.get(name, 0) + 1
    row.update(
        converged=f"{len(converged)}/{len(fit['runs'])}",
        failed_checks=";".join(f"{k}x{v}" for k, v in sorted(failed_checks.items())),
        winner=fit.get("winner_start_id"),
        selection=fit.get("selection_status"),
        ridge_flag=fit.get("identifiability_ridge_detected"),
    )
    winner = next((r for r in runs if r["start_id"] == fit.get("winner_start_id")), None)
    if winner is None:
        return row
    best = winner["best_constrained_parameters"]
    nll = float(winner["recalculated_best_training_mean_nll"])
    row.update(
        alpha=_sigmoid(best[9]),
        eta=_sigmoid(best[8]),
        a=best[9],
        train_nll=nll,
        grad_norm=winner.get("gradient_norm_at_final"),
        tail_ok=winner.get("tail_occupancy_acceptable"),
    )
    near = [
        r for r in converged
        if float(r["recalculated_best_training_mean_nll"]) - nll <= tolerance
    ]
    alphas = [_sigmoid(r["best_constrained_parameters"][9]) for r in near]
    etas = [_sigmoid(r["best_constrained_parameters"][8]) for r in near]
    row.update(
        near_optimal_starts=len(near),
        alpha_range=f"{min(alphas):.3f}-{max(alphas):.3f}",
        eta_range=f"{min(etas):.3f}-{max(etas):.3f}",
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mouse", required=True)
    parser.add_argument("--e-root", type=Path, default=SLIDING_ROOT,
                        help="Sliding root holding E's results (default: results_sliding_window)")
    parser.add_argument("--comparator-root", type=Path, default=SLIDING_ROOT)
    parser.add_argument("--ridge-tolerance", type=float, default=1e-3,
                        help="Training mean-NLL tolerance for 'near-optimal' starts (default 1e-3)")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    e_windows = _windows(args.e_root, E_MODEL, args.mouse)
    if not e_windows:
        raise SystemExit(f"no {E_MODEL} windows for {args.mouse} under {args.e_root}")
    comparator_windows = {
        key: _windows(args.comparator_root, model, args.mouse) for key, model in COMPARATORS.items()
    }

    rows = []
    for window_id, window in e_windows.items():
        row = _e_row(window_id, window, args.ridge_tolerance)
        e_split = _split_nll(window)
        for split in ("validation", "test"):
            row[f"{split}_nll"] = e_split.get(split)
        for key, windows in comparator_windows.items():
            other = windows.get(window_id)
            if other is None:
                continue
            other_train = _winner_train_nll(_fit(other))
            other_split = _split_nll(other)
            if other_train is not None and row.get("train_nll") is not None:
                # Positive train gap vs the parent = E failed to reach its own alpha=1 boundary.
                row[f"train_minus_{key}"] = row["train_nll"] - other_train
            for split in ("validation", "test"):
                if split in other_split and row.get(f"{split}_nll") is not None:
                    row[f"{split}_minus_{key}"] = row[f"{split}_nll"] - other_split[split]
        rows.append(row)

    columns: list[str] = []
    for row in rows:
        columns.extend(c for c in row if c not in columns)
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def fmt(value: Any) -> str:
        return f"{value:.4g}" if isinstance(value, float) else ("" if value is None else str(value))

    print("\t".join(columns))
    for row in rows:
        print("\t".join(fmt(row.get(c)) for c in columns))

    nested = [r["train_minus_parent"] for r in rows if "train_minus_parent" in r]
    if nested:
        violations = [g for g in nested if g > 1e-6]
        print(f"\nnesting check vs parent (alpha=1): {len(violations)}/{len(nested)} windows "
              f"with E train NLL above the parent's (max gap {max(nested):.3g})")
    alphas = [r["alpha"] for r in rows if "alpha" in r]
    if alphas:
        alphas.sort()
        print(f"alpha across windows: median {alphas[len(alphas) // 2]:.3f}, "
              f"range {alphas[0]:.3f}-{alphas[-1]:.3f}, "
              f"{sum(a > 0.95 for a in alphas)}/{len(alphas)} windows with alpha > 0.95")


if __name__ == "__main__":
    main()
