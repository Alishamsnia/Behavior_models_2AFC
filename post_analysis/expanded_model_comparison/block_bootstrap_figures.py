"""Two-page MC08 block-bootstrap model-comparison figure PDF.

Page 1 implements the procedure in
``Block_Bootstrap_Behavioral_Model_Comparison_Guide.pdf``: equal-session
held-out NLL, paired non-circular moving blocks, percentile intervals,
uncentred two-minimum bootstrap-tail p-values, and Holm correction.

Page 2 reports the deliberately separate hybrid sensitivity analysis:
centred/basic intervals, a null-centred two-sided block-bootstrap test, and
Holm-adjusted TOST practical-equivalence comparisons.  This module reads the
already audited expanded-comparison tables; it never fits or changes a model.

Run from the repository root:

    python -m post_analysis.expanded_model_comparison.block_bootstrap_figures
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd

from post_analysis.common import REPO_ROOT
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, MODEL_ORDER


ANALYSIS_ROOT = REPO_ROOT / "results_post_analysis" / "expanded_model_comparison"
DEFAULT_REPLICATES = 50_000
DEFAULT_PRIMARY_LENGTH = 5
DOCUMENT_LENGTHS = (3, 4, 5, 7, 10, 12)
HYBRID_LENGTHS = (3, 4, 5, 7, 9, 10, 12)
BASE_SEED = 20260917
ZERO_TOLERANCE = 1e-12
EQUIVALENCE_MARGIN = 0.005
EQUIVALENCE_BLOCK_LENGTH = 9

@dataclass(frozen=True)
class LoadedScores:
    matrix: np.ndarray
    model_ids: tuple[str, ...]
    test_ranks: np.ndarray
    trial_counts: np.ndarray


@dataclass(frozen=True)
class PairResult:
    first: str
    second: str
    observed: float
    ci95_low: float
    ci95_high: float
    raw_p: float
    holm_p: float
    edge_shift: float
    tail_lower: int
    tail_upper: int
    tail_zero: int


def _format_p_short(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return "<.001" if value < 0.001 else f"{value:.3f}".lstrip("0")


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm adjustment requires a nonempty vector")
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    work = np.where(finite, values, 1.0)
    order = np.argsort(work, kind="stable")
    ranked = (len(work) - np.arange(len(work))) * work[order]
    adjusted = np.empty_like(work)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(ranked))
    adjusted[~finite] = np.nan
    return adjusted


def _moving_block_indices(n: int, length: int, replicates: int) -> np.ndarray:
    if not 1 <= length < n:
        raise ValueError("block length must be between 1 and n-1")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    blocks = int(np.ceil(n / length))
    rng = np.random.default_rng(BASE_SEED + length)
    starts = rng.integers(0, n - length + 1, size=(replicates, blocks))
    offsets = np.arange(length)
    return (starts[..., None] + offsets).reshape(replicates, -1)[:, :n]


def _moving_block_weights(n: int, length: int) -> np.ndarray:
    """Exact expected non-circular row weights, including final trimming."""
    if not 1 <= length < n:
        raise ValueError("block length must be between 1 and n-1")
    full_blocks, remainder = divmod(n, length)
    candidate_blocks = n - length + 1
    weights = np.zeros(n, dtype=float)
    for start in range(candidate_blocks):
        weights[start:start + length] += full_blocks / (n * candidate_blocks)
        if remainder:
            weights[start:start + remainder] += 1 / (n * candidate_blocks)
    return weights


def _acf(values: np.ndarray, max_lag: int = 7) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    centered = series - series.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-28:
        return np.full(max_lag + 1, np.nan)
    return np.asarray(
        [1.0] + [float(centered[:-lag] @ centered[lag:]) / denominator for lag in range(1, max_lag + 1)]
    )


def _load_scores(mouse_id: str) -> LoadedScores:
    tables = ANALYSIS_ROOT / mouse_id / "tables"
    metrics_path = tables / "sliding_window_metrics.csv"
    status_path = tables / "model_completion_status.csv"
    if not metrics_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"Expanded comparison tables are missing for {mouse_id}")

    manifest_path = ANALYSIS_ROOT / mouse_id / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expanded comparison manifest is missing for {mouse_id}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("analysis") != "expanded_model_comparison"
        or manifest.get("mouse_id") != mouse_id
        or manifest.get("provisional") is not False
        or set(manifest.get("eligible_models", ())) != set(MODEL_ORDER)
    ):
        raise ValueError("analysis manifest is not the final complete MC08 comparison")

    status = pd.read_csv(status_path)
    required_status = {
        "model_id", "eligible_for_comparison", "complete_window_count", "source_snapshot_sha256",
    }
    if not required_status.issubset(status.columns):
        raise ValueError("model completion table lacks required columns")
    normalized_eligibility = status["eligible_for_comparison"].astype(str).str.strip().str.lower()
    if not normalized_eligibility.isin({"true", "false", "1", "0", "1.0", "0.0"}).all():
        raise ValueError("model eligibility contains non-boolean values")
    eligible = status[normalized_eligibility.isin({"true", "1", "1.0"})]
    eligible_ids = tuple(eligible["model_id"].astype(str))
    if len(eligible_ids) != len(set(eligible_ids)) or set(eligible_ids) != set(MODEL_ORDER):
        raise ValueError("eligible model set does not match the frozen 13-model registry")
    complete_counts = pd.to_numeric(eligible["complete_window_count"], errors="coerce")
    if not complete_counts.eq(30).all():
        raise ValueError("every model must have all 30 MC08 windows")
    manifest_snapshots = manifest.get("source_snapshots", {})
    status_snapshots = dict(zip(eligible["model_id"].astype(str), eligible["source_snapshot_sha256"].astype(str)))
    if status_snapshots != manifest_snapshots or any(len(value) != 64 for value in status_snapshots.values()):
        raise ValueError("audited source snapshots disagree with the analysis manifest")

    metrics = pd.read_csv(metrics_path)
    required = {
        "model_id", "window_id", "test_rank", "split_role", "valid_trial_count", "mean_nll",
    }
    if not required.issubset(metrics.columns):
        raise ValueError("sliding-window metrics lack required columns")
    test = metrics[metrics["split_role"].eq("test")].copy()
    test = test[test["model_id"].isin(MODEL_ORDER)]
    if len(test) != 30 * len(MODEL_ORDER):
        raise ValueError("expected exactly 30 test windows for each of 13 models")
    if test.duplicated(["window_id", "model_id"]).any():
        raise ValueError("duplicate model/window test rows")
    for column in ("window_id", "test_rank", "valid_trial_count", "mean_nll"):
        test[column] = pd.to_numeric(test[column], errors="coerce")
    if not np.isfinite(test[["window_id", "test_rank", "valid_trial_count", "mean_nll"]]).all().all():
        raise ValueError("test metrics contain nonfinite values")

    integer_columns = test[["window_id", "test_rank", "valid_trial_count"]].to_numpy(dtype=float)
    if not np.equal(integer_columns, np.floor(integer_columns)).all() or (integer_columns <= 0).any():
        raise ValueError("window, rank, and trial-count fields must be positive integers")
    identity = test[["window_id", "test_rank"]].drop_duplicates().sort_values("window_id")
    if len(identity) != 30 or identity["window_id"].duplicated().any() or identity["test_rank"].duplicated().any():
        raise ValueError("test windows do not map one-to-one to chronological ranks")
    if identity["window_id"].astype(int).tolist() != list(range(1, 31)):
        raise ValueError("MC08 test window IDs must be the frozen sequence 1..30")
    if identity["test_rank"].astype(int).tolist() != list(range(13, 43)):
        raise ValueError("MC08 window/rank mapping disagrees with the frozen 9/3/1 schedule")
    count_versions = test.groupby("window_id")["valid_trial_count"].nunique()
    if not count_versions.eq(1).all():
        raise ValueError("models were not scored on identical test-trial counts")

    ordered_windows = identity["window_id"].astype(int).tolist()
    wide = test.pivot(index="window_id", columns="model_id", values="mean_nll")
    wide = wide.reindex(index=ordered_windows, columns=MODEL_ORDER)
    if wide.isna().any().any():
        raise ValueError("incomplete finite model-by-window score matrix")
    counts = (
        test.drop_duplicates("window_id")
        .set_index("window_id")
        .reindex(ordered_windows)["valid_trial_count"]
        .to_numpy(dtype=int)
    )
    return LoadedScores(
        matrix=wide.to_numpy(dtype=float),
        model_ids=tuple(MODEL_ORDER),
        test_ranks=identity["test_rank"].to_numpy(dtype=int),
        trial_counts=counts,
    )


def _sampled_means(matrix: np.ndarray, length: int, replicates: int) -> np.ndarray:
    indices = _moving_block_indices(len(matrix), length, replicates)
    sampled = np.empty((replicates, matrix.shape[1]), dtype=float)
    for start in range(0, replicates, 2_000):
        stop = min(start + 2_000, replicates)
        sampled[start:stop] = matrix[indices[start:stop]].mean(axis=1)
    return sampled


def _document_pair_results(
    matrix: np.ndarray,
    model_ids: tuple[str, ...],
    length: int,
    replicates: int,
) -> tuple[list[PairResult], np.ndarray]:
    sampled = _sampled_means(matrix, length, replicates)
    observed_means = matrix.mean(axis=0)
    expected_means = _moving_block_weights(len(matrix), length) @ matrix
    temporary: list[dict] = []
    raw_values: list[float] = []
    for first_index, second_index in combinations(range(len(model_ids)), 2):
        # Positive delta means the first-named model has the lower NLL.
        observed = float(observed_means[second_index] - observed_means[first_index])
        boot = sampled[:, second_index] - sampled[:, first_index]
        boot = np.where(np.abs(boot) <= ZERO_TOLERANCE, 0.0, boot)
        low, high = np.quantile(boot, [0.025, 0.975], method="linear")
        lower = int(np.count_nonzero(boot <= 0))
        upper = int(np.count_nonzero(boot >= 0))
        zeros = int(np.count_nonzero(boot == 0))
        raw_p = min(1.0, 2.0 * (min(lower, upper) + 1) / (replicates + 1))
        difference = matrix[:, second_index] - matrix[:, first_index]
        if np.ptp(difference) <= ZERO_TOLERANCE:
            raw_p = 1.0 if abs(observed) <= ZERO_TOLERANCE else np.nan
        raw_values.append(raw_p)
        temporary.append({
            "first": model_ids[first_index],
            "second": model_ids[second_index],
            "observed": observed,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "raw_p": raw_p,
            "edge_shift": float(
                expected_means[second_index] - expected_means[first_index] - observed
            ),
            "tail_lower": lower,
            "tail_upper": upper,
            "tail_zero": zeros,
        })
    adjusted = _holm_adjust(np.asarray(raw_values))
    results = [PairResult(**row, holm_p=float(adjusted[index])) for index, row in enumerate(temporary)]
    return results, sampled


def _hybrid_pair_results(
    matrix: np.ndarray,
    model_ids: tuple[str, ...],
    length: int,
    replicates: int,
) -> tuple[list[PairResult], np.ndarray]:
    sampled = _sampled_means(matrix, length, replicates)
    observed_means = matrix.mean(axis=0)
    expected_means = _moving_block_weights(len(matrix), length) @ matrix
    temporary: list[dict] = []
    raw_values: list[float] = []
    for first_index, second_index in combinations(range(len(model_ids)), 2):
        observed = float(observed_means[second_index] - observed_means[first_index])
        boot = sampled[:, second_index] - sampled[:, first_index]
        error = boot - boot.mean()
        q025, q975 = np.quantile(error, [0.025, 0.975], method="linear")
        raw_p = (np.count_nonzero(np.abs(error) >= abs(observed)) + 1) / (replicates + 1)
        difference = matrix[:, second_index] - matrix[:, first_index]
        if np.ptp(difference) <= ZERO_TOLERANCE:
            raw_p = 1.0 if abs(observed) <= ZERO_TOLERANCE else np.nan
        raw_values.append(raw_p)
        temporary.append({
            "first": model_ids[first_index],
            "second": model_ids[second_index],
            "observed": observed,
            "ci95_low": float(observed - q975),
            "ci95_high": float(observed - q025),
            "raw_p": raw_p,
            "edge_shift": float(
                expected_means[second_index] - expected_means[first_index] - observed
            ),
            "tail_lower": int(np.count_nonzero(error <= -abs(observed))),
            "tail_upper": int(np.count_nonzero(error >= abs(observed))),
            "tail_zero": int(np.count_nonzero(np.abs(error) <= ZERO_TOLERANCE)),
        })
    adjusted = _holm_adjust(np.asarray(raw_values))
    results = [PairResult(**row, holm_p=float(adjusted[index])) for index, row in enumerate(temporary)]
    return results, sampled


def _result_lookup(results: list[PairResult]) -> dict[tuple[str, str], PairResult]:
    return {(row.first, row.second): row for row in results}


def _oriented_result(
    lookup: dict[tuple[str, str], PairResult], first: str, second: str,
) -> PairResult:
    if (first, second) in lookup:
        return lookup[(first, second)]
    row = lookup[(second, first)]
    return PairResult(
        first=first,
        second=second,
        observed=-row.observed,
        ci95_low=-row.ci95_high,
        ci95_high=-row.ci95_low,
        raw_p=row.raw_p,
        holm_p=row.holm_p,
        edge_shift=-row.edge_shift,
        tail_lower=row.tail_upper,
        tail_upper=row.tail_lower,
        tail_zero=row.tail_zero,
    )


def _model_label(model_id: str, max_chars: int = 28) -> str:
    """Return the canonical registry label, wrapped only at underscore boundaries."""
    name = MODEL_BY_ID[model_id].label
    if len(name) <= max_chars:
        return name
    lines: list[str] = []
    current = ""
    for part in name.split("_"):
        candidate = f"{current}_{part}" if current else part
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _ranking_panel(ax: plt.Axes, scores: LoadedScores) -> tuple[str, str]:
    means = scores.matrix.mean(axis=0)
    order = np.argsort(means)
    y = np.arange(len(order))
    colors = [MODEL_BY_ID[scores.model_ids[index]].color for index in order]
    for yi, index, color in zip(y, order, colors):
        oracle = MODEL_BY_ID[scores.model_ids[index]].oracle
        ax.scatter(
            means[index], yi, s=34, facecolors="white" if oracle else color,
            edgecolors=color, linewidths=1.5, zorder=3,
        )
    ax.set_yticks(
        y, [MODEL_BY_ID[scores.model_ids[index]].label for index in order], fontsize=6.5,
    )
    for tick, index in zip(ax.get_yticklabels(), order):
        if MODEL_BY_ID[scores.model_ids[index]].oracle:
            tick.set_color("#9C0006")
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mean held-out NLL (nats/choice; lower is better)", fontsize=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_title("A  Observed held-out ranking", loc="left", fontweight="bold", fontsize=9)
    ax.text(
        0.99, 0.01, "Hollow markers / dark-red names = oracle benchmarks",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
    )
    return scores.model_ids[order[0]], scores.model_ids[order[1]]


def _pair_matrix_panel(
    ax: plt.Axes,
    model_ids: tuple[str, ...],
    results: list[PairResult],
    title: str,
    label_fontsize: float = 4.6,
) -> None:
    lookup = _result_lookup(results)
    n = len(model_ids)
    values = np.full((n, n), np.nan)
    raw_p_values = np.full((n, n), np.nan)
    stars = np.zeros((n, n), dtype=bool)
    for row_index in range(n):
        for col_index in range(row_index + 1, n):
            row = _oriented_result(lookup, model_ids[row_index], model_ids[col_index])
            values[row_index, col_index] = row.observed
            raw_p_values[row_index, col_index] = row.raw_p
            stars[row_index, col_index] = row.holm_p < 0.05
    display_values = 1_000 * values
    finite = np.abs(display_values[np.isfinite(display_values)])
    maximum = max(float(finite.max(initial=0.01)), 0.01)
    image = ax.imshow(
        display_values, cmap="RdBu", norm=SymLogNorm(linthresh=2.0, vmin=-maximum, vmax=maximum),
        interpolation="nearest", aspect="auto",
    )
    for row_index in range(n):
        for col_index in range(row_index + 1, n):
            value = values[row_index, col_index]
            holm_marker = "*" if stars[row_index, col_index] else ""
            raw_p_text = _format_p_short(raw_p_values[row_index, col_index])
            label = f"{1000 * value:+.1f}\n{holm_marker}\n{raw_p_text}"
            text_color = "white" if abs(value) > 0.025 else "black"
            ax.text(
                col_index, row_index, label, ha="center", va="center",
                fontsize=label_fontsize, linespacing=1.05, color=text_color,
            )
    labels = [_model_label(model_id, 24) for model_id in model_ids]
    ax.set_xticks(np.arange(n), labels, rotation=55, ha="right", fontsize=6.3)
    ax.set_yticks(np.arange(n), labels, fontsize=6.3)
    for tick, model_id in zip(ax.get_xticklabels(), model_ids):
        if MODEL_BY_ID[model_id].oracle:
            tick.set_color("#9C0006")
            tick.set_fontweight("bold")
    for tick, model_id in zip(ax.get_yticklabels(), model_ids):
        if MODEL_BY_ID[model_id].oracle:
            tick.set_color("#9C0006")
            tick.set_fontweight("bold")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Row advantage x1,000 (nats/choice)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.5)
    ax.text(
        0.0, -0.28,
        "Upper triangle only. Each cell: row advantage x1,000; Holm significance (* p<0.05); raw uncorrected p. "
        "Positive favors the row; dark-red names are oracle models.",
        transform=ax.transAxes, fontsize=6.8, va="top",
    )


def _trajectory_panel(
    ax: plt.Axes, scores: LoadedScores, leader: str, runner_up: str,
) -> None:
    first = scores.model_ids.index(leader)
    second = scores.model_ids.index(runner_up)
    difference = scores.matrix[:, second] - scores.matrix[:, first]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(difference.mean(), color="#666666", linewidth=0.8, linestyle=":")
    ax.plot(scores.test_ranks, difference, color=MODEL_BY_ID[leader].color, marker="o", ms=3, lw=1)
    ax.set_xlabel("Chronological test rank 13-42 (30 test sessions)", fontsize=8)
    ax.set_ylabel("NLL(runner-up) - NLL(leader)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#e6e6e6", linewidth=0.5)
    ax.set_title(
        "C  Top-two session difference",
        loc="left", fontweight="bold", fontsize=9,
    )


def _acf_panel(
    ax: plt.Axes, scores: LoadedScores, leader: str, runner_up: str,
) -> None:
    first = scores.model_ids.index(leader)
    second = scores.model_ids.index(runner_up)
    difference = scores.matrix[:, second] - scores.matrix[:, first]
    acf = _acf(difference, 7)
    bound = 1.96 / np.sqrt(len(difference))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(bound, color="#777777", lw=0.7, ls="--")
    ax.axhline(-bound, color="#777777", lw=0.7, ls="--")
    ax.bar(np.arange(1, 8), acf[1:], color="#777777", width=0.65)
    ax.set_xticks(np.arange(1, 8))
    ax.set_ylim(-0.48, 0.48)
    ax.set_xlabel("Lag (test sessions)", fontsize=7)
    ax.set_ylabel("ACF", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
    ax.set_title("ACF diagnostic; dashed = +/-1.96/sqrt(W)", loc="left", fontsize=7.5)


def _document_sensitivity_panel(
    ax: plt.Axes,
    scores: LoadedScores,
    leader: str,
    runner_up: str,
    replicates: int,
) -> None:
    rows = []
    for length in DOCUMENT_LENGTHS:
        results, _ = _document_pair_results(scores.matrix, scores.model_ids, length, replicates)
        row = _oriented_result(_result_lookup(results), leader, runner_up)
        rows.append((length, row))
    positions = np.arange(len(rows))
    observed = np.asarray([row.observed for _, row in rows])
    low = np.asarray([row.ci95_low for _, row in rows])
    high = np.asarray([row.ci95_high for _, row in rows])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.errorbar(
        positions, observed, yerr=[observed - low, high - observed], fmt="o-",
        color=MODEL_BY_ID[leader].color, capsize=3, lw=1,
    )
    for position, (_, row) in zip(positions, rows):
        ax.annotate(
            f"pH={row.holm_p:.3g}\npR={row.raw_p:.3g}\nedge={row.edge_shift:+.4f}",
            (position, high[position]), xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=5.3,
        )
    ax.set_xticks(positions, [str(length) for length, _ in rows])
    ax.set_xlabel("Moving-block length L", fontsize=8)
    ax.set_ylabel("Leader advantage (nats/choice)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
    ax.margins(y=0.28)
    ax.set_title(
        "D  Guide-method block-length sensitivity", loc="left",
        fontweight="bold", fontsize=9, pad=12,
    )


def _document_page(scores: LoadedScores, replicates: int, primary_length: int) -> plt.Figure:
    primary_results, _ = _document_pair_results(
        scores.matrix, scores.model_ids, primary_length, replicates,
    )
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(
        5, 2, width_ratios=(0.86, 1.34),
        height_ratios=(1.0, 0.72, 0.42, 0.72, 0.13),
    )
    ranking_ax = figure.add_subplot(grid[0, 0])
    leader, runner_up = _ranking_panel(ranking_ax, scores)
    top_two = _oriented_result(_result_lookup(primary_results), leader, runner_up)
    ranking_ax.text(
        0.99, 0.96,
        f"Top two: Holm p={top_two.holm_p:.3g} (raw p={top_two.raw_p:.3g}); superiority not established",
        transform=ranking_ax.transAxes, ha="right", va="top", fontsize=6.3,
    )
    matrix_ax = figure.add_subplot(grid[:4, 1])
    _pair_matrix_panel(
        matrix_ax, scores.model_ids, primary_results,
        "B  Guide method: all 78 paired comparisons",
    )
    trajectory_ax = figure.add_subplot(grid[1, 0])
    _trajectory_panel(trajectory_ax, scores, leader, runner_up)
    acf_ax = figure.add_subplot(grid[2, 0])
    _acf_panel(acf_ax, scores, leader, runner_up)
    sensitivity_ax = figure.add_subplot(grid[3, 0])
    _document_sensitivity_panel(sensitivity_ax, scores, leader, runner_up, replicates)
    footer_ax = figure.add_subplot(grid[4, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"W=30 sessions; 13 models; B={replicates:,}; primary L={primary_length}; equal-session weighting; "
        "pointwise percentile 95% CIs; uncentred 2x-min tails; Holm family=78.\n"
        "Edge shift = exact expected bootstrap effect minus observed effect. "
        "Within-mouse exploratory inference; no model was retrained.",
        ha="center", va="center", fontsize=7.2, wrap=True,
    )
    figure.suptitle(
        "MC08 held-out comparison - paired moving-block bootstrap specified by the guide",
        fontsize=13, fontweight="bold",
    )
    return figure


def _reference_forest_panel(
    ax: plt.Axes,
    scores: LoadedScores,
    results: list[PairResult],
    reference: str,
) -> None:
    lookup = _result_lookup(results)
    rows = []
    for model in scores.model_ids:
        if model == reference:
            continue
        row = _oriented_result(lookup, reference, model)
        rows.append((model, row))
    rows.sort(key=lambda item: item[1].observed, reverse=True)
    y = np.arange(len(rows))
    observed = np.asarray([row.observed for _, row in rows])
    low = np.asarray([row.ci95_low for _, row in rows])
    high = np.asarray([row.ci95_high for _, row in rows])
    colors = [MODEL_BY_ID[model].color for model, _ in rows]
    for yi, value, lo, hi, color, (_, row) in zip(y, observed, low, high, colors, rows):
        ax.errorbar(value, yi, xerr=[[value - lo], [hi - value]], fmt="o", color=color, capsize=2, ms=4)
        ax.text(hi, yi, f"  Holm p={row.holm_p:.3g} (raw {row.raw_p:.3g})", va="center", fontsize=6.2)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yticks(y, [MODEL_BY_ID[model].label for model, _ in rows], fontsize=6.2)
    for tick, (model, _) in zip(ax.get_yticklabels(), rows):
        if MODEL_BY_ID[model].oracle:
            tick.set_color("#9C0006")
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Reference advantage: NLL(candidate) - NLL(reference); symmetric-log axis", fontsize=7,
    )
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.5)
    ax.set_title(
        f"B  Pointwise basic 95% CIs vs {MODEL_BY_ID[reference].label}",
        loc="left", fontweight="bold", fontsize=9,
    )


def _equivalence_rows(
    scores: LoadedScores,
    sampled: np.ndarray,
    reference: str,
    replicates: int,
) -> list[dict]:
    reference_index = scores.model_ids.index(reference)
    rows = []
    raw = []
    for index, model in enumerate(scores.model_ids):
        if model == reference:
            continue
        observed = float(scores.matrix[:, index].mean() - scores.matrix[:, reference_index].mean())
        boot = sampled[:, index] - sampled[:, reference_index]
        error = boot - boot.mean()
        q05, q95 = np.quantile(error, [0.05, 0.95], method="linear")
        ci_low, ci_high = observed - q95, observed - q05
        lower_null = -EQUIVALENCE_MARGIN + error
        upper_null = EQUIVALENCE_MARGIN + error
        p_lower = (np.count_nonzero(lower_null >= observed) + 1) / (replicates + 1)
        p_upper = (np.count_nonzero(upper_null <= observed) + 1) / (replicates + 1)
        p_value = max(p_lower, p_upper)
        raw.append(p_value)
        rows.append({
            "model": model, "observed": observed, "ci90_low": float(ci_low),
            "ci90_high": float(ci_high), "raw_p": float(p_value),
        })
    adjusted = _holm_adjust(np.asarray(raw))
    for index, value in enumerate(adjusted):
        rows[index]["holm_p"] = float(value)
    return rows


def _equivalence_panel(ax: plt.Axes, rows: list[dict], reference: str) -> None:
    rows = sorted(rows, key=lambda row: row["observed"])
    y = np.arange(len(rows))
    ax.axvspan(-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN, color="#d9ead3", alpha=0.7)
    ax.axvline(-EQUIVALENCE_MARGIN, color="#4b8b3b", lw=0.8, ls="--")
    ax.axvline(EQUIVALENCE_MARGIN, color="#4b8b3b", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xscale("symlog", linthresh=0.01)
    for yi, row in zip(y, rows):
        equivalent = row["holm_p"] < 0.05
        color = "#2a7f62" if equivalent else "#777777"
        ax.errorbar(
            row["observed"], yi,
            xerr=[[row["observed"] - row["ci90_low"]], [row["ci90_high"] - row["observed"]]],
            fmt="o", color=color, capsize=2, ms=4,
        )
        ax.text(row["ci90_high"], yi, f"  Holm p={row['holm_p']:.3g} (raw {row['raw_p']:.3g})", va="center", fontsize=6.2)
    ax.set_yticks(y, [MODEL_BY_ID[row["model"]].label for row in rows], fontsize=6.2)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if MODEL_BY_ID[row["model"]].oracle:
            tick.set_color("#9C0006")
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Candidate - reference NLL (90% basic CI; symmetric-log axis)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.5)
    ax.set_title(
        f"C  TOST vs {MODEL_BY_ID[reference].label} (+/-{EQUIVALENCE_MARGIN:.3f}; L={EQUIVALENCE_BLOCK_LENGTH})",
        loc="left", fontweight="bold", fontsize=9,
    )
    if not any(row["holm_p"] < 0.05 for row in rows):
        ax.text(
            0.02, 0.04, "No equivalence established",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
            color="#555555",
        )


def _hybrid_sensitivity_panel(
    ax: plt.Axes,
    scores: LoadedScores,
    leader: str,
    runner_up: str,
    replicates: int,
) -> None:
    rows = []
    for length in HYBRID_LENGTHS:
        results, _ = _hybrid_pair_results(scores.matrix, scores.model_ids, length, replicates)
        row = _oriented_result(_result_lookup(results), leader, runner_up)
        rows.append((length, row))
    positions = np.arange(len(rows))
    observed = np.asarray([row.observed for _, row in rows])
    low = np.asarray([row.ci95_low for _, row in rows])
    high = np.asarray([row.ci95_high for _, row in rows])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.errorbar(
        positions, observed, yerr=[observed - low, high - observed], fmt="o-",
        color=MODEL_BY_ID[leader].color, capsize=3, lw=1,
    )
    for position, (_, row) in zip(positions, rows):
        ax.annotate(
            f"pH={row.holm_p:.3g}\npR={row.raw_p:.3g}", (position, high[position]), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=5.8,
        )
    ax.set_xticks(positions, [str(length) for length, _ in rows])
    ax.set_xlabel("Moving-block length L", fontsize=7)
    ax.set_ylabel("Leader advantage (nats/choice)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
    ax.margins(y=0.25)
    ax.set_title(
        "D  Pairwise top-two sensitivity (not TOST)", loc="left",
        fontweight="bold", fontsize=9, pad=10,
    )


def _hybrid_page(scores: LoadedScores, replicates: int, primary_length: int) -> plt.Figure:
    means = scores.matrix.mean(axis=0)
    order = np.argsort(means)
    leader = scores.model_ids[order[0]]
    runner_up = scores.model_ids[order[1]]
    results, _ = _hybrid_pair_results(
        scores.matrix, scores.model_ids, primary_length, replicates,
    )
    equivalence_sampled = _sampled_means(
        scores.matrix, EQUIVALENCE_BLOCK_LENGTH, replicates,
    )
    equivalence = _equivalence_rows(scores, equivalence_sampled, leader, replicates)

    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(
        4, 2, width_ratios=(1.3, 0.7), height_ratios=(1.0, 0.92, 0.72, 0.12),
    )
    matrix_ax = figure.add_subplot(grid[:3, 0])
    _pair_matrix_panel(
        matrix_ax, scores.model_ids, results,
        "A  Hybrid null-centred test: all 78 model pairs",
        label_fontsize=4.2,
    )
    forest_ax = figure.add_subplot(grid[0, 1])
    _reference_forest_panel(forest_ax, scores, results, leader)
    equivalence_ax = figure.add_subplot(grid[1, 1])
    _equivalence_panel(equivalence_ax, equivalence, leader)
    sensitivity_ax = figure.add_subplot(grid[2, 1])
    _hybrid_sensitivity_panel(sensitivity_ax, scores, leader, runner_up, replicates)
    footer_ax = figure.add_subplot(grid[3, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"W=30 sessions; equal-session weighting; all 13 models including 2 oracle benchmarks; B={replicates:,}; pairwise primary L={primary_length}; "
        "basic 95% CIs; two-sided null-centred tails; Holm family=78.\n"
        f"TOST L={EQUIVALENCE_BLOCK_LENGTH}, margin +/-0.005, 90% basic CIs, Holm family=12 post-selection leader contrasts. "
        "Within-mouse exploratory only; top two show neither superiority nor equivalence.",
        ha="center", va="center", fontsize=7.0, wrap=True,
    )
    figure.suptitle(
        "MC08 held-out comparison - hybrid centred bootstrap and practical equivalence",
        fontsize=13, fontweight="bold",
    )
    return figure


def generate(mouse_id: str = "MC08", replicates: int = DEFAULT_REPLICATES) -> Path:
    if mouse_id != "MC08":
        raise ValueError("This two-page figure is defined for the complete MC08 expanded comparison")
    if replicates < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replicates")
    scores = _load_scores(mouse_id)
    figures_dir = ANALYSIS_ROOT / mouse_id / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / "fig25_block_bootstrap_document_and_hybrid.pdf"
    pending = figures_dir / ".fig25_block_bootstrap_document_and_hybrid.pending.pdf"
    style = {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
    pending.unlink(missing_ok=True)
    try:
        with plt.rc_context(style), PdfPages(
            pending,
            metadata={
                "Title": "MC08 block-bootstrap model comparison: guide and hybrid methods",
                "Author": "Bayesian_model_v2 post-analysis",
                "Subject": "Two-page held-out comparison of 13 behavioural models",
            },
        ) as pdf:
            document = _document_page(scores, replicates, DEFAULT_PRIMARY_LENGTH)
            pdf.savefig(document)
            plt.close(document)
            hybrid = _hybrid_page(scores, replicates, DEFAULT_PRIMARY_LENGTH)
            pdf.savefig(hybrid)
            plt.close(hybrid)
        pending.replace(output)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC08")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args()
    print(generate(args.mouse, args.replicates))


if __name__ == "__main__":
    main()
