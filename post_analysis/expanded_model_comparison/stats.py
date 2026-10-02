"""Paired descriptive statistics for overlapping chronological windows."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from post_analysis.model_comparison.stats import holm_adjust

from .config import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EQUIVALENCE_ALPHA,
    EQUIVALENCE_BLOCK_LENGTH,
    EQUIVALENCE_MARGIN_NLL,
    MODEL_BY_ID,
    MODEL_FAMILY_ORDER,
    SPLIT_ROLES,
    TIE_TOLERANCE_NLL,
)


EVENT_SELECTION_DOMAINS = (
    "rare_long_in_short_block",
    "rare_short_in_long_block",
    "short_to_long_short_trials",
    "long_to_short_short_trials",
    "short_to_long_long_trials",
    "long_to_short_long_trials",
)
EVENT_SELECTION_LABELS = {
    "rare_long_in_short_block": "Rare long\nin short block",
    "rare_short_in_long_block": "Rare short\nin long block",
    "short_to_long_short_trials": "Short->long\nshort trials",
    "long_to_short_short_trials": "Long->short\nshort trials",
    "short_to_long_long_trials": "Short->long\nlong trials",
    "long_to_short_long_trials": "Long->short\nlong trials",
}


def complete_matrix(metrics: pd.DataFrame, model_order: tuple[str, ...]) -> pd.DataFrame:
    subset = metrics.copy()
    subset["mean_nll"] = pd.to_numeric(subset["mean_nll"], errors="coerce")
    subset.loc[~np.isfinite(subset["mean_nll"]), "mean_nll"] = np.nan
    counts = subset.groupby(["window_id", "model_id"]).size()
    if (counts > 1).any():
        raise ValueError("Duplicate model/window rows in ranking metrics")
    return subset.pivot(index="window_id", columns="model_id", values="mean_nll").reindex(columns=model_order).dropna()


def strict_complete_matrix(
    metrics: pd.DataFrame,
    model_order: tuple[str, ...],
    *,
    expected_window_count: int | None = None,
) -> pd.DataFrame:
    """Return a complete paired matrix, failing instead of silently dropping windows."""
    expected_windows = set(metrics["window_id"].unique())
    matrix = complete_matrix(metrics, model_order)
    if set(matrix.index) != expected_windows:
        missing = sorted(expected_windows - set(matrix.index))
        raise ValueError(f"Figure 0 requires complete finite model data for every window; incomplete windows: {missing}")
    if matrix.empty:
        raise ValueError("Figure 0 equivalence analysis has no complete windows")
    if expected_window_count is not None and len(matrix) != expected_window_count:
        raise ValueError(
            f"Figure 0 requires {expected_window_count} complete windows; found {len(matrix)}"
        )
    return matrix


def moving_block_indices(n: int, replicates: int = BOOTSTRAP_REPLICATES, block_length: int = BOOTSTRAP_BLOCK_LENGTH) -> np.ndarray:
    if n < 1:
        return np.empty((0, 0), dtype=int)
    length = min(block_length, n)
    blocks = int(np.ceil(n / length))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = rng.integers(0, n - length + 1, size=(replicates, blocks))
    offsets = np.arange(length)
    return (starts[..., None] + offsets).reshape(replicates, -1)[:, :n]


def _tolerance_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, treating adjacent sorted NLLs within tolerance as tied."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        anchor = values[order[start]]
        while stop < len(order) and values[order[stop]] - anchor <= TIE_TOLERANCE_NLL:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def ranking_table(matrix: pd.DataFrame, labels: dict[str, str], oracle: dict[str, bool], scope: str) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    values = matrix.to_numpy(float)
    sampled = values[moving_block_indices(len(matrix))].mean(axis=1)
    ranks = np.apply_along_axis(_tolerance_ranks, 1, sampled)
    minima = sampled.min(axis=1, keepdims=True)
    best_mask = sampled - minima <= TIE_TOLERANCE_NLL
    best_credit = best_mask / best_mask.sum(axis=1, keepdims=True)
    rows = []
    means = values.mean(axis=0)
    for index, model_id in enumerate(matrix.columns):
        lower, upper = np.quantile(sampled[:, index], [0.025, 0.975])
        rows.append({
            "ranking_scope": scope,
            "model_id": model_id,
            "model_label": labels[model_id],
            "oracle_benchmark": oracle[model_id],
            "n_windows": len(matrix),
            "mean_window_nll": means[index],
            "median_window_nll": np.median(values[:, index]),
            "moving_block_ci_lower": lower,
            "moving_block_ci_upper": upper,
            "bootstrap_probability_best": best_credit[:, index].mean(),
            "bootstrap_mean_rank": ranks[:, index].mean(),
            "bootstrap_block_length": min(BOOTSTRAP_BLOCK_LENGTH, len(matrix)),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        })
    table = pd.DataFrame(rows).sort_values(["mean_window_nll", "model_id"]).reset_index(drop=True)
    observed = table["mean_window_nll"].to_numpy(float)
    rank_values = []
    current_rank = 1
    group_start = 0
    for index in range(len(observed)):
        if index and observed[index] - observed[group_start] > TIE_TOLERANCE_NLL:
            current_rank = index + 1
            group_start = index
        rank_values.append(current_rank)
    table.insert(2, "rank", rank_values)
    return table


def pairwise_table(matrix: pd.DataFrame, split_role: str, final_all_models: bool) -> pd.DataFrame:
    rows = []
    for model_a, model_b in combinations(matrix.columns, 2):
        a = matrix[model_a].to_numpy(float)
        b = matrix[model_b].to_numpy(float)
        difference = a - b
        if np.allclose(difference, 0.0):
            w_stat, w_p, t_stat, t_p = np.nan, 1.0, np.nan, 1.0
        else:
            try:
                w_stat, w_p = scipy_stats.wilcoxon(a, b, zero_method="pratt")
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            t_stat, t_p = scipy_stats.ttest_rel(a, b)
        rows.append({
            "split_role": split_role,
            "model_a": model_a,
            "model_b": model_b,
            "n_windows": len(matrix),
            "mean_diff_a_minus_b": difference.mean(),
            "median_diff_a_minus_b": np.median(difference),
            "wins_a": int(np.sum(difference < -1e-6)),
            "ties": int(np.sum(np.abs(difference) <= 1e-6)),
            "wins_b": int(np.sum(difference > 1e-6)),
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p": w_p,
            "ttest_statistic": t_stat,
            "ttest_p": t_p,
            "units_independent": False,
            "inferential_status": "final_exploratory" if final_all_models else "provisional_descriptive_do_not_interpret",
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["wilcoxon_p_holm_within_split"] = holm_adjust(table["wilcoxon_p"].tolist())
        table["ttest_p_holm_within_split"] = holm_adjust(table["ttest_p"].tolist())
    return table


def _boundary_bootstrap_tost(
    difference: np.ndarray,
    sampled_indices: np.ndarray,
    margin: float,
) -> dict[str, float]:
    """Moving-block bootstrap TOST against the least-favourable null boundaries.

    The two null distributions are formed by centring the paired-difference
    series at ``-margin`` and ``+margin``.  The +1 correction prevents a zero
    Monte Carlo p-value.
    """
    observed = float(np.mean(difference))
    sampled_raw = difference[sampled_indices].mean(axis=1)
    # Non-circular moving blocks underweight series endpoints.  Centre the
    # bootstrap distribution itself so that this edge effect cannot shift the
    # TOST null away from its stated boundary.
    sampling_error = sampled_raw - sampled_raw.mean()
    lower_null = -margin + sampling_error
    upper_null = margin + sampling_error
    denominator = len(sampled_indices) + 1
    p_lower = (np.count_nonzero(lower_null >= observed) + 1) / denominator
    p_upper = (np.count_nonzero(upper_null <= observed) + 1) / denominator
    # Basic bootstrap intervals invert the same centered-error tails used by
    # the boundary-null tests.  Percentile intervals would reverse neither
    # tail and can disagree with TOST when the distribution is asymmetric.
    error_q05, error_q95 = np.quantile(sampling_error, [0.05, 0.95])
    error_q025, error_q975 = np.quantile(sampling_error, [0.025, 0.975])
    ci90_lower, ci90_upper = observed - error_q95, observed - error_q05
    ci95_lower, ci95_upper = observed - error_q975, observed - error_q025
    return {
        "mean_diff_candidate_minus_reference": observed,
        "ci90_lower": float(ci90_lower),
        "ci90_upper": float(ci90_upper),
        "ci95_lower": float(ci95_lower),
        "ci95_upper": float(ci95_upper),
        "tost_p_lower": float(p_lower),
        "tost_p_upper": float(p_upper),
        "tost_p": float(max(p_lower, p_upper)),
        "ci90_strictly_inside_margin": bool(ci90_lower > -margin and ci90_upper < margin),
    }


def _basic_bootstrap_difference(
    difference: np.ndarray,
    sampled_indices: np.ndarray,
) -> dict[str, float]:
    """Paired basic moving-block-bootstrap intervals for a mean difference."""
    observed = float(np.mean(difference))
    sampled = difference[sampled_indices].mean(axis=1)
    sampling_error = sampled - sampled.mean()
    q025, q975 = np.quantile(sampling_error, [0.025, 0.975])
    q05, q95 = np.quantile(sampling_error, [0.05, 0.95])
    return {
        "mean_difference": observed,
        "ci90_lower": float(observed - q95),
        "ci90_upper": float(observed - q05),
        "ci95_lower": float(observed - q975),
        "ci95_upper": float(observed - q025),
    }


def _event_domain_rows(events: pd.DataFrame, model_order: tuple[str, ...]) -> pd.DataFrame:
    """Select and label the six prespecified held-out event conditions."""
    required = {
        "model_id", "window_id", "split_role", "event", "event_span",
        "relative_position", "trial_type", "mean_nll", "n_events",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event selection table lacks required columns: {sorted(missing)}")
    selected = events[
        events["model_id"].isin(model_order) & events["split_role"].eq("test")
    ].copy()
    selected["event_domain"] = ""
    selected.loc[
        selected["event"].eq("rare_long")
        & selected["event_span"].eq("focus")
        & selected["relative_position"].eq(0)
        & selected["trial_type"].eq("long"),
        "event_domain",
    ] = "rare_long_in_short_block"
    selected.loc[
        selected["event"].eq("rare_short")
        & selected["event_span"].eq("focus")
        & selected["relative_position"].eq(0)
        & selected["trial_type"].eq("short"),
        "event_domain",
    ] = "rare_short_in_long_block"
    focused_transition = (
        selected["event"].isin(("short_to_long", "long_to_short"))
        & selected["event_span"].eq("focus")
        & selected["relative_position"].between(-5, 5)
    )
    for event in ("short_to_long", "long_to_short"):
        for trial_type in ("short", "long"):
            selected.loc[
                focused_transition
                & selected["event"].eq(event)
                & selected["trial_type"].eq(trial_type),
                "event_domain",
            ] = f"{event}_{trial_type}_trials"
    selected = selected[selected["event_domain"].isin(EVENT_SELECTION_DOMAINS)].copy()
    if selected.empty:
        raise ValueError("No held-out rows match the six Figure 0 event conditions")

    selected["mean_nll"] = pd.to_numeric(selected["mean_nll"], errors="coerce")
    selected["n_events"] = pd.to_numeric(selected["n_events"], errors="coerce")
    counts = selected["n_events"].to_numpy(float)
    nll = selected["mean_nll"].to_numpy(float)
    if not np.isfinite(nll).all() or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("Figure 0 event conditions require finite NLL and positive event counts")
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Figure 0 event counts must be integers")

    structural_key = [
        "window_id", "event", "event_span", "relative_position", "trial_type",
        "event_domain",
    ]
    duplicates = selected.groupby(["model_id", *structural_key], sort=False).size()
    if (duplicates > 1).any():
        raise ValueError("Figure 0 event conditions contain duplicate model/position rows")
    coverage = selected.groupby(structural_key, sort=False).agg(
        models=("model_id", "nunique"), count_versions=("n_events", "nunique"),
    )
    if (coverage["models"] != len(model_order)).any():
        raise ValueError("Figure 0 event conditions are not paired across every deployable model")
    if (coverage["count_versions"] != 1).any():
        raise ValueError("Figure 0 event counts disagree across paired models")
    return selected


def event_condition_window_scores(
    events: pd.DataFrame,
    model_order: tuple[str, ...],
    *,
    expected_window_count: int | None = None,
    expected_window_ids: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Return event-count-weighted domain NLLs on a strict common-window set."""
    models = tuple(model_order)
    selected = _event_domain_rows(events, models)
    selected["weighted_nll"] = selected["mean_nll"] * selected["n_events"]
    scores = selected.groupby(
        ["model_id", "window_id", "event_domain"], as_index=False,
    ).agg(weighted_nll=("weighted_nll", "sum"), n_events=("n_events", "sum"))
    scores["mean_nll"] = scores["weighted_nll"] / scores["n_events"]

    required_cells = len(models) * len(EVENT_SELECTION_DOMAINS)
    complete = scores.groupby("window_id").size()
    common_windows = tuple(sorted(int(window) for window in complete.index[complete.eq(required_cells)]))
    if not common_windows:
        raise ValueError("No window has all six event conditions for every deployable model")
    if expected_window_count is not None and len(common_windows) != expected_window_count:
        raise ValueError(
            f"Figure 0 requires {expected_window_count} complete event windows; "
            f"found {len(common_windows)}"
        )
    if expected_window_ids is not None:
        required_windows = tuple(sorted(int(window) for window in expected_window_ids))
        if len(required_windows) != len(set(required_windows)):
            raise ValueError("Figure 0 expected event-window IDs must be unique")
        if common_windows != required_windows:
            raise ValueError(
                "Figure 0 complete event-window IDs disagree with the prespecified set; "
                f"expected {required_windows}, found {common_windows}"
            )
    scores = scores[scores["window_id"].isin(common_windows)].copy()
    expected = pd.MultiIndex.from_product(
        [models, common_windows, EVENT_SELECTION_DOMAINS],
        names=["model_id", "window_id", "event_domain"],
    )
    actual = pd.MultiIndex.from_frame(scores[["model_id", "window_id", "event_domain"]])
    if len(actual) != len(expected) or not actual.is_unique or set(actual) != set(expected):
        raise ValueError("Figure 0 event score matrix is incomplete or duplicated")
    transition_rows = selected[
        selected["window_id"].isin(common_windows)
        & selected["event"].isin(("short_to_long", "long_to_short"))
    ]
    expected_positions = set(range(-5, 6))
    position_sets = transition_rows.groupby(
        ["model_id", "window_id", "event"], sort=False,
    )["relative_position"].agg(lambda values: set(int(value) for value in values))
    expected_transition_keys = pd.MultiIndex.from_product(
        [models, common_windows, ("short_to_long", "long_to_short")],
        names=["model_id", "window_id", "event"],
    )
    if set(position_sets.index) != set(expected_transition_keys) or any(
        positions != expected_positions for positions in position_sets
    ):
        raise ValueError("Figure 0 transition conditions require complete positions -5 through +5")
    order = {domain: index for index, domain in enumerate(EVENT_SELECTION_DOMAINS)}
    scores["_domain_order"] = scores["event_domain"].map(order)
    scores = scores.sort_values(["model_id", "window_id", "_domain_order"]).drop(
        columns=["weighted_nll", "_domain_order"],
    ).reset_index(drop=True)
    return scores, common_windows


def combined_selection_tables(
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    model_order: tuple[str, ...],
    *,
    margin: float = EQUIVALENCE_MARGIN_NLL,
    alpha: float = EQUIVALENCE_ALPHA,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_length: int = EQUIVALENCE_BLOCK_LENGTH,
    expected_total_windows: int | None = None,
    expected_event_windows: int | None = None,
    expected_event_window_ids: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gate on total test NLL, then compare six-condition event robustness.

    The total gate is a Holm-adjusted, one-sided moving-block-bootstrap
    non-inferiority test against the observed total-test leader. Event ranking
    is conditional on that observed gate. All event domains and windows receive
    equal weight after within-domain event-count weighting.
    """
    models = tuple(model_order)
    if len(models) < 2:
        raise ValueError("Combined Figure 0 selection requires at least two models")
    if any(model not in MODEL_BY_ID or MODEL_BY_ID[model].oracle for model in models):
        raise ValueError("Combined Figure 0 selection accepts known deployable models only")
    if not margin > 0 or not 0 < alpha < 0.5:
        raise ValueError("Selection margin must be positive and alpha must lie in (0, 0.5)")

    total = strict_complete_matrix(
        metrics[metrics["split_role"].eq("test")], models,
        expected_window_count=expected_total_windows,
    )
    total_means = total.mean(axis=0)
    total_minimum = float(total_means.min())
    total_leaders = [
        model for model in models
        if float(total_means[model]) - total_minimum <= TIE_TOLERANCE_NLL
    ]
    total_reference = next(model for model in models if model in total_leaders)
    total_indices = moving_block_indices(
        len(total), replicates=replicates, block_length=block_length,
    )
    total_rows: list[dict] = []
    for model in models:
        difference = total[model].to_numpy(float) - total[total_reference].to_numpy(float)
        interval = _basic_bootstrap_difference(difference, total_indices)
        if model == total_reference:
            p_value = np.nan
        else:
            p_value = _boundary_bootstrap_tost(difference, total_indices, margin)["tost_p_upper"]
        total_rows.append({
            "model_id": model,
            "total_mean_nll": float(total_means[model]),
            "total_reference_model_id": total_reference,
            "total_reference_tie_count": len(total_leaders),
            "total_n_windows": len(total),
            "total_noninferiority_p": p_value,
            **{f"total_{key}": value for key, value in interval.items()},
        })
    summary = pd.DataFrame(total_rows)
    summary["total_noninferiority_p_holm"] = np.nan
    summary["passes_total_noninferiority"] = summary["model_id"].eq(total_reference)
    candidate_mask = ~summary["model_id"].eq(total_reference)
    adjusted = holm_adjust(summary.loc[candidate_mask, "total_noninferiority_p"].tolist())
    summary.loc[candidate_mask, "total_noninferiority_p_holm"] = adjusted
    summary.loc[candidate_mask, "passes_total_noninferiority"] = np.asarray(adjusted) < alpha
    shortlisted = tuple(summary.loc[summary["passes_total_noninferiority"], "model_id"])

    domain_scores, common_windows = event_condition_window_scores(
        events, models,
        expected_window_count=expected_event_windows,
        expected_window_ids=expected_event_window_ids,
    )
    if not set(common_windows).issubset(set(int(window) for window in total.index)):
        raise ValueError("Figure 0 event windows are not a subset of total held-out windows")
    event_matrix = domain_scores.groupby(["model_id", "window_id"], as_index=False).agg(
        event_composite_nll=("mean_nll", "mean"),
    ).pivot(index="window_id", columns="model_id", values="event_composite_nll").reindex(
        index=common_windows, columns=models,
    )
    if event_matrix.isna().any().any():
        raise ValueError("Figure 0 event composite unexpectedly contains missing values")
    event_means = event_matrix.mean(axis=0)
    prevalence = domain_scores.assign(
        weighted_nll=lambda frame: frame["mean_nll"] * frame["n_events"],
    ).groupby(["model_id", "window_id"], as_index=False).agg(
        weighted_nll=("weighted_nll", "sum"), n_events=("n_events", "sum"),
    )
    prevalence["prevalence_weighted_event_nll"] = (
        prevalence["weighted_nll"] / prevalence["n_events"]
    )
    prevalence_means = prevalence.groupby("model_id")["prevalence_weighted_event_nll"].mean()
    event_minimum = min(float(event_means[model]) for model in shortlisted)
    event_leaders = [
        model for model in shortlisted
        if float(event_means[model]) - event_minimum <= TIE_TOLERANCE_NLL
    ]
    event_reference = next(model for model in models if model in event_leaders)
    event_indices = moving_block_indices(
        len(event_matrix), replicates=replicates, block_length=block_length,
    )
    sampled_shortlist = event_matrix.loc[:, shortlisted].to_numpy(float)[event_indices].mean(axis=1)
    minima = sampled_shortlist.min(axis=1, keepdims=True)
    best = sampled_shortlist - minima <= TIE_TOLERANCE_NLL
    best_credit = best / best.sum(axis=1, keepdims=True)
    best_probability = dict(zip(shortlisted, best_credit.mean(axis=0), strict=True))

    event_rows = []
    for model in models:
        difference = (
            event_matrix[model].to_numpy(float)
            - event_matrix[event_reference].to_numpy(float)
        )
        interval = _basic_bootstrap_difference(difference, event_indices)
        if model == event_reference:
            tost_p = np.nan
        elif model in shortlisted:
            tost_p = _boundary_bootstrap_tost(difference, event_indices, margin)["tost_p"]
        else:
            tost_p = np.nan
        event_rows.append({
            "model_id": model,
            "event_composite_mean_nll": float(event_means[model]),
            "event_reference_model_id": event_reference,
            "event_n_windows": len(event_matrix),
            "event_tost_p": tost_p,
            "conditional_probability_event_best": best_probability.get(model, 0.0),
            **{f"event_{key}": value for key, value in interval.items()},
        })
    summary = summary.merge(pd.DataFrame(event_rows), on="model_id", validate="one_to_one")
    summary["event_tost_p_holm"] = np.nan
    summary["event_equivalent_holm"] = summary["model_id"].eq(event_reference)
    event_candidate_mask = (
        summary["passes_total_noninferiority"]
        & ~summary["model_id"].eq(event_reference)
    )
    event_adjusted = holm_adjust(summary.loc[event_candidate_mask, "event_tost_p"].tolist())
    summary.loc[event_candidate_mask, "event_tost_p_holm"] = event_adjusted
    summary.loc[event_candidate_mask, "event_equivalent_holm"] = np.asarray(event_adjusted) < alpha
    summary["observed_combined_winner"] = summary["model_id"].isin(event_leaders)
    prevalence_minimum = min(float(prevalence_means[model]) for model in shortlisted)
    prevalence_leaders = {
        model for model in shortlisted
        if float(prevalence_means[model]) - prevalence_minimum <= TIE_TOLERANCE_NLL
    }
    summary["prevalence_weighted_event_mean_nll"] = summary["model_id"].map(prevalence_means)
    summary["prevalence_weighted_event_winner"] = summary["model_id"].isin(prevalence_leaders)
    summary.insert(0, "family", summary["model_id"].map(lambda model: MODEL_BY_ID[model].family))
    summary.insert(2, "model_label", summary["model_id"].map(lambda model: MODEL_BY_ID[model].label))
    summary["selection_margin_nll"] = margin
    summary["alpha"] = alpha
    summary["bootstrap_block_length"] = min(block_length, len(event_matrix))
    summary["bootstrap_replicates"] = replicates
    summary["inferential_status"] = "exploratory_post_selection_within_mouse"

    domain_rows = []
    for domain in EVENT_SELECTION_DOMAINS:
        matrix = domain_scores[domain_scores["event_domain"].eq(domain)].pivot(
            index="window_id", columns="model_id", values="mean_nll",
        ).reindex(index=common_windows, columns=models)
        reference_values = matrix[event_reference].to_numpy(float)
        for model in models:
            difference = matrix[model].to_numpy(float) - reference_values
            interval = _basic_bootstrap_difference(difference, event_indices)
            domain_rows.append({
                "family": MODEL_BY_ID[model].family,
                "model_id": model,
                "model_label": MODEL_BY_ID[model].label,
                "event_domain": domain,
                "event_domain_label": EVENT_SELECTION_LABELS[domain].replace("\n", " "),
                "reference_model_id": event_reference,
                "n_windows": len(matrix),
                "mean_nll": float(matrix[model].mean()),
                **interval,
            })
    domain_table = pd.DataFrame(domain_rows)
    family_rank = {family: index for index, family in enumerate(MODEL_FAMILY_ORDER)}
    model_rank = {model: index for index, model in enumerate(models)}
    summary["_family_rank"] = summary["family"].map(family_rank)
    summary["_model_rank"] = summary["model_id"].map(model_rank)
    summary = summary.sort_values(["_family_rank", "_model_rank"]).drop(
        columns=["_family_rank", "_model_rank"],
    ).reset_index(drop=True)
    return summary, domain_table


def equivalence_table(
    metrics: pd.DataFrame,
    model_order: tuple[str, ...],
    *,
    margin: float = EQUIVALENCE_MARGIN_NLL,
    alpha: float = EQUIVALENCE_ALPHA,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_length: int = EQUIVALENCE_BLOCK_LENGTH,
    expected_window_count: int | None = None,
) -> pd.DataFrame:
    """Compare every eligible deployable model with the observed test minimum.

    This is an exploratory, conditional-on-selection analysis for one mouse.
    Windows are equally weighted, matching Figure 0's ranking estimand.  Holm
    adjustment is applied to the candidate-level TOST p-values within split.
    """
    models = tuple(model_order)
    if len(models) < 2:
        raise ValueError("Equivalence analysis requires at least two deployable models")
    unknown = [model for model in models if model not in MODEL_BY_ID]
    if unknown:
        raise ValueError(f"Unknown models in equivalence analysis: {unknown}")
    if any(MODEL_BY_ID[model].oracle for model in models):
        raise ValueError("Oracle benchmarks cannot enter the deployable equivalence analysis")
    if not margin > 0 or not 0 < alpha < 0.5:
        raise ValueError("Equivalence margin must be positive and alpha must lie in (0, 0.5)")

    matrices = {
        split_role: strict_complete_matrix(
            metrics[metrics["split_role"].eq(split_role)], models,
            expected_window_count=expected_window_count,
        )
        for split_role in SPLIT_ROLES
    }
    expected_windows = set(matrices["test"].index)
    if any(set(matrix.index) != expected_windows for matrix in matrices.values()):
        raise ValueError("Figure 0 equivalence analysis requires the same window IDs in every split")

    test_means = matrices["test"].mean(axis=0)
    minimum = float(test_means.min())
    tied_best = [model for model in models if float(test_means[model]) - minimum <= TIE_TOLERANCE_NLL]
    reference_model = next(model for model in models if model in tied_best)

    rows: list[dict] = []
    for split_role in SPLIT_ROLES:
        matrix = matrices[split_role]
        sampled_indices = moving_block_indices(
            len(matrix), replicates=replicates, block_length=block_length,
        )
        reference = matrix[reference_model].to_numpy(float)
        for model in models:
            difference = matrix[model].to_numpy(float) - reference
            if model == reference_model:
                result = {
                    "mean_diff_candidate_minus_reference": 0.0,
                    "ci90_lower": 0.0,
                    "ci90_upper": 0.0,
                    "ci95_lower": 0.0,
                    "ci95_upper": 0.0,
                    "tost_p_lower": np.nan,
                    "tost_p_upper": np.nan,
                    "tost_p": np.nan,
                    "ci90_strictly_inside_margin": True,
                }
            else:
                result = _boundary_bootstrap_tost(difference, sampled_indices, margin)
            rows.append({
                "split_role": split_role,
                "family": MODEL_BY_ID[model].family,
                "model_id": model,
                "model_label": MODEL_BY_ID[model].label,
                "reference_model_id": reference_model,
                "reference_model_label": MODEL_BY_ID[reference_model].label,
                "reference_tie_count": len(tied_best),
                "selected_as_observed_test_best": model == reference_model,
                "n_windows": len(matrix),
                "candidate_mean_window_nll": float(matrix[model].mean()),
                "reference_mean_window_nll": float(matrix[reference_model].mean()),
                **result,
                "equivalence_margin_nll": margin,
                "alpha": alpha,
                "bootstrap_block_length": min(block_length, len(matrix)),
                "bootstrap_replicates": replicates,
                "bootstrap_distribution_centering": "resampled_means_centered_on_observed_mean",
                "units_independent": False,
                "inferential_status": "exploratory_post_selection_within_mouse",
            })

    table = pd.DataFrame(rows)
    table["tost_p_holm_within_split"] = np.nan
    table["equivalent_raw"] = table["selected_as_observed_test_best"]
    table["equivalent_holm"] = table["selected_as_observed_test_best"]
    for split_role in SPLIT_ROLES:
        mask = table["split_role"].eq(split_role) & ~table["selected_as_observed_test_best"]
        adjusted = holm_adjust(table.loc[mask, "tost_p"].tolist())
        table.loc[mask, "tost_p_holm_within_split"] = adjusted
        table.loc[mask, "equivalent_raw"] = table.loc[mask, "tost_p"].lt(alpha)
        table.loc[mask, "equivalent_holm"] = table.loc[mask, "tost_p_holm_within_split"].lt(alpha)
    family_rank = {family: index for index, family in enumerate(MODEL_FAMILY_ORDER)}
    model_rank = {model: index for index, model in enumerate(models)}
    split_rank = {split: index for index, split in enumerate(SPLIT_ROLES)}
    table["_split_rank"] = table["split_role"].map(split_rank)
    table["_family_rank"] = table["family"].map(family_rank)
    table["_model_rank"] = table["model_id"].map(model_rank)
    return table.sort_values(["_split_rank", "_family_rank", "_model_rank"]).drop(
        columns=["_split_rank", "_family_rank", "_model_rank"],
    ).reset_index(drop=True)


def family_summary_table(equivalence: pd.DataFrame) -> pd.DataFrame:
    """Observed test-best deployable model in each represented family."""
    test = equivalence[equivalence["split_role"].eq("test")].copy()
    rows = []
    for family in MODEL_FAMILY_ORDER:
        group = test[test["family"].eq(family)]
        if group.empty:
            rows.append({
                "family": family,
                "eligible_deployable_models": 0,
                "family_best_model_id": "",
                "family_best_model_label": "not available",
                "family_best_tie_count": 0,
                "observed_test_mean_window_nll": np.nan,
                "difference_from_observed_best": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "equivalent_to_observed_best_holm": False,
                "observed_overall_best": False,
            })
            continue
        family_minimum = float(group["candidate_mean_window_nll"].min())
        tied = group[
            group["candidate_mean_window_nll"] - family_minimum <= TIE_TOLERANCE_NLL
        ]
        best = tied.iloc[0]
        rows.append({
            "family": family,
            "eligible_deployable_models": len(group),
            "family_best_model_id": best["model_id"],
            "family_best_model_label": best["model_label"],
            "family_best_tie_count": len(tied),
            "observed_test_mean_window_nll": best["candidate_mean_window_nll"],
            "difference_from_observed_best": best["mean_diff_candidate_minus_reference"],
            "ci95_lower": best["ci95_lower"],
            "ci95_upper": best["ci95_upper"],
            "equivalent_to_observed_best_holm": bool(best["equivalent_holm"]),
            "observed_overall_best": bool(best["selected_as_observed_test_best"]),
        })
    return pd.DataFrame(rows)
