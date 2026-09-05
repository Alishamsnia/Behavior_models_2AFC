"""Paired significance testing matched by session or rolling-window identity.

The registered confirmatory contrasts follow the nested model sequence:
M0-M1, M1-M2, M2-M3, and M3-M4.  Those four comparisons are selected by the
scientific nesting, not by the observed data, and their Wilcoxon p-values are
Holm-corrected together within each declared panel/metric family.  Paired
t-tests remain in the exported tables as a parametric cross-check.

Rolling train/validation and information-criterion windows overlap heavily.
Callers retain an explicit independence flag so their p-values are presented
as exploratory within-mouse summaries rather than independent-replicate
population inference.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

from post_analysis.common import MODEL_ORDER


ADJACENT_MODEL_PAIRS = (("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M3", "M4"))
PAIRWISE_RESULT_COLUMNS = [
    "model_a", "model_b", "n_pairs", "median_diff_a_minus_b",
    "mean_diff_a_minus_b", "wilcoxon_statistic", "wilcoxon_p",
    "ttest_statistic", "ttest_p", "wilcoxon_p_holm", "ttest_p_holm",
]


def complete_case_matrix(
    df: pd.DataFrame, unit_col: str, value_col: str, model_order=MODEL_ORDER,
) -> pd.DataFrame:
    """Unit x model matrix, restricted to units where every model in
    ``model_order`` has exactly one finite value. Non-finite values
    (NaN/+-inf) are treated as missing, not as real observations, so a
    single infinite fit cannot silently masquerade as a comparable unit.
    """
    working = df.copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working.loc[~np.isfinite(working[value_col]), value_col] = np.nan
    counts = working.groupby([unit_col, "model_id"]).size()
    if (counts > 1).any():
        dupes = counts[counts > 1].index.tolist()
        raise ValueError(f"Duplicate (unit, model) rows in complete_case_matrix: {dupes[:5]}")
    wide = working.pivot_table(index=unit_col, columns="model_id", values=value_col, aggfunc="first")
    wide = wide.reindex(columns=model_order)
    return wide.dropna(axis=0, how="any")


def friedman_test(matrix: pd.DataFrame) -> dict:
    """Omnibus paired test across all models in ``matrix.columns`` at once.
    Requires >=3 units and >=3 models (scipy's own minimum for the chi-square
    approximation to be meaningful); returns NaNs below that rather than
    raising, so callers can render "unavailable" instead of crashing.
    """
    n_units, n_models = matrix.shape
    if n_units < 3 or n_models < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_units": n_units, "n_models": n_models}
    columns = [matrix[m].to_numpy() for m in matrix.columns]
    if all(np.allclose(col, columns[0]) for col in columns[1:]):
        # Every model tied on every unit: no variation for Friedman to test.
        return {"statistic": 0.0, "p_value": 1.0, "n_units": n_units, "n_models": n_models}
    stat, p = sps.friedmanchisquare(*columns)
    return {"statistic": float(stat), "p_value": float(p), "n_units": n_units, "n_models": n_models}


def holm_adjust(pvalues: list) -> list:
    """Holm-Bonferroni step-down adjustment. NaN p-values are treated as 1.0
    for ranking purposes and returned as NaN (not silently promoted to 1.0)
    so a downstream reader can tell "not significant" from "test failed".
    """
    arr = np.asarray(pvalues, dtype=float)
    is_nan = ~np.isfinite(arr)
    filled = np.where(is_nan, 1.0, arr)
    n = len(filled)
    order = np.argsort(filled)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((n - rank) * filled[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    adjusted[is_nan] = np.nan
    return adjusted.tolist()


def _paired_diff_tests(a: np.ndarray, b: np.ndarray) -> tuple:
    """(wilcoxon_stat, wilcoxon_p, ttest_stat, ttest_p) for one pair of
    matched columns. ``zero_method="pratt"`` keeps zero-difference units in
    the ranking (recommended over the "wilcox" default, which drops them
    outright -- with only ~20-30 pairs, discarding near-ties changes the
    test more than down-weighting them does; near-tie NLL margins as small
    as ~1e-6 occur in this dataset between nested models, see
    ``collect._compute_win_counts``).
    """
    diff = a - b
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if np.all(diff == 0.0):
        return float("nan"), 1.0, float("nan"), 1.0
    try:
        w_stat, w_p = sps.wilcoxon(a, b, zero_method="pratt", alternative="two-sided", mode="auto")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    t_stat, t_p = sps.ttest_rel(a, b)
    return float(w_stat), float(w_p), float(t_stat), float(t_p)


def pairwise_tests(matrix: pd.DataFrame, model_order=MODEL_ORDER) -> pd.DataFrame:
    """All C(k,2) pairwise comparisons (Wilcoxon signed-rank + paired t-test),
    Holm-corrected as one family per test type. This is the single source of
    truth for pairwise significance -- ``select_reference_rows`` only
    re-displays a subset of it, it never re-runs or re-corrects anything.
    """
    models = [m for m in model_order if m in matrix.columns]
    rows = []
    for a, b in combinations(models, 2):
        av, bv = matrix[a].to_numpy(), matrix[b].to_numpy()
        w_stat, w_p, t_stat, t_p = _paired_diff_tests(av, bv)
        rows.append({
            "model_a": a, "model_b": b, "n_pairs": len(matrix),
            "median_diff_a_minus_b": float(np.median(av - bv)),
            "mean_diff_a_minus_b": float(np.mean(av - bv)),
            "wilcoxon_statistic": w_stat, "wilcoxon_p": w_p,
            "ttest_statistic": t_stat, "ttest_p": t_p,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["wilcoxon_p_holm"] = holm_adjust(table["wilcoxon_p"].tolist())
    table["ttest_p_holm"] = holm_adjust(table["ttest_p"].tolist())
    return table


def adjacent_pairwise_tests(
    matrix: pd.DataFrame, pairs=ADJACENT_MODEL_PAIRS,
) -> pd.DataFrame:
    """Run the four pre-specified adjacent nested-model contrasts.

    The matrix must already be complete-case matched.  Holm adjustment is
    applied over exactly the supplied contrasts, once per test type.  This is
    intentionally separate from :func:`pairwise_tests`, whose ten-comparison
    family remains available for legacy callers but is not used by Figures
    01, 03, or 15.
    """
    required_models = {model for pair in pairs for model in pair}
    missing_models = sorted(required_models - set(matrix.columns))
    if missing_models:
        raise ValueError(
            f"Cannot run the registered adjacent contrasts; missing model columns: {missing_models}"
        )
    if matrix.empty:
        return pd.DataFrame(columns=PAIRWISE_RESULT_COLUMNS)

    rows = []
    for model_a, model_b in pairs:
        a = matrix[model_a].to_numpy(dtype=float)
        b = matrix[model_b].to_numpy(dtype=float)
        w_stat, w_p, t_stat, t_p = _paired_diff_tests(a, b)
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "n_pairs": len(matrix),
            "median_diff_a_minus_b": float(np.median(a - b)),
            "mean_diff_a_minus_b": float(np.mean(a - b)),
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p": w_p,
            "ttest_statistic": t_stat,
            "ttest_p": t_p,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["wilcoxon_p_holm"] = holm_adjust(table["wilcoxon_p"].tolist())
    table["ttest_p_holm"] = holm_adjust(table["ttest_p"].tolist())
    return table


def select_reference_rows(pairwise_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Re-orient the already Holm-corrected ``pairwise_tests`` output around
    one reference model (typically the model with the best/lowest observed
    summary on some already-visible glyph, e.g. fig01's diamond markers) for
    compact figure annotation. Does NOT recompute or re-correct p-values --
    reusing the full-family Holm adjustment here (rather than a fresh
    Holm-correction over only the reference's comparisons) avoids
    understating multiplicity from a data-dependent reference choice.
    """
    if pairwise_df.empty:
        return pairwise_df
    rows = []
    for _, row in pairwise_df.iterrows():
        if row["model_a"] == reference:
            other, sign = row["model_b"], 1.0
        elif row["model_b"] == reference:
            other, sign = row["model_a"], -1.0
        else:
            continue
        rows.append({
            "reference_model": reference, "other_model": other, "n_pairs": row["n_pairs"],
            "median_diff_reference_minus_other": sign * row["median_diff_a_minus_b"],
            "wilcoxon_p": row["wilcoxon_p"], "wilcoxon_p_holm": row["wilcoxon_p_holm"],
            "ttest_p": row["ttest_p"], "ttest_p_holm": row["ttest_p_holm"],
        })
    return pd.DataFrame(rows)


def significance_stars(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
