"""Compact sliding-window tables used by expanded Figures 7--23."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from post_analysis.common import REPO_ROOT

from .collect import (
    _bool_mask,
    _converged_winner,
    _window_directories,
    canonical_schedule,
    source_fingerprint,
)
from .config import (
    FIGURE20_WINDOW_SAMPLE_COUNT, FIGURE20_WINDOW_SAMPLE_SEED, MODEL_BY_ID, MODEL_ORDER, SPLIT_ROLES,
)

# This machine also has an OLDER, DIVERGENT sibling checkout pip-installed
# under the same top-level package names (see post_analysis/identifiability
# /collect.py's docstring for the full hazard). Force THIS repo's src/ to
# the front of sys.path before importing the model package so it always
# wins the lookup -- a bare `import m0_sensory_v1` would otherwise risk
# silently importing the wrong codebase with no error.
_SRC_ROOT = str(REPO_ROOT / "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
elif sys.path[0] != _SRC_ROOT:
    sys.path.remove(_SRC_ROOT)
    sys.path.insert(0, _SRC_ROOT)

import torch  # noqa: E402

from m0_sensory_v1.data.task_moments import TaskMoments  # noqa: E402
from m0_sensory_v1.model.bayesian import (  # noqa: E402
    DecodedParameters, category_variances, log_odds_coefficients, sensory_standard_deviation,
)
from m0_sensory_v1.model.m1_bayesian import integrate_general_decision_region  # noqa: E402

TRIAL_SUBSETS = {
    "all": lambda d: pd.Series(True, index=d.index),
    "short_block": lambda d: d.true_block.eq("S"),
    "long_block": lambda d: d.true_block.eq("L"),
    "common_short": lambda d: d.category_long.eq(0) & d.true_block.eq("S"),
    "common_long": lambda d: d.category_long.eq(1) & d.true_block.eq("L"),
    "rare_short": lambda d: d.category_long.eq(0) & d.true_block.eq("L"),
    "rare_long": lambda d: d.category_long.eq(1) & d.true_block.eq("S"),
}
EVENT_RANGES = {
    "short_to_long": range(-15, 21), "long_to_short": range(-15, 21),
    "rare_short": range(-2, 3), "rare_long": range(-2, 3),
}
FOCUSED_TRANSITION_RANGE = range(-5, 6)
REQUIRED = {
    "session_id", "trial_index", "source_trial_index", "block_index", "true_run_length",
    "split_role", "true_block", "category_long", "duration_s", "choice_long",
    "model_probability_long", "per_trial_negative_log_likelihood", "model_evaluated",
}
OPTIONAL = {
    "model_prior_long", "model_sensory_sd_s", "fitted_w", "fitted_s0",
    "fitted_ell_short", "fitted_ell_long", "fitted_kappa",
}
STRUCTURAL_COLUMNS = (
    "split_role", "session_id", "source_trial_index", "trial_index", "block_index",
    "true_run_length", "true_block", "category_long", "duration_s", "choice_long",
)
TABLE_COLUMNS = {
    "extended_trial_subset_metrics.csv": ["model_id", "window_id", "split_role", "trial_subset", "n_trials", "mean_nll", "nll_sem"],
    "extended_event_aligned.csv": ["model_id", "window_id", "split_role", "event", "event_span", "relative_position", "trial_type", "mean_nll", "mouse_correct", "model_probability_correct", "n_events"],
    "extended_parameters.csv": ["model_id", "window_id", "test_rank", "param_name", "param_value"],
    "extended_psychometrics.csv": ["model_id", "window_id", "block", "duration_s", "n_trials", "mouse_long_rate", "model_long_rate"],
    "extended_prior_logit_samples.csv": ["model_id", "window_id", "split_role", "session_id", "trial_index", "true_block", "sensory_logit", "belief_logit", "summation_logit"],
    "extended_logit_event_aligned.csv": ["model_id", "window_id", "split_role", "event", "event_span", "relative_position", "mean_sensory_logit", "mean_belief_logit", "mean_summation_logit", "n_events"],
    "extended_logit_psychometrics.csv": ["model_id", "window_id", "split_role", "true_block", "duration_s", "n_trials", "sensory_probability", "belief_probability", "summation_probability"],
    "extended_prior_sensory_bins.csv": ["model_id", "window_id", "bin_label", "category", "difficulty", "n_trials", "mean_abs_prior_logit", "mean_abs_sensory_logit"],
    "extended_prior_sensory_difficulty.csv": ["model_id", "window_id", "condition", "n_trials", "mean_abs_prior_logit"],
}

# Figure 24's 8 ISI bins: the task presents exactly 5 discrete durations per
# category (0.20/0.35/0.50/0.65/0.80s short; 1.70/1.85/2.00/2.15/2.30s long).
# The two middle levels on each side are merged so both extremes survive as
# pure single-level bins -- page 2's hard-vs-easy contrast needs the hardest
# and easiest bins to be actual presented durations, not an average of two.
DURATION_BIN_DEFINITIONS = (
    {"label": "Short 0.20s (easy)", "category": "short", "difficulty": "easy", "values": (0.20,)},
    {"label": "Short 0.35–0.50s", "category": "short", "difficulty": "mid", "values": (0.35, 0.50)},
    {"label": "Short 0.65s", "category": "short", "difficulty": "mid", "values": (0.65,)},
    {"label": "Short 0.80s (hard)", "category": "short", "difficulty": "hard", "values": (0.80,)},
    {"label": "Long 1.70s (hard)", "category": "long", "difficulty": "hard", "values": (1.70,)},
    {"label": "Long 1.85–2.00s", "category": "long", "difficulty": "mid", "values": (1.85, 2.00)},
    {"label": "Long 2.15s", "category": "long", "difficulty": "mid", "values": (2.15,)},
    {"label": "Long 2.30s (easy)", "category": "long", "difficulty": "easy", "values": (2.30,)},
)
_DURATION_TO_BIN = {
    round(value, 2): meta for meta in DURATION_BIN_DEFINITIONS for value in meta["values"]
}


def _task_moments(mouse_id: str) -> dict:
    """Fixed empirical task-distribution moments backing the sensory term of
    every Bayesian-family model's decision rule (see
    ``src/m0_sensory_v1/model/bayesian.py:category_variances``). Loaded from
    the same registered config the model side uses, not re-derived, so this
    analysis can never silently drift from the fitted models' own constants.
    """
    suffix = "" if mouse_id.upper() == "MC08" else f"_{mouse_id.lower()}"
    path = REPO_ROOT / "config" / f"m0_sensory_v1{suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"No task-moments config for mouse {mouse_id!r}: {path}")
    return json.loads(path.read_text())["task_moments"]


def _prior_sensory_component_rows(
    model_id: str, window_id: int, timeline: pd.DataFrame, moments: dict,
) -> tuple[list[dict], list[dict]]:
    """Per-window (bin, difficulty-condition) aggregates of the pre-lapse
    decision decomposition ``Lambda(x) = sensory_logit(x) + prior_logit -
    log(kappa)`` (see ``log_odds_coefficients`` in bayesian.py, numerically
    verified against this project's own saved ``model_decision_*_boundary_s``
    columns). ``sensory_logit`` uses the closed-form expectation of
    ``log[f_long(X)/f_short(X)]`` over the model's own fitted measurement
    noise ``X ~ Normal(duration_s, model_sensory_sd_s**2)``, since the true
    per-trial internal measurement is never observed. WSLS models have no
    sensory/prior decomposition at all and are skipped (empty return).
    """
    spec = MODEL_BY_ID[model_id]
    if "w" not in spec.parameter_names:
        return [], []
    evaluated = timeline.loc[_bool_mask(timeline.model_evaluated, "model_evaluated")].copy()
    if evaluated.empty:
        return [], []

    w = pd.to_numeric(evaluated.fitted_w, errors="raise").to_numpy(float)
    s0 = pd.to_numeric(evaluated.fitted_s0, errors="raise").to_numpy(float)
    duration = pd.to_numeric(evaluated.duration_s, errors="raise").to_numpy(float)
    sensory_sd = pd.to_numeric(evaluated.model_sensory_sd_s, errors="raise").to_numpy(float)
    if "model_prior_long" in evaluated.columns:
        prior_long = pd.to_numeric(evaluated.model_prior_long, errors="coerce").to_numpy(float)
        prior_long = np.where(np.isfinite(prior_long), prior_long, 0.5)
    else:
        prior_long = np.full(len(evaluated), 0.5)

    v_short = moments["var_short_s2"] + w * moments["mu_short_s"] ** 2 + s0 ** 2
    v_long = moments["var_long_s2"] + w * moments["mu_long_s"] ** 2 + s0 ** 2
    if np.any(v_short <= 0) or np.any(v_long <= 0):
        raise ValueError(f"{model_id} window {window_id} produced a nonpositive category variance")
    a2 = 0.5 / v_short - 0.5 / v_long
    a1 = moments["mu_long_s"] / v_long - moments["mu_short_s"] / v_short
    a0_sensory = (
        moments["mu_short_s"] ** 2 / (2 * v_short) - moments["mu_long_s"] ** 2 / (2 * v_long)
        + 0.5 * np.log(v_short / v_long)
    )
    sensory_logit = a2 * (duration ** 2 + sensory_sd ** 2) + a1 * duration + a0_sensory
    clipped_prior = np.clip(prior_long, 1e-12, 1 - 1e-12)
    prior_logit = np.log(clipped_prior) - np.log1p(-clipped_prior)
    if not (np.isfinite(sensory_logit).all() and np.isfinite(prior_logit).all()):
        raise ValueError(f"{model_id} window {window_id} produced a nonfinite prior/sensory logit")

    duration_key = np.round(duration, 2)
    bin_meta = [_DURATION_TO_BIN.get(value) for value in duration_key]
    if any(meta is None for meta in bin_meta):
        bad = sorted({value for value, meta in zip(duration_key, bin_meta) if meta is None})
        raise ValueError(f"{model_id} window {window_id} has unregistered durations for binning: {bad}")
    evaluated = evaluated.assign(
        _prior_logit=prior_logit, _sensory_logit=sensory_logit,
        _bin_label=[meta["label"] for meta in bin_meta],
        _difficulty=[meta["difficulty"] for meta in bin_meta],
    )

    bin_rows = []
    for meta in DURATION_BIN_DEFINITIONS:
        group = evaluated[evaluated["_bin_label"].eq(meta["label"])]
        bin_rows.append({
            "model_id": model_id, "window_id": window_id, "bin_label": meta["label"],
            "category": meta["category"], "difficulty": meta["difficulty"], "n_trials": len(group),
            "mean_abs_prior_logit": group["_prior_logit"].abs().mean() if len(group) else np.nan,
            "mean_abs_sensory_logit": group["_sensory_logit"].abs().mean() if len(group) else np.nan,
        })

    next_prior_by_trial = dict(zip(zip(evaluated.session_id, evaluated.trial_index), evaluated["_prior_logit"]))
    difficulty_rows = []
    for difficulty in ("hard", "easy"):
        own = evaluated[evaluated["_difficulty"].eq(difficulty)]
        difficulty_rows.append({
            "model_id": model_id, "window_id": window_id, "condition": f"{difficulty}_own",
            "n_trials": len(own),
            "mean_abs_prior_logit": own["_prior_logit"].abs().mean() if len(own) else np.nan,
        })
        next_values = [
            next_prior_by_trial[key]
            for key in zip(own.session_id, own.trial_index + 1)
            if key in next_prior_by_trial
        ]
        difficulty_rows.append({
            "model_id": model_id, "window_id": window_id, "condition": f"{difficulty}_next",
            "n_trials": len(next_values),
            "mean_abs_prior_logit": float(np.mean(np.abs(next_values))) if next_values else np.nan,
        })
    return bin_rows, difficulty_rows


def _read_stable(path: Path) -> pd.DataFrame:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Prediction file changed during extended collection: {path}")
    table = pd.read_csv(io.BytesIO(payload), usecols=lambda c: c in REQUIRED | OPTIONAL)
    missing = REQUIRED - set(table.columns)
    if missing:
        raise ValueError(f"Extended prediction schema missing {sorted(missing)}: {path}")
    return table


def _events(timeline: pd.DataFrame) -> list[dict]:
    events: list[dict] = []
    for session_id, session in timeline.groupby("session_id", sort=False):
        session = session.sort_values("trial_index")
        blocks = session.drop_duplicates("block_index").set_index("block_index")["true_block"]
        for _, row in session[session.true_run_length.eq(1)].iterrows():
            block = int(row.block_index)
            if block - 1 not in blocks.index:
                continue
            previous, current = blocks.loc[block - 1], blocks.loc[block]
            kind = "short_to_long" if (previous, current) == ("S", "L") else (
                "long_to_short" if (previous, current) == ("L", "S") else None
            )
            if kind:
                events.append({"session_id": session_id, "anchor": int(row.trial_index), "event": kind,
                               "allowed_blocks": (block - 1, block)})
        rare_short = session.category_long.eq(0) & session.true_block.eq("L")
        rare_long = session.category_long.eq(1) & session.true_block.eq("S")
        for row in session.loc[rare_short, ["trial_index"]].itertuples(index=False):
            block = int(session.loc[session.trial_index.eq(row.trial_index), "block_index"].iloc[0])
            events.append({"session_id": session_id, "anchor": int(row.trial_index), "event": "rare_short",
                           "allowed_blocks": (block,)})
        for row in session.loc[rare_long, ["trial_index"]].itertuples(index=False):
            block = int(session.loc[session.trial_index.eq(row.trial_index), "block_index"].iloc[0])
            events.append({"session_id": session_id, "anchor": int(row.trial_index), "event": "rare_long",
                           "allowed_blocks": (block,)})
    return events


def _event_rows(
    timeline: pd.DataFrame, model_id: str, window_id: int,
    range_overrides: dict[str, tuple[str, range]] | None = None,
) -> list[dict]:
    """Expand each event anchor into its window of relative-position rows.

    ``range_overrides``, when given, replaces the default wide+focus span
    selection with a single named span per event type. An event anchor is
    kept only if every offset in its selected span is evaluated and stays
    within the anchor's allowed block(s) (see the ``good_ids`` gate below),
    so narrowing the span here recovers anchors near session boundaries that
    the default (wider) span would have dropped entirely.
    """
    evaluated = _bool_mask(timeline.model_evaluated, "model_evaluated")
    work = timeline.copy()
    work["mouse_correct_derived"] = (work.choice_long == work.category_long).astype(float)
    p = pd.to_numeric(work.model_probability_long, errors="coerce")
    category = pd.to_numeric(work.category_long, errors="coerce")
    work["model_probability_correct_derived"] = np.where(category.eq(1), p, 1 - p)
    work.loc[~evaluated, ["mouse_correct_derived", "model_probability_correct_derived"]] = np.nan
    events = _events(work)
    if not events:
        return []
    expanded = []
    event_id = 0
    for event in events:
        if range_overrides is not None:
            ranges = (range_overrides[event["event"]],)
        else:
            ranges = (("wide", EVENT_RANGES[event["event"]]), ("focus", FOCUSED_TRANSITION_RANGE)) \
                if event["event"] in {"short_to_long", "long_to_short"} else (("focus", EVENT_RANGES[event["event"]]),)
        for event_span, offsets in ranges:
            for offset in offsets:
                expanded.append({"event_id": event_id, **event, "event_span": event_span,
                                 "relative_position": offset, "trial_index": event["anchor"] + offset})
            event_id += 1
    frame = pd.DataFrame(expanded).merge(
        work[["session_id", "trial_index", "block_index", "split_role", "category_long", "model_evaluated",
              "per_trial_negative_log_likelihood", "mouse_correct_derived",
              "model_probability_correct_derived"]],
        on=["session_id", "trial_index"], how="left", validate="many_to_one",
    )
    frame["evaluated"] = _bool_mask(frame.model_evaluated.fillna(False), "model_evaluated")
    frame["within_event_blocks"] = [
        pd.notna(block) and int(block) in allowed
        for block, allowed in zip(frame.block_index, frame.allowed_blocks, strict=True)
    ]
    good = frame.groupby("event_id").agg(evaluated=("evaluated", "all"),
                                           within_event_blocks=("within_event_blocks", "all"))
    good_ids = good.index[good.evaluated & good.within_event_blocks]
    frame = frame[frame.event_id.isin(good_ids)].copy()
    if frame.empty:
        return []
    frame["trial_type"] = np.where(frame.category_long.eq(1), "long", "short")
    frame["model_id"] = model_id
    frame["window_id"] = window_id
    frame = frame.rename(columns={"per_trial_negative_log_likelihood": "nll",
                                  "mouse_correct_derived": "mouse_correct",
                                  "model_probability_correct_derived": "model_probability_correct"})
    return frame.groupby(
        ["model_id", "window_id", "split_role", "event", "event_span", "relative_position", "trial_type"],
        as_index=False,
    ).agg(mean_nll=("nll", "mean"), mouse_correct=("mouse_correct", "mean"),
          model_probability_correct=("model_probability_correct", "mean"), n_events=("nll", "size")).to_dict("records")


def _parameter_rows(model_id: str, window_id: int, test_rank: int, winner: dict) -> list[dict]:
    spec = MODEL_BY_ID[model_id]
    values = winner.get("best_constrained_parameters")
    if not isinstance(values, list) or len(values) != len(spec.parameter_names):
        raise ValueError(f"{model_id} window {window_id} parameter vector length mismatch")
    numeric = np.asarray(values, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{model_id} window {window_id} parameter vector is nonfinite")
    rows = [{"model_id": model_id, "window_id": window_id, "test_rank": test_rank,
             "param_name": name, "param_value": value}
            for name, value in zip(spec.parameter_names, numeric, strict=True)]
    derived = {}
    if "a" in spec.parameter_names:
        derived["alpha"] = 1 / (1 + np.exp(-numeric[spec.parameter_names.index("a")]))
    if "eta_logit" in spec.parameter_names:
        derived["eta"] = 1 / (1 + np.exp(-numeric[spec.parameter_names.index("eta_logit")]))
    return rows + [{"model_id": model_id, "window_id": window_id, "test_rank": test_rank,
                    "param_name": name, "param_value": value} for name, value in derived.items()]


def _sensory_only_probability(model_id: str, evaluated: pd.DataFrame, moments: dict) -> np.ndarray:
    """P(choose long) driven by sensory evidence alone, computed with the
    model's own Gaussian decision-region integral (``integrate_general_
    decision_region``) -- not a sigmoid approximation -- with the prior
    fixed at neutral 0.5 and kappa fixed at neutral 1 (no belief, no
    decision-cost bias), lapse-mixed with the model's own fitted ell_short/
    ell_long exactly as the real pipeline does. NaN for models with no
    sensory term at all (WSLS).

    This exact code path (DecodedParameters -> category_variances ->
    log_odds_coefficients -> sensory_standard_deviation ->
    integrate_general_decision_region -> lapse mix), run with the model's
    REAL fitted kappa and REAL per-trial model_prior_long instead of these
    neutral stand-ins, was verified against this project's own stored
    model_probability_long/model_non_lapse_probability_long columns to
    float64 precision (max abs diff ~3e-15) before this function was
    written -- confirming both the correct source package is imported (see
    the sys.path guard above) and the formula is wired correctly.
    """
    spec = MODEL_BY_ID[model_id]
    if "w" not in spec.parameter_names:
        return np.full(len(evaluated), np.nan)
    moments_obj = TaskMoments(**moments)
    w = torch.tensor(pd.to_numeric(evaluated.fitted_w, errors="raise").to_numpy(float), dtype=torch.float64)
    s0 = torch.tensor(pd.to_numeric(evaluated.fitted_s0, errors="raise").to_numpy(float), dtype=torch.float64)
    ell_short = torch.tensor(
        pd.to_numeric(evaluated.fitted_ell_short, errors="raise").to_numpy(float), dtype=torch.float64,
    )
    ell_long = torch.tensor(
        pd.to_numeric(evaluated.fitted_ell_long, errors="raise").to_numpy(float), dtype=torch.float64,
    )
    duration = torch.tensor(pd.to_numeric(evaluated.duration_s, errors="raise").to_numpy(float), dtype=torch.float64)
    category_long = torch.tensor(
        pd.to_numeric(evaluated.category_long, errors="raise").to_numpy(float), dtype=torch.float64,
    )
    neutral_kappa = torch.ones_like(w)
    neutral_prior = torch.tensor(0.5, dtype=torch.float64)
    parameters = DecodedParameters(w=w, s0=s0, ell_short=ell_short, ell_long=ell_long, kappa=neutral_kappa)
    variances = category_variances(parameters, moments_obj)
    coefficients = log_odds_coefficients(variances, moments_obj, neutral_prior, parameters.log_kappa)
    sensory_sd = sensory_standard_deviation(duration, parameters)
    non_lapse_probability, _ = integrate_general_decision_region(duration, sensory_sd, coefficients)
    lapse_rate = torch.where(category_long == 1.0, ell_long, ell_short)
    probability = (1.0 - lapse_rate) * non_lapse_probability + 0.5 * lapse_rate
    result = probability.numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"{model_id} produced a nonfinite sensory-only probability")
    return result


def _per_trial_logits(model_id: str, window_id: int, timeline: pd.DataFrame, moments: dict) -> pd.DataFrame:
    """Every evaluated trial's sensory_logit/belief_logit/summation_logit in
    this (model, window) -- the full timeline, not restricted to any
    sampled session. Same closed-form split as
    ``_prior_sensory_component_rows`` (see its docstring for the derivation
    and its accuracy caveat), just kept per-trial instead of aggregated to
    bins, so Figure 20 can plot trajectories, event-aligned averages, and
    psychometric curves from it.

    A model contributes ``sensory_logit`` only if it has a sensory term at
    all (``"w" in parameter_names`` -- WSLS models don't), and
    ``belief_logit`` only if it has a block-structure prior (every model
    except M0, which has no belief/prior term to speak of). ``summation_logit``
    is whichever of the two a model has, or their sum if it has both.
    """
    spec = MODEL_BY_ID[model_id]
    has_sensory = "w" in spec.parameter_names
    has_belief = spec.group != "m0"
    evaluated = timeline.loc[_bool_mask(timeline.model_evaluated, "model_evaluated")].copy()
    if evaluated.empty:
        return evaluated.assign(sensory_logit=np.nan, belief_logit=np.nan, summation_logit=np.nan)

    sensory_logit = np.full(len(evaluated), np.nan)
    if has_sensory:
        w = pd.to_numeric(evaluated.fitted_w, errors="raise").to_numpy(float)
        s0 = pd.to_numeric(evaluated.fitted_s0, errors="raise").to_numpy(float)
        duration = pd.to_numeric(evaluated.duration_s, errors="raise").to_numpy(float)
        sensory_sd = pd.to_numeric(evaluated.model_sensory_sd_s, errors="raise").to_numpy(float)
        v_short = moments["var_short_s2"] + w * moments["mu_short_s"] ** 2 + s0 ** 2
        v_long = moments["var_long_s2"] + w * moments["mu_long_s"] ** 2 + s0 ** 2
        if np.any(v_short <= 0) or np.any(v_long <= 0):
            raise ValueError(f"{model_id} window {window_id} produced a nonpositive category variance")
        a2 = 0.5 / v_short - 0.5 / v_long
        a1 = moments["mu_long_s"] / v_long - moments["mu_short_s"] / v_short
        a0_sensory = (
            moments["mu_short_s"] ** 2 / (2 * v_short) - moments["mu_long_s"] ** 2 / (2 * v_long)
            + 0.5 * np.log(v_short / v_long)
        )
        sensory_logit = a2 * (duration ** 2 + sensory_sd ** 2) + a1 * duration + a0_sensory
        if not np.isfinite(sensory_logit).all():
            raise ValueError(f"{model_id} window {window_id} produced a nonfinite sensory logit")

    belief_logit = np.full(len(evaluated), np.nan)
    if has_belief:
        prior_long = pd.to_numeric(evaluated.model_prior_long, errors="raise").to_numpy(float)
        if not np.isfinite(prior_long).all() or np.any((prior_long < 0) | (prior_long > 1)):
            raise ValueError(f"{model_id} window {window_id} produced a nonfinite/out-of-range model_prior_long")
        clipped = np.clip(prior_long, 1e-12, 1 - 1e-12)
        belief_logit = np.log(clipped) - np.log1p(-clipped)

    summation_logit = np.nansum(np.vstack([sensory_logit, belief_logit]), axis=0)
    sensory_probability = _sensory_only_probability(model_id, evaluated, moments)
    return evaluated.assign(
        sensory_logit=sensory_logit, belief_logit=belief_logit, summation_logit=summation_logit,
        sensory_probability=sensory_probability,
    )


def _figure20_logit_rows(
    model_id: str, window_id: int, per_trial: pd.DataFrame, session_by_split: dict[str, str],
) -> list[dict]:
    """The one session Figure 20 (pages 1-2) samples per split in this
    window, sliced out of ``per_trial`` (see ``_per_trial_logits``)."""
    if per_trial.empty:
        return []
    mask = pd.Series(False, index=per_trial.index)
    for split, session_id in session_by_split.items():
        mask |= per_trial.split_role.eq(split) & per_trial.session_id.eq(session_id)
    chosen = per_trial.loc[mask]
    if chosen.empty:
        return []
    return [
        {
            "model_id": model_id, "window_id": window_id, "split_role": row.split_role,
            "session_id": row.session_id, "trial_index": row.trial_index, "true_block": row.true_block,
            "sensory_logit": row.sensory_logit, "belief_logit": row.belief_logit,
            "summation_logit": row.summation_logit,
        }
        for row in chosen.itertuples()
    ]


def _logit_event_rows(
    timeline: pd.DataFrame, per_trial: pd.DataFrame, model_id: str, window_id: int,
) -> list[dict]:
    """Per-window, event-aligned means of sensory_logit/belief_logit/
    summation_logit around block transitions and rare trials -- the same
    event geometry ``_event_rows`` uses for NLL, so Figure 20's pages 3-4
    can average these across windows exactly like Figures 16/17/19/23
    average NLL/correctness (see ``_event_grid``/``_window_weighted``).
    """
    if per_trial.empty:
        return []
    events = _events(timeline)
    if not events:
        return []
    expanded = []
    event_id = 0
    for event in events:
        ranges = (("wide", EVENT_RANGES[event["event"]]), ("focus", FOCUSED_TRANSITION_RANGE)) \
            if event["event"] in {"short_to_long", "long_to_short"} else (("focus", EVENT_RANGES[event["event"]]),)
        for event_span, offsets in ranges:
            for offset in offsets:
                expanded.append({"event_id": event_id, **event, "event_span": event_span,
                                 "relative_position": offset, "trial_index": event["anchor"] + offset})
            event_id += 1
    frame = pd.DataFrame(expanded).merge(
        per_trial[["session_id", "trial_index", "block_index", "split_role",
                   "sensory_logit", "belief_logit", "summation_logit"]],
        on=["session_id", "trial_index"], how="left", validate="many_to_one",
    )
    frame["within_event_blocks"] = [
        pd.notna(block) and int(block) in allowed
        for block, allowed in zip(frame.block_index, frame.allowed_blocks, strict=True)
    ]
    frame["evaluated"] = frame.summation_logit.notna()
    good = frame.groupby("event_id").agg(evaluated=("evaluated", "all"),
                                           within_event_blocks=("within_event_blocks", "all"))
    good_ids = good.index[good.evaluated & good.within_event_blocks]
    frame = frame[frame.event_id.isin(good_ids)].copy()
    if frame.empty:
        return []
    frame["model_id"] = model_id
    frame["window_id"] = window_id
    return frame.groupby(
        ["model_id", "window_id", "split_role", "event", "event_span", "relative_position"], as_index=False,
    ).agg(mean_sensory_logit=("sensory_logit", "mean"), mean_belief_logit=("belief_logit", "mean"),
          mean_summation_logit=("summation_logit", "mean"), n_events=("summation_logit", "size")).to_dict("records")


def _logit_psychometric_rows(per_trial: pd.DataFrame, model_id: str, window_id: int) -> list[dict]:
    """Per-window mean P(long) by presented duration, split role, and true
    block. No sigmoid anywhere: ``sensory_probability`` is the model's own
    Gaussian decision-region integral (see ``_sensory_only_probability``,
    already attached to ``per_trial``); ``belief_probability`` is the
    model's own ``model_prior_long`` used as-is (sigmoid(logit(p)) is
    exactly p, so the earlier logit round-trip was a no-op); and
    ``summation_probability`` is the model's own ``model_probability_long``
    used as-is -- the real full probability, not an approximation (unlike
    the logit-sum used in pages 3-4). The per-trial probability is averaged
    first (matching ``extended_psychometrics.csv``'s ``model_long_rate``
    convention), not the logit, to avoid Jensen's-inequality bias.
    """
    if per_trial.empty:
        return []
    working = per_trial.copy()
    working["belief_probability"] = (
        working["model_prior_long"] if "model_prior_long" in working.columns else np.nan
    )
    working["summation_probability"] = working["model_probability_long"]
    working["model_id"] = model_id
    working["window_id"] = window_id
    return working.groupby(
        ["model_id", "window_id", "split_role", "true_block", "duration_s"], as_index=False,
    ).agg(n_trials=("summation_probability", "size"), sensory_probability=("sensory_probability", "mean"),
          belief_probability=("belief_probability", "mean"),
          summation_probability=("summation_probability", "mean")).to_dict("records")


def collect(mouse_id: str, status: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _, schedule = canonical_schedule(mouse_id)
    eligible = tuple(model for model in MODEL_ORDER if bool(
        status.set_index("model_id").loc[model, "eligible_for_comparison"]
    ))
    subset_rows, event_rows, parameter_rows, psych_rows, figure20_rows = [], [], [], [], []
    prior_sensory_bin_rows, prior_sensory_difficulty_rows = [], []
    logit_event_rows, logit_psych_rows = [], []
    task_moments = _task_moments(mouse_id)
    reference_metadata: dict[int, pd.DataFrame] = {}
    # Figure 20 illustrates 5 randomly drawn windows (fixed seed: reproducible
    # across re-runs, not re-randomized every collect() call), and within
    # each window one randomly drawn train and validation session (the test
    # session is deterministic -- a window has only one).
    window_sample_rng = np.random.default_rng(FIGURE20_WINDOW_SAMPLE_SEED)
    figure20_window_ids = sorted(
        window_sample_rng.choice(
            sorted(schedule), size=min(FIGURE20_WINDOW_SAMPLE_COUNT, len(schedule)), replace=False,
        ).tolist()
    )
    figure20_sessions_by_window = {}
    for window_id in figure20_window_ids:
        item = schedule[window_id]
        figure20_sessions_by_window[window_id] = {
            "train": str(window_sample_rng.choice(item["training_session_ids"])),
            "validation": str(window_sample_rng.choice(item["validation_session_ids"])),
            "test": item["test_session_id"],
        }
    for model_id in eligible:
        spec = MODEL_BY_ID[model_id]
        directories = _window_directories(spec.result_root(mouse_id))
        for window_id in sorted(schedule):
            directory = directories[window_id]
            winner = _converged_winner(directory)
            if winner is None or winner.get("converged") is not True:
                raise ValueError(f"Eligible {model_id} window {window_id} lost its converged winner")
            parameter_rows.extend(_parameter_rows(model_id, window_id, schedule[window_id]["test_rank"], winner))
            timeline = _read_stable(directory / "trial_predictions.csv")
            evaluated = timeline.loc[_bool_mask(timeline.model_evaluated, "model_evaluated")].copy()
            metadata = evaluated[list(STRUCTURAL_COLUMNS)].sort_values(
                ["split_role", "session_id", "source_trial_index"]
            ).reset_index(drop=True)
            if window_id in reference_metadata and not metadata.equals(reference_metadata[window_id]):
                raise ValueError(f"Extended analysis trial structure differs in window {window_id}")
            reference_metadata.setdefault(window_id, metadata)
            evaluated["mouse_correct"] = (evaluated.choice_long == evaluated.category_long).astype(float)
            probability = pd.to_numeric(evaluated.model_probability_long, errors="raise")
            evaluated["model_probability_correct"] = np.where(evaluated.category_long.eq(1), probability, 1 - probability)
            for split in SPLIT_ROLES:
                split_data = evaluated[evaluated.split_role.eq(split)]
                for name, mask in TRIAL_SUBSETS.items():
                    values = split_data.loc[mask(split_data), "per_trial_negative_log_likelihood"].astype(float)
                    subset_rows.append({"model_id": model_id, "window_id": window_id, "split_role": split,
                                        "trial_subset": name, "n_trials": len(values),
                                        "mean_nll": values.mean() if len(values) else np.nan,
                                        "nll_sem": values.sem() if len(values) > 1 else np.nan})
            event_rows.extend(_event_rows(timeline, model_id, window_id))
            new_bin_rows, new_difficulty_rows = _prior_sensory_component_rows(
                model_id, window_id, timeline, task_moments,
            )
            prior_sensory_bin_rows.extend(new_bin_rows)
            prior_sensory_difficulty_rows.extend(new_difficulty_rows)
            heldout = evaluated[evaluated.split_role.eq("test")].copy()
            heldout["duration_bin"] = heldout.duration_s.round(3)
            for (block, duration), group in heldout.groupby(["true_block", "duration_bin"]):
                psych_rows.append({"model_id": model_id, "window_id": window_id, "block": block,
                                   "duration_s": duration, "n_trials": len(group),
                                   "mouse_long_rate": group.choice_long.mean(),
                                   "model_long_rate": group.model_probability_long.mean()})
            per_trial = _per_trial_logits(model_id, window_id, timeline, task_moments)
            if window_id in figure20_sessions_by_window:
                figure20_rows.extend(_figure20_logit_rows(
                    model_id, window_id, per_trial, figure20_sessions_by_window[window_id],
                ))
            logit_event_rows.extend(_logit_event_rows(timeline, per_trial, model_id, window_id))
            logit_psych_rows.extend(_logit_psychometric_rows(per_trial, model_id, window_id))
        expected_fingerprint = str(status.set_index("model_id").loc[model_id, "source_snapshot_sha256"])
        if source_fingerprint(spec, mouse_id, set(schedule)) != expected_fingerprint:
            raise RuntimeError(f"Audited source snapshot changed during extended collection: {model_id}")
    rows_by_file = {
        "extended_trial_subset_metrics.csv": subset_rows,
        "extended_event_aligned.csv": event_rows,
        "extended_parameters.csv": parameter_rows,
        "extended_psychometrics.csv": psych_rows,
        "extended_prior_logit_samples.csv": figure20_rows,
        "extended_logit_event_aligned.csv": logit_event_rows,
        "extended_logit_psychometrics.csv": logit_psych_rows,
        "extended_prior_sensory_bins.csv": prior_sensory_bin_rows,
        "extended_prior_sensory_difficulty.csv": prior_sensory_difficulty_rows,
    }
    return {name: pd.DataFrame(rows, columns=TABLE_COLUMNS[name]) for name, rows in rows_by_file.items()}


# Figure 26 needs narrower per-event windows than the shared extraction above
# (rare trials: 1 before/1 after; transitions: 2 before/4 after). Reusing the
# already-collected extended_event_aligned.csv by filtering relative_position
# would silently under-count: an anchor is kept there only if its full wide
# window survives (see the all-or-nothing gate in _event_rows), so an anchor
# that fails only outside this narrower window is entirely absent from that
# file. This collects a dedicated, correctly-windowed table instead.
FIGURE26_EVENT_RANGES: dict[str, tuple[str, range]] = {
    "rare_short": ("focus", range(-1, 2)),
    "rare_long": ("focus", range(-1, 2)),
    "short_to_long": ("focus", range(-2, 5)),
    "long_to_short": ("focus", range(-2, 5)),
}


def collect_figure26_events(mouse_id: str, status: pd.DataFrame) -> pd.DataFrame:
    """Event-aligned NLL for Figure 26, using FIGURE26_EVENT_RANGES windows.

    Mirrors collect()'s cross-model structural check: every model must share
    the identical evaluated-trial structure per window (same STRUCTURAL_COLUMNS
    rows), so that per-model event-conditioned means in a window are computed
    over the same underlying trials/event instances and remain validly paired.
    """
    _, schedule = canonical_schedule(mouse_id)
    eligible = tuple(model for model in MODEL_ORDER if bool(
        status.set_index("model_id").loc[model, "eligible_for_comparison"]
    ))
    event_rows: list[dict] = []
    reference_metadata: dict[int, pd.DataFrame] = {}
    for model_id in eligible:
        spec = MODEL_BY_ID[model_id]
        directories = _window_directories(spec.result_root(mouse_id))
        for window_id in sorted(schedule):
            directory = directories[window_id]
            winner = _converged_winner(directory)
            if winner is None or winner.get("converged") is not True:
                raise ValueError(f"Eligible {model_id} window {window_id} lost its converged winner")
            timeline = _read_stable(directory / "trial_predictions.csv")
            evaluated = timeline.loc[_bool_mask(timeline.model_evaluated, "model_evaluated")]
            metadata = evaluated[list(STRUCTURAL_COLUMNS)].sort_values(
                ["split_role", "session_id", "source_trial_index"]
            ).reset_index(drop=True)
            if window_id in reference_metadata and not metadata.equals(reference_metadata[window_id]):
                raise ValueError(f"Figure 26 trial structure differs in window {window_id}: {model_id}")
            reference_metadata.setdefault(window_id, metadata)
            event_rows.extend(
                _event_rows(timeline, model_id, window_id, range_overrides=FIGURE26_EVENT_RANGES)
            )
        expected_fingerprint = str(status.set_index("model_id").loc[model_id, "source_snapshot_sha256"])
        if source_fingerprint(spec, mouse_id, set(schedule)) != expected_fingerprint:
            raise RuntimeError(f"Audited source snapshot changed during figure26 collection: {model_id}")
    return pd.DataFrame(event_rows, columns=TABLE_COLUMNS["extended_event_aligned.csv"])
