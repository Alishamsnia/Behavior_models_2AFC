"""Paired predictive comparison between frozen M0 and M1 outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class M1VersusM0Metrics:
    """Paired choice-likelihood improvement on one development split."""

    split_role: str
    valid_trial_count: int
    m0_elpd: float
    m1_elpd: float
    delta_elpd_m1_minus_m0: float
    delta_bits_per_trial_m1_minus_m0: float
    per_session_delta_elpd: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict:
        """Return JSON-safe comparison values."""

        return asdict(self)


def compare_m1_to_m0_predictions(
    m0_predictions: pd.DataFrame,
    m1_predictions: pd.DataFrame,
    *,
    split_role: str = "validation",
    allow_test: bool = False,
) -> M1VersusM0Metrics:
    """Compare paired trial NLLs without refitting either model."""

    if split_role not in {"train", "validation", "test"}:
        raise ValueError("split_role must be train, validation, or test")
    if split_role == "test" and not allow_test:
        raise PermissionError("Test comparison requires allow_test=True")
    keys = ["mouse_id", "session_id", "trial_index", "split_role"]
    required = set(keys) | {
        "choice_long", "model_evaluated", "per_trial_negative_log_likelihood"
    }
    for name, frame in (("M0", m0_predictions), ("M1", m1_predictions)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"{name} prediction table is missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} prediction table has duplicate trial keys")
    columns = keys + [
        "choice_long", "model_evaluated", "per_trial_negative_log_likelihood"
    ]
    merged = m0_predictions[columns].merge(
        m1_predictions[columns],
        on=keys,
        how="inner",
        suffixes=("_m0", "_m1"),
        validate="one_to_one",
    )
    expected_count = int(m0_predictions["split_role"].eq(split_role).sum())
    observed = merged.loc[merged["split_role"].eq(split_role)].copy()
    if len(observed) != expected_count or len(observed) != int(
        m1_predictions["split_role"].eq(split_role).sum()
    ):
        raise ValueError("M0 and M1 trial timelines are not exactly aligned")
    choice0 = pd.to_numeric(observed["choice_long_m0"], errors="coerce")
    choice1 = pd.to_numeric(observed["choice_long_m1"], errors="coerce")
    if not np.allclose(choice0, choice1, equal_nan=True):
        raise ValueError("M0 and M1 observed choices disagree")
    valid0 = observed["model_evaluated_m0"].astype(bool)
    valid1 = observed["model_evaluated_m1"].astype(bool)
    if not valid0.equals(valid1):
        raise ValueError("M0 and M1 valid-choice masks disagree")
    paired = observed.loc[valid0].copy()
    if paired.empty:
        raise ValueError("No paired valid choices are available")
    nll0 = pd.to_numeric(
        paired["per_trial_negative_log_likelihood_m0"], errors="raise"
    ).to_numpy(float)
    nll1 = pd.to_numeric(
        paired["per_trial_negative_log_likelihood_m1"], errors="raise"
    ).to_numpy(float)
    if not np.all(np.isfinite(nll0)) or not np.all(np.isfinite(nll1)):
        raise FloatingPointError("Paired NLL values must be finite")
    paired["delta_elpd"] = nll0 - nll1
    per_session = tuple(
        (str(session_id), float(group["delta_elpd"].sum()))
        for session_id, group in paired.groupby("session_id", sort=False)
    )
    delta = float(np.sum(nll0 - nll1))
    return M1VersusM0Metrics(
        split_role=split_role,
        valid_trial_count=len(paired),
        m0_elpd=-float(np.sum(nll0)),
        m1_elpd=-float(np.sum(nll1)),
        delta_elpd_m1_minus_m0=delta,
        delta_bits_per_trial_m1_minus_m0=delta / (len(paired) * np.log(2.0)),
        per_session_delta_elpd=per_session,
    )


def save_model_comparison(result: M1VersusM0Metrics, path: str | Path) -> None:
    """Atomically save a paired M1-versus-M0 comparison."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
