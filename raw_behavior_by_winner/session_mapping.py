"""Map each subject's Bayesian single-session model winner onto that
subject's raw trial data, by session date.

Reads (never writes to):
  - results_post_analysis/model_comparison/<SUBJECT>/tables/single_session_win_counts_per_session.csv
  - <raw_behavior_root>/<SUBJECT>_final_data.csv  (original Behavior/New_version codebase)

A subject only contributes once it has a completed single-session
model-comparison win-count table; every other subject in the raw-behavior
codebase is skipped ("no data yet"). Adding a new subject later needs no
code change here -- it is picked up automatically the next time this module
runs, as soon as its win-count table exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4"]

# raw_behavior_by_winner/session_mapping.py -> Bayesian_model_v2 repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPARISON_ROOT = REPO_ROOT / "results_post_analysis" / "model_comparison"

# Original raw-behavior codebase. Read-only: never copied from or written to
# by anything in this package.
DEFAULT_RAW_BEHAVIOR_ROOT = Path(
    "/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Behavior/New_version"
)

_SESSION_DATE_RE = re.compile(r"_(\d{8})_\d{6}$")


def date_from_session_id(session_id: str) -> str:
    """Extract the YYYYMMDD date embedded in a Bayesian-pipeline session_id,
    e.g. 'MC06_SChR2_..._20260112_183827' -> '20260112'.
    """
    match = _SESSION_DATE_RE.search(str(session_id))
    if not match:
        raise ValueError(f"Could not parse a date out of session_id={session_id!r}")
    return match.group(1)


def available_subjects() -> list[str]:
    """Subjects with a completed single-session model-comparison win-count
    table, sorted. Only these ever contribute to a model-winner folder.
    """
    if not MODEL_COMPARISON_ROOT.is_dir():
        return []
    subjects = []
    for child in sorted(MODEL_COMPARISON_ROOT.iterdir()):
        win_table = child / "tables" / "single_session_win_counts_per_session.csv"
        if win_table.is_file():
            subjects.append(child.name)
    return subjects


def _raw_session_idx_by_date(subject: str, raw_behavior_root: Path) -> pd.Series:
    """raw session_idx indexed by session_date (one row per physical raw
    session) for one subject's raw trial CSV.
    """
    raw_csv = raw_behavior_root / f"{subject}_final_data.csv"
    if not raw_csv.is_file():
        raise FileNotFoundError(
            f"No raw behavior CSV for subject {subject!r} at {raw_csv}. "
            "This subject has model results but no matching raw-behavior input."
        )
    df = pd.read_csv(raw_csv, usecols=["session_idx", "session_date"])
    df = df.drop_duplicates().copy()
    df["session_date"] = df["session_date"].astype(str).str.zfill(8)
    duplicated_dates = df.loc[df["session_date"].duplicated(), "session_date"]
    if not duplicated_dates.empty:
        raise ValueError(
            f"{subject}: multiple raw session_idx share the same session_date "
            f"{sorted(duplicated_dates.unique())}; date-based mapping is ambiguous "
            "and needs a more specific key (e.g. session time) to resolve."
        )
    return df.set_index("session_date")["session_idx"].astype(int)


def subject_winner_assignments(
    subject: str, raw_behavior_root: Path = DEFAULT_RAW_BEHAVIOR_ROOT,
) -> pd.DataFrame:
    """One row per comparable session for `subject`: which model won, plus
    everything needed to filter that subject's raw trial CSV down to it.

    Interpretation caveat: `winning_model` here comes from each session's
    held-out choice likelihood (`single_session_win_counts_per_session.csv`,
    an argmin over models' heldout mean NLL). Any downstream behavioral
    analysis that groups sessions by this winner and then compares those
    groups' behavior (lick rates, learning curves, psychometric slopes,
    etc.) is therefore analyzing a winner-selected sample, not an
    independent replication: the same held-out choices used to pick the
    winner are drawn from the session being described, so a difference
    between winner groups can partly reflect that selection rather than a
    genuine behavioral difference between mice/sessions the models
    happened to fit best. Treat such comparisons as descriptive, not as
    independent validation of a model's account of behavior.
    """
    win_table_path = (
        MODEL_COMPARISON_ROOT / subject / "tables" / "single_session_win_counts_per_session.csv"
    )
    wins = pd.read_csv(win_table_path)
    date_to_idx = _raw_session_idx_by_date(subject, raw_behavior_root)

    rows = []
    for _, row in wins.iterrows():
        session_id = row["session_id"]
        date = date_from_session_id(session_id)
        if date not in date_to_idx.index:
            raise ValueError(
                f"{subject}: session_id={session_id!r} (date {date}) has no matching "
                f"raw session_idx in {subject}_final_data.csv."
            )
        rows.append({
            "subject": subject,
            "model_id": row["winning_model"],
            "session_id": session_id,
            "chronological_rank": int(row["chronological_rank"]),
            "raw_session_idx": int(date_to_idx.loc[date]),
            "session_date": date,
        })
    return pd.DataFrame(rows)


def all_winner_assignments(raw_behavior_root: Path = DEFAULT_RAW_BEHAVIOR_ROOT) -> pd.DataFrame:
    """Winner assignments across every subject with model results, pooled.
    Raises if no subject has model results at all (nothing to do).
    """
    subjects = available_subjects()
    if not subjects:
        raise RuntimeError(
            f"No subjects with single-session model results found under "
            f"{MODEL_COMPARISON_ROOT}. Run the model-comparison post-analysis first."
        )
    frames = [subject_winner_assignments(s, raw_behavior_root) for s in subjects]
    return pd.concat(frames, ignore_index=True)


def sessions_won_by_model(
    model_id: str, raw_behavior_root: Path = DEFAULT_RAW_BEHAVIOR_ROOT,
) -> pd.DataFrame:
    """Rows of all_winner_assignments() where `model_id` won, across every
    subject with model results. May be empty (a model can win zero sessions).
    """
    assignments = all_winner_assignments(raw_behavior_root)
    return assignments.loc[assignments["model_id"] == model_id].reset_index(drop=True)
