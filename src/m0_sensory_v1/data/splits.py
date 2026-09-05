"""Frozen per-mouse chronological session splitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SPLIT_PROPORTIONS: dict[str, float] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
LARGEST_REMAINDER_TIE_PRIORITY: tuple[str, ...] = (
    "test",
    "validation",
    "train",
)
DEFAULT_SPLIT_VERSION = "chronological_70_15_15_v1"
SPLIT_ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "mouse_id",
    "session_id",
    "session_date",
    "chronological_rank",
    "split_role",
    "split_version",
)


def largest_remainder_counts(
    session_count: int,
    *,
    proportions: Mapping[str, float] = DEFAULT_SPLIT_PROPORTIONS,
    tie_priority: Sequence[str] = LARGEST_REMAINDER_TIE_PRIORITY,
) -> dict[str, int]:
    r"""Allocate integer split sizes by the registered largest-remainder rule.

    Purpose
    -------
    Convert fractional 70/15/15 targets into deterministic integer counts for
    one mouse without arbitrary rounding.

    Mathematical definition
    -----------------------
    For :math:`D` sessions and role fraction :math:`r_j`, calculate

    .. math::

       q_j=r_jD,\qquad n_j=\lfloor q_j\rfloor.

    Assign each remaining session to the largest fractional remainder
    :math:`q_j-\lfloor q_j\rfloor`. Exact ties follow
    ``test -> validation -> train``.

    Parameter interpretation
    ------------------------
    The tie priority protects the locked test period from accidental
    undersizing; it is not a random seed.

    Input units and shapes
    ----------------------
    ``session_count`` is a nonnegative integer. Proportions map the three roles
    to dimensionless nonnegative fractions summing to one.

    Returned quantities and shapes
    ------------------------------
    Returns integer counts keyed by ``train``, ``validation``, and ``test``;
    their sum is exactly ``session_count``.

    Constraints
    -----------
    Every role must appear exactly once in ``tie_priority`` and proportions
    must sum to one within floating tolerance.

    Numerical stability
    -------------------
    Fractional remainders are compared in double precision. The explicit tie
    rank resolves exact equal values deterministically.

    Causal restrictions
    -------------------
    Counts depend only on the number of sessions, never trial outcomes,
    durations, choices, blocks, or fitted model performance.

    Important edge cases
    --------------------
    Small datasets may receive zero validation or test sessions. For three
    sessions, the tied validation/test remainder is assigned to test first.

    Related tests
    -------------
    ``test_largest_remainder_counts_and_tie_priority``.
    """

    if not isinstance(session_count, int) or isinstance(session_count, bool):
        raise TypeError("session_count must be an integer")
    if session_count < 0:
        raise ValueError("session_count must be nonnegative")
    roles = ("train", "validation", "test")
    if set(proportions) != set(roles):
        raise ValueError(f"proportions must define exactly {roles}")
    if set(tie_priority) != set(roles) or len(tie_priority) != len(roles):
        raise ValueError("tie_priority must contain each split role exactly once")
    values = {role: float(proportions[role]) for role in roles}
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("Split proportions must be finite and nonnegative")
    if not np.isclose(sum(values.values()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Split proportions must sum to one")

    quotas = {role: session_count * values[role] for role in roles}
    counts = {role: int(np.floor(quotas[role])) for role in roles}
    remaining = session_count - sum(counts.values())
    tie_rank = {role: index for index, role in enumerate(tie_priority)}
    allocation_order = sorted(
        roles,
        key=lambda role: (-(quotas[role] - counts[role]), tie_rank[role]),
    )
    for role in allocation_order[:remaining]:
        counts[role] += 1
    return counts


def create_chronological_splits(
    sessions: pd.DataFrame,
    *,
    split_version: str = DEFAULT_SPLIT_VERSION,
    proportions: Mapping[str, float] = DEFAULT_SPLIT_PROPORTIONS,
) -> pd.DataFrame:
    r"""Assign chronological train/validation/locked-test sessions per mouse.

    Purpose
    -------
    Freeze split roles using session metadata only, independently for each
    mouse.

    Mathematical definition
    -----------------------
    For each mouse, sort by ``(session_date, session_id)``, allocate counts with
    :func:`largest_remainder_counts`, then assign the earliest :math:`n_{train}`
    sessions to train, the next :math:`n_{validation}` to validation, and the
    latest :math:`n_{test}` to locked test.

    Parameter interpretation
    ------------------------
    ``session_id`` is the stable deterministic tie-breaker when timestamps are
    identical. ``split_version`` identifies this frozen assignment policy.

    Input units and shapes
    ----------------------
    Input has one row per session and columns ``mouse_id``, ``session_id``, and
    ``session_date``. Trial data are neither required nor inspected.

    Returned quantities and shapes
    ------------------------------
    Returns one row per input session with the six registered assignment
    columns, including one-based chronological rank within mouse.

    Constraints
    -----------
    Mouse/session pairs must be unique and dates/identifiers must be present.

    Numerical stability
    -------------------
    Dates are normalized with pandas and sorting uses stable mergesort.

    Causal restrictions
    -------------------
    Outcomes, choices, durations, task categories, model predictions, and
    validation performance cannot influence the split.

    Important edge cases
    --------------------
    Exact timestamp ties are resolved by session ID. Each mouse receives its
    own largest-remainder allocation, even when mice have different counts.

    Related tests
    -------------
    ``test_chronological_split_is_per_mouse_stable_and_disjoint`` and the frozen
    MC08 assignment audit.
    """

    required = {"mouse_id", "session_id", "session_date"}
    missing = sorted(required.difference(sessions.columns))
    if missing:
        raise KeyError(f"Session table is missing columns: {missing}")
    if sessions.empty:
        raise ValueError("Cannot split an empty session table")
    if not split_version:
        raise ValueError("split_version must be nonempty")

    metadata = sessions.loc[:, ["mouse_id", "session_id", "session_date"]].copy()
    if metadata[["mouse_id", "session_id"]].isna().any().any():
        raise ValueError("Session identifiers cannot be missing")
    if metadata.duplicated(["mouse_id", "session_id"]).any():
        raise ValueError("Mouse/session identifiers must be unique")
    metadata["session_date"] = pd.to_datetime(metadata["session_date"], errors="raise")
    if metadata["session_date"].isna().any():
        raise ValueError("Session dates cannot be missing")

    frames: list[pd.DataFrame] = []
    for _, mouse_sessions in metadata.groupby("mouse_id", sort=True, observed=False):
        ordered = mouse_sessions.sort_values(
            ["session_date", "session_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        counts = largest_remainder_counts(len(ordered), proportions=proportions)
        roles = (
            ["train"] * counts["train"]
            + ["validation"] * counts["validation"]
            + ["test"] * counts["test"]
        )
        ordered["chronological_rank"] = np.arange(1, len(ordered) + 1)
        ordered["split_role"] = roles
        ordered["split_version"] = split_version
        frames.append(ordered)

    assignments = pd.concat(frames, ignore_index=True).loc[
        :,
        SPLIT_ASSIGNMENT_COLUMNS,
    ]
    validate_chronological_splits(
        assignments,
        split_version=split_version,
        proportions=proportions,
    )
    return assignments


def validate_chronological_splits(
    assignments: pd.DataFrame,
    *,
    split_version: str = DEFAULT_SPLIT_VERSION,
    proportions: Mapping[str, float] = DEFAULT_SPLIT_PROPORTIONS,
) -> None:
    """Validate counts, ordering, uniqueness, roles, and version invariants."""

    missing = sorted(set(SPLIT_ASSIGNMENT_COLUMNS).difference(assignments.columns))
    if missing:
        raise KeyError(f"Split assignments are missing columns: {missing}")
    if assignments.empty:
        raise ValueError("Split assignments are empty")
    if assignments.duplicated(["mouse_id", "session_id"]).any():
        raise ValueError("Split mouse/session identifiers are not unique")
    if set(assignments["split_role"].unique()).difference(
        {"train", "validation", "test"}
    ):
        raise ValueError("Split assignments contain an unknown role")
    if not assignments["split_version"].eq(split_version).all():
        raise ValueError("Split version does not match the registered version")

    dated = assignments.copy()
    dated["session_date"] = pd.to_datetime(dated["session_date"], errors="raise")
    for mouse_id, group in dated.groupby("mouse_id", sort=False, observed=False):
        ordered = group.sort_values("chronological_rank").reset_index(drop=True)
        expected_rank = np.arange(1, len(ordered) + 1)
        if not np.array_equal(ordered["chronological_rank"].to_numpy(), expected_rank):
            raise ValueError(f"Nonconsecutive chronological ranks for {mouse_id}")
        chronology = group.sort_values(
            ["session_date", "session_id"],
            kind="mergesort",
        )["session_id"].tolist()
        if ordered["session_id"].tolist() != chronology:
            raise ValueError(f"Chronological ordering mismatch for {mouse_id}")
        expected_counts = largest_remainder_counts(
            len(ordered),
            proportions=proportions,
        )
        observed_counts = ordered["split_role"].value_counts().to_dict()
        for role, expected in expected_counts.items():
            if int(observed_counts.get(role, 0)) != expected:
                raise ValueError(f"Incorrect {role} count for {mouse_id}")
        expected_roles = (
            ["train"] * expected_counts["train"]
            + ["validation"] * expected_counts["validation"]
            + ["test"] * expected_counts["test"]
        )
        if ordered["split_role"].tolist() != expected_roles:
            raise ValueError(f"Nonchronological role blocks for {mouse_id}")


def select_session_sources(
    session_audit: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    roles: Sequence[str] = ("train",),
    allow_locked_test: bool = False,
) -> list[Path]:
    """Select raw source files by frozen role before trial data are loaded."""

    requested = tuple(roles)
    if not requested or set(requested).difference({"train", "validation", "test"}):
        raise ValueError("roles must be a nonempty subset of train/validation/test")
    if "test" in requested and not allow_locked_test:
        raise ValueError("Locked test sources cannot be selected during development")
    if "split_version" not in assignments.columns:
        raise KeyError("Split assignments are missing column: split_version")
    versions = assignments["split_version"].dropna().unique()
    if len(versions) != 1:
        raise ValueError("Split assignments must contain exactly one split version")
    validate_chronological_splits(
        assignments,
        split_version=str(versions[0]),
    )
    required = {"mouse_id", "session_id", "source_file"}
    missing = sorted(required.difference(session_audit.columns))
    if missing:
        raise KeyError(f"Session audit is missing columns: {missing}")
    if session_audit.duplicated(["mouse_id", "session_id"]).any():
        raise ValueError("Session audit mouse/session identifiers are not unique")

    merged = assignments.merge(
        session_audit.loc[:, ["mouse_id", "session_id", "source_file"]],
        on=["mouse_id", "session_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["source_file"].isna().any():
        raise ValueError("Some frozen split sessions have no source file")
    selected = merged.loc[merged["split_role"].isin(requested)].sort_values(
        ["mouse_id", "chronological_rank"]
    )
    return [Path(path).expanduser().resolve() for path in selected["source_file"]]
