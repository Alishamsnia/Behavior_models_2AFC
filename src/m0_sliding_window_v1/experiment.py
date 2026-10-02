"""Data contracts for the M0 9/3/1 chronological sliding-window experiment.

This module is intentionally separate from :mod:`m0_sensory_v1`.  It changes
which sessions/trials are exposed to fitting, never M0's probability equation,
parameter transforms, optimizer, or registered eight-start table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from m0_sensory_v1.data import prepare_dataset


PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa")
LEGACY_PARTITIONED_WORKFLOW = "legacy_partitioned"
FULL_SESSION_MATCHED_WORKFLOW = "full_session_matched"
WORKFLOW_CHOICES = (LEGACY_PARTITIONED_WORKFLOW, FULL_SESSION_MATCHED_WORKFLOW)


def clamp_fine_tune_initialization(
    values: Sequence[float],
    parameter_names: Sequence[str],
    *,
    parameter_epsilon: float,
    lapse_max: float,
    margin: float = 1e-6,
) -> tuple[list[float], tuple[str, ...]]:
    """Move serialized fine-tuning starts just inside their open domains.

    A finite raw optimizer value can decode to an exact float64 boundary when
    its distance from that boundary is below machine precision. Re-encoding
    the reported constrained value would then fail. This adjustment applies
    only to the next optimizer's starting point; the source winner remains
    unchanged in initialization provenance.
    """

    if len(values) != len(parameter_names) or len(set(parameter_names)) != len(
        parameter_names
    ):
        raise ValueError("Initialization values and unique parameter names must align")
    if parameter_epsilon <= 0.0 or margin <= 0.0:
        raise ValueError("parameter_epsilon and margin must be positive")
    if not 0.0 < lapse_max <= 1.0 or 2.0 * margin >= lapse_max:
        raise ValueError("lapse_max must leave room for the requested open-bound margin")

    adjusted = np.asarray(values, dtype=float).copy()
    if not np.all(np.isfinite(adjusted)):
        raise ValueError("Fine-tuning initialization values must be finite")
    original = adjusted.copy()
    indices = {name: index for index, name in enumerate(parameter_names)}
    floor = parameter_epsilon + margin
    for name in ("w", "s0", "delta0", "bh"):
        if name in indices and adjusted[indices[name]] <= parameter_epsilon:
            adjusted[indices[name]] = floor
    for name in ("ell_short", "ell_long"):
        if name in indices:
            if adjusted[indices[name]] <= 0.0:
                adjusted[indices[name]] = margin
            elif adjusted[indices[name]] >= lapse_max:
                adjusted[indices[name]] = lapse_max - margin
    if "kappa" in indices and adjusted[indices["kappa"]] <= 0.0:
        adjusted[indices["kappa"]] = margin
    if "alpha_theta" in indices:
        if adjusted[indices["alpha_theta"]] <= 0.0:
            adjusted[indices["alpha_theta"]] = margin
        elif adjusted[indices["alpha_theta"]] >= 1.0:
            adjusted[indices["alpha_theta"]] = 1.0 - margin
    changed = tuple(
        name
        for index, name in enumerate(parameter_names)
        if adjusted[index] != original[index]
    )
    return adjusted.tolist(), changed


@dataclass(frozen=True)
class SlidingWindow:
    """One chronological 9-training/3-validation/1-test assignment."""

    window_id: int
    training_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    test_session_id: str
    training_ranks: tuple[int, ...]
    validation_ranks: tuple[int, ...]
    test_rank: int

    @property
    def directory_name(self) -> str:
        return (
            f"window_{self.window_id:03d}_"
            f"train_{self.training_ranks[0]:03d}-{self.training_ranks[-1]:03d}_"
            f"val_{self.validation_ranks[0]:03d}-{self.validation_ranks[-1]:03d}_"
            f"test_{self.test_rank:03d}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "training_session_ids": list(self.training_session_ids),
            "validation_session_ids": list(self.validation_session_ids),
            "test_session_id": self.test_session_id,
            "training_ranks": list(self.training_ranks),
            "validation_ranks": list(self.validation_ranks),
            "test_rank": self.test_rank,
        }


@dataclass(frozen=True)
class BlockPartition:
    """Auditable prefix/test/right-censored block partition for one session."""

    session_id: str
    prefix_block_indices: tuple[int, ...]
    heldout_block_indices: tuple[int, int]
    ignored_terminal_block_index: int | None
    heldout_start_trial: int
    heldout_stop_trial: int

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "prefix_block_indices": list(self.prefix_block_indices),
            "heldout_block_indices": list(self.heldout_block_indices),
            "ignored_terminal_block_index": self.ignored_terminal_block_index,
            "heldout_start_trial": self.heldout_start_trial,
            "heldout_stop_trial": self.heldout_stop_trial,
            "completion_policy": (
                "terminal block is conservatively right-censored; the preceding "
                "two transition-terminated blocks are held out"
                if self.ignored_terminal_block_index is not None
                else "literal final two chronological blocks are held out"
            ),
        }


@dataclass(frozen=True)
class PreparedSlidingData:
    """Complete chronological M0 experiment table and frozen partitions."""

    mouse_id: str
    assignments: pd.DataFrame
    canonical_trials: pd.DataFrame
    partitions: Mapping[str, BlockPartition]
    block_audit: pd.DataFrame
    source_files: tuple[str, ...]
    source_sha256: tuple[str, ...]

    def frame(self, session_id: str, scope: str, role: str) -> pd.DataFrame:
        """Return a copied session partition with an experiment split role."""

        if scope not in {
            "prefix",
            "heldout",
            "ignored",
            "full",
            "final_two_prefix",
            "final_two_heldout",
        }:
            raise ValueError(f"Unknown partition scope: {scope!r}")
        session = self.canonical_trials.loc[
            self.canonical_trials["session_id"].astype(str).eq(session_id)
        ].copy()
        if scope == "full":
            selected = session
            selected["experiment_partition"] = "full_session"
        elif scope in {"final_two_prefix", "final_two_heldout"}:
            partitioned, _, _ = partition_final_two_blocks(session)
            partition_name = (
                "prefix" if scope == "final_two_prefix" else "heldout"
            )
            selected = partitioned.loc[
                partitioned["experiment_partition"].eq(partition_name)
            ].copy()
        else:
            selected = session.loc[session["experiment_partition"].eq(scope)].copy()
        if selected.empty:
            raise ValueError(f"Session {session_id} has no {scope} rows")
        selected["split_role"] = role
        return selected.sort_values("trial_index").reset_index(drop=True)

    def final_two_block_audit(self) -> pd.DataFrame:
        """Return the literal-final-two partition audit for every session."""

        audits: list[pd.DataFrame] = []
        for assignment in self.assignments.sort_values(
            "chronological_rank", kind="mergesort"
        ).itertuples(index=False):
            session_id = str(assignment.session_id)
            session = self.canonical_trials.loc[
                self.canonical_trials["session_id"].astype(str).eq(session_id)
            ]
            _, _, audit = partition_final_two_blocks(session)
            audit["chronological_rank"] = int(assignment.chronological_rank)
            audits.append(audit)
        return pd.concat(audits, ignore_index=True).sort_values(
            ["chronological_rank", "block_index"]
        ).reset_index(drop=True)

    def full_session_block_audit(self) -> pd.DataFrame:
        """Return a block inventory showing that rolling uses every block."""

        audit = self.block_audit.copy()
        audit["experiment_partition"] = "full_session"
        audit["included_in_rolling_stage"] = 1
        return audit


def build_sliding_windows(
    assignments: pd.DataFrame,
    *,
    training_size: int = 9,
    validation_size: int = 3,
    test_size: int = 1,
    step_size: int = 1,
) -> tuple[SlidingWindow, ...]:
    """Create deterministic chronological rolling assignments."""

    if min(training_size, validation_size, test_size, step_size) <= 0:
        raise ValueError("Window sizes and step size must be positive")
    if test_size != 1:
        raise ValueError("This experiment requires one following test session")
    required = {"session_id", "chronological_rank"}
    if missing := sorted(required.difference(assignments.columns)):
        raise KeyError(f"Assignments are missing columns: {missing}")
    ordered = assignments.sort_values("chronological_rank", kind="mergesort").reset_index(drop=True)
    ranks = ordered["chronological_rank"].astype(int).to_numpy()
    if not np.array_equal(ranks, np.arange(1, len(ordered) + 1)):
        raise ValueError("Chronological ranks must be consecutive and one-based")
    total = training_size + validation_size + test_size
    if len(ordered) < total:
        raise ValueError(f"At least {total} sessions are required")
    windows: list[SlidingWindow] = []
    for offset in range(0, len(ordered) - total + 1, step_size):
        train = ordered.iloc[offset : offset + training_size]
        validation = ordered.iloc[
            offset + training_size : offset + training_size + validation_size
        ]
        test = ordered.iloc[offset + training_size + validation_size]
        windows.append(
            SlidingWindow(
                window_id=len(windows) + 1,
                training_session_ids=tuple(train["session_id"].astype(str)),
                validation_session_ids=tuple(validation["session_id"].astype(str)),
                test_session_id=str(test["session_id"]),
                training_ranks=tuple(train["chronological_rank"].astype(int)),
                validation_ranks=tuple(validation["chronological_rank"].astype(int)),
                test_rank=int(test["chronological_rank"]),
            )
        )
    return tuple(windows)


def partition_session_blocks(frame: pd.DataFrame) -> tuple[pd.DataFrame, BlockPartition, pd.DataFrame]:
    """Hold out the last two observed-complete blocks and censor the terminal block.

    A block is observed complete only when a following block begins.  Therefore
    the last contiguous block is right-censored, the preceding two blocks are
    held out, and all earlier blocks form the fitting prefix.
    """

    required = {"session_id", "trial_index", "block_index", "true_block", "choice_long"}
    if missing := sorted(required.difference(frame.columns)):
        raise KeyError(f"Canonical session is missing columns: {missing}")
    if frame.empty or frame["session_id"].nunique() != 1:
        raise ValueError("Block partitioning requires exactly one nonempty session")
    work = frame.sort_values("trial_index").reset_index(drop=True).copy()
    if work["block_index"].isna().any():
        raise ValueError("Every task row requires a block_index")
    block_ids = tuple(dict.fromkeys(work["block_index"].astype(int).tolist()))
    if len(block_ids) < 4:
        raise ValueError(
            f"Session {work['session_id'].iloc[0]} needs a prefix, two complete "
            "held-out blocks, and one terminal censored block"
        )
    if block_ids != tuple(range(block_ids[0], block_ids[0] + len(block_ids))):
        raise ValueError("Block indices must be chronological and consecutive")
    heldout = (block_ids[-3], block_ids[-2])
    terminal = block_ids[-1]
    prefix = block_ids[:-3]
    partition_by_block = {
        **{value: "prefix" for value in prefix},
        heldout[0]: "heldout",
        heldout[1]: "heldout",
        terminal: "ignored",
    }
    work["experiment_partition"] = work["block_index"].astype(int).map(partition_by_block)
    heldout_rows = work.loc[work["experiment_partition"].eq("heldout")]
    partition = BlockPartition(
        session_id=str(work["session_id"].iloc[0]),
        prefix_block_indices=prefix,
        heldout_block_indices=heldout,
        ignored_terminal_block_index=terminal,
        heldout_start_trial=int(heldout_rows["trial_index"].min()),
        heldout_stop_trial=int(heldout_rows["trial_index"].max()),
    )
    audit_rows: list[dict[str, object]] = []
    for block_index, block in work.groupby("block_index", sort=False, observed=False):
        choices = pd.to_numeric(block["choice_long"], errors="coerce")
        audit_rows.append(
            {
                "session_id": partition.session_id,
                "block_index": int(block_index),
                "true_block": str(block["true_block"].iloc[0]),
                "experiment_partition": str(block["experiment_partition"].iloc[0]),
                "observed_complete": int(int(block_index) != terminal),
                "n_timeline_trials": int(len(block)),
                "n_valid_choices": int(choices.isin([0.0, 1.0]).sum()),
                "first_trial_index": int(block["trial_index"].min()),
                "last_trial_index": int(block["trial_index"].max()),
            }
        )
    return work, partition, pd.DataFrame(audit_rows)


def partition_final_two_blocks(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, BlockPartition, pd.DataFrame]:
    """Hold out the literal final two chronological blocks for fine-tuning.

    This is the opt-in matched-session workflow. Unlike the legacy partition,
    the terminal block is included in the held-out test portion by explicit
    design, and every earlier block belongs to the fine-tuning prefix.
    """

    required = {
        "session_id",
        "trial_index",
        "block_index",
        "true_block",
        "choice_long",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise KeyError(f"Canonical session is missing columns: {missing}")
    if frame.empty or frame["session_id"].nunique() != 1:
        raise ValueError("Block partitioning requires exactly one nonempty session")
    work = frame.sort_values("trial_index").reset_index(drop=True).copy()
    if work["block_index"].isna().any():
        raise ValueError("Every task row requires a block_index")
    block_ids = tuple(dict.fromkeys(work["block_index"].astype(int).tolist()))
    if len(block_ids) < 3:
        raise ValueError(
            f"Session {work['session_id'].iloc[0]} needs a fine-tuning prefix "
            "and two held-out final blocks"
        )
    if block_ids != tuple(range(block_ids[0], block_ids[0] + len(block_ids))):
        raise ValueError("Block indices must be chronological and consecutive")
    heldout = (block_ids[-2], block_ids[-1])
    prefix = block_ids[:-2]
    partition_by_block = {
        **{value: "prefix" for value in prefix},
        heldout[0]: "heldout",
        heldout[1]: "heldout",
    }
    work["experiment_partition"] = work["block_index"].astype(int).map(
        partition_by_block
    )
    heldout_rows = work.loc[work["experiment_partition"].eq("heldout")]
    partition = BlockPartition(
        session_id=str(work["session_id"].iloc[0]),
        prefix_block_indices=prefix,
        heldout_block_indices=heldout,
        ignored_terminal_block_index=None,
        heldout_start_trial=int(heldout_rows["trial_index"].min()),
        heldout_stop_trial=int(heldout_rows["trial_index"].max()),
    )
    audit_rows: list[dict[str, object]] = []
    for block_index, block in work.groupby(
        "block_index", sort=False, observed=False
    ):
        choices = pd.to_numeric(block["choice_long"], errors="coerce")
        audit_rows.append(
            {
                "session_id": partition.session_id,
                "block_index": int(block_index),
                "true_block": str(block["true_block"].iloc[0]),
                "experiment_partition": str(
                    block["experiment_partition"].iloc[0]
                ),
                "observed_complete": int(int(block_index) != block_ids[-1]),
                "right_censored_terminal": int(int(block_index) == block_ids[-1]),
                "n_timeline_trials": int(len(block)),
                "n_valid_choices": int(choices.isin([0.0, 1.0]).sum()),
                "first_trial_index": int(block["trial_index"].min()),
                "last_trial_index": int(block["trial_index"].max()),
            }
        )
    return work, partition, pd.DataFrame(audit_rows)


def _discover_paths(raw_data_root: str | Path, assignments: pd.DataFrame) -> tuple[Path, ...]:
    root = Path(raw_data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw data root does not exist: {root}")
    ordered = assignments.sort_values("chronological_rank")
    requested = tuple(ordered["session_id"].astype(str))
    matches: dict[str, list[Path]] = {session_id: [] for session_id in requested}
    for path in root.rglob("*.mat"):
        if path.stem in matches:
            matches[path.stem].append(path.resolve())
    missing = [session_id for session_id, paths in matches.items() if not paths]
    duplicate = [session_id for session_id, paths in matches.items() if len(paths) > 1]
    if missing:
        raise FileNotFoundError(f"Missing registered raw session: {missing[0]}")
    if duplicate:
        raise ValueError(f"Duplicate registered raw session: {duplicate[0]}")
    return tuple(matches[session_id][0] for session_id in requested)


def prepare_sliding_data(raw_data_root: str | Path, registry) -> PreparedSlidingData:
    """Load every included chronological session for a new sliding experiment."""

    assignments = pd.read_csv(registry.split.assignment_path).sort_values(
        "chronological_rank", kind="mergesort"
    ).reset_index(drop=True)
    if assignments["mouse_id"].astype(str).nunique() != 1:
        raise ValueError("A sliding run requires one mouse registry")
    paths = _discover_paths(raw_data_root, assignments)
    prepared = prepare_dataset(paths, require_legacy_isi_bins=True)
    excluded = prepared.sessions.loc[
        ~prepared.sessions["included_in_model_trials"].astype(bool), "session_id"
    ].astype(str).tolist()
    if excluded:
        raise ValueError(f"Registered included session failed legacy filtering: {excluded[0]}")
    expected = tuple(assignments["session_id"].astype(str))
    observed = tuple(prepared.sessions["session_id"].astype(str))
    if observed != expected:
        raise ValueError("Prepared session order disagrees with chronological assignments")
    partitioned: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    partitions: dict[str, BlockPartition] = {}
    for session_id in expected:
        frame = prepared.canonical_trials.loc[
            prepared.canonical_trials["session_id"].astype(str).eq(session_id)
        ]
        selected, partition, audit = partition_session_blocks(frame)
        partitioned.append(selected)
        audits.append(audit)
        partitions[session_id] = partition
    canonical = pd.concat(partitioned, ignore_index=True).merge(
        assignments[["session_id", "chronological_rank"]],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    return PreparedSlidingData(
        mouse_id=str(assignments["mouse_id"].iloc[0]),
        assignments=assignments,
        canonical_trials=canonical.sort_values(
            ["chronological_rank", "trial_index"]
        ).reset_index(drop=True),
        partitions=partitions,
        block_audit=pd.concat(audits, ignore_index=True).merge(
            assignments[["session_id", "chronological_rank"]],
            on="session_id",
            how="left",
            validate="many_to_one",
        ).sort_values(["chronological_rank", "block_index"]).reset_index(drop=True),
        source_files=tuple(prepared.sessions["source_file"].astype(str)),
        source_sha256=tuple(prepared.sessions["source_sha256"].astype(str)),
    )


def prepare_full_session_matched_data(
    raw_data_root: str | Path, registry
) -> PreparedSlidingData:
    """Load chronological sessions without invoking the legacy block partition.

    Rolling-window callers use each session's full timeline.  The stored
    partition is the literal-final-two split needed later for matched-session
    fine-tuning, so a valid three-block session is accepted and the terminal
    block remains part of the held-out test portion.
    """

    assignments = pd.read_csv(registry.split.assignment_path).sort_values(
        "chronological_rank", kind="mergesort"
    ).reset_index(drop=True)
    if assignments["mouse_id"].astype(str).nunique() != 1:
        raise ValueError("A sliding run requires one mouse registry")
    paths = _discover_paths(raw_data_root, assignments)
    prepared = prepare_dataset(paths, require_legacy_isi_bins=True)
    excluded = prepared.sessions.loc[
        ~prepared.sessions["included_in_model_trials"].astype(bool), "session_id"
    ].astype(str).tolist()
    if excluded:
        raise ValueError(
            f"Registered included session failed model-trial filtering: {excluded[0]}"
        )
    expected = tuple(assignments["session_id"].astype(str))
    observed = tuple(prepared.sessions["session_id"].astype(str))
    if observed != expected:
        raise ValueError("Prepared session order disagrees with chronological assignments")

    partitioned: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    partitions: dict[str, BlockPartition] = {}
    for session_id in expected:
        frame = prepared.canonical_trials.loc[
            prepared.canonical_trials["session_id"].astype(str).eq(session_id)
        ]
        selected, partition, audit = partition_final_two_blocks(frame)
        partitioned.append(selected)
        audits.append(audit)
        partitions[session_id] = partition
    canonical = pd.concat(partitioned, ignore_index=True).merge(
        assignments[["session_id", "chronological_rank"]],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    return PreparedSlidingData(
        mouse_id=str(assignments["mouse_id"].iloc[0]),
        assignments=assignments,
        canonical_trials=canonical.sort_values(
            ["chronological_rank", "trial_index"]
        ).reset_index(drop=True),
        partitions=partitions,
        block_audit=pd.concat(audits, ignore_index=True).merge(
            assignments[["session_id", "chronological_rank"]],
            on="session_id",
            how="left",
            validate="many_to_one",
        ).sort_values(["chronological_rank", "block_index"]).reset_index(drop=True),
        source_files=tuple(prepared.sessions["source_file"].astype(str)),
        source_sha256=tuple(prepared.sessions["source_sha256"].astype(str)),
    )


def aggregate_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
) -> pd.DataFrame:
    """Average constrained winners for training-member sessions.

    Sessions never appearing in a training window (the final four in a 9/3/1
    design) use the final chronological window winner as an explicit fallback.
    """

    if not windows:
        raise ValueError("At least one completed window is required")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(winner_parameters[final_window.window_id], dtype=float)
    if final_values.shape != (5,) or not np.all(np.isfinite(final_values)):
        raise ValueError("Window winners must contain five finite parameters")
    rows: list[dict[str, object]] = []
    for assignment in assignments.sort_values("chronological_rank").itertuples(index=False):
        session_id = str(assignment.session_id)
        contributors = [
            window.window_id for window in windows
            if session_id in window.training_session_ids
        ]
        fallback = not contributors
        used = contributors or [final_window.window_id]
        matrix = np.asarray([winner_parameters[item] for item in used], dtype=float)
        if matrix.shape != (len(used), 5) or np.any(matrix[:, 4] <= 0.0):
            raise ValueError("Every M0 winner requires five values and positive kappa")
        mean = matrix.mean(axis=0)
        mean[4] = np.exp(np.log(matrix[:, 4]).mean())
        std = matrix.std(axis=0, ddof=0)
        row: dict[str, object] = {
            "mouse_id": str(assignment.mouse_id),
            "session_id": session_id,
            "chronological_rank": int(assignment.chronological_rank),
            "contributing_window_ids": ";".join(str(item) for item in used),
            "n_contributing_windows": len(used),
            "initialization_policy": (
                "final_window_winner_fallback_no_training_membership"
                if fallback else "unweighted_mean_of_training_member_window_winners"
            ),
        }
        for index, name in enumerate(PARAMETER_NAMES):
            row[f"initial_{name}"] = float(mean[index])
            row[f"window_sd_{name}"] = float(std[index])
            row[f"window_min_{name}"] = float(matrix[:, index].min())
            row[f"window_max_{name}"] = float(matrix[:, index].max())
        rows.append(row)
    return pd.DataFrame(rows)


def match_window_winners_to_test_sessions(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
    parameter_names: Sequence[str],
) -> pd.DataFrame:
    """Map each rolling winner directly to its one following test session."""

    if not windows:
        raise ValueError("At least one completed window is required")
    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must be nonempty and unique")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every window requires one winning parameter vector")
    assignment_rows = assignments.set_index(
        assignments["session_id"].astype(str), drop=False
    )
    rows: list[dict[str, object]] = []
    for window in windows:
        values = np.asarray(winner_parameters[window.window_id], dtype=float)
        if values.shape != (len(parameter_names),) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Every matched window winner must contain one finite value per parameter"
            )
        if "kappa" in parameter_names and values[list(parameter_names).index("kappa")] <= 0:
            raise ValueError("Every matched window winner requires positive kappa")
        session_id = window.test_session_id
        if session_id not in assignment_rows.index:
            raise ValueError(f"Unknown rolling test session: {session_id}")
        assignment = assignment_rows.loc[session_id]
        row: dict[str, object] = {
            "mouse_id": str(assignment["mouse_id"]),
            "session_id": session_id,
            "chronological_rank": int(assignment["chronological_rank"]),
            "source_window_id": window.window_id,
            "source_window_training_ranks": ";".join(
                str(value) for value in window.training_ranks
            ),
            "source_window_validation_ranks": ";".join(
                str(value) for value in window.validation_ranks
            ),
            "source_window_test_rank": window.test_rank,
            "n_contributing_windows": 1,
            "initialization_policy": "direct_matched_rolling_test_window_winner",
        }
        for name, value in zip(parameter_names, values, strict=True):
            row[f"initial_{name}"] = float(value)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("chronological_rank").reset_index(drop=True)
    expected_ranks = [window.test_rank for window in windows]
    if result["chronological_rank"].tolist() != expected_ranks:
        raise ValueError("Matched fine-tuning sessions disagree with rolling test ranks")
    return result


def prediction_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize predictions for train/validation/test without fitting access."""

    valid = predictions.loc[predictions["model_evaluated"].astype(bool)].copy()
    if valid.empty:
        raise ValueError("Predictions contain no evaluated choices")
    valid["squared_error"] = np.square(
        valid["model_probability_long"].astype(float)
        - pd.to_numeric(valid["choice_long"], errors="coerce")
    )
    groups = valid.groupby(["split_role", "session_id"], observed=False).agg(
        valid_trial_count=("model_evaluated", "size"),
        elpd=("per_trial_negative_log_likelihood", lambda values: -float(values.sum())),
        mean_nll=("per_trial_negative_log_likelihood", "mean"),
        brier_score=("squared_error", "mean"),
        observed_long_rate=("choice_long", "mean"),
        predicted_long_rate=("model_probability_long", "mean"),
        mouse_accuracy=("mouse_correct", "mean"),
        model_expected_accuracy=("model_probability_correct", "mean"),
    ).reset_index()
    pooled = valid.groupby("split_role", observed=False).agg(
        valid_trial_count=("model_evaluated", "size"),
        elpd=("per_trial_negative_log_likelihood", lambda values: -float(values.sum())),
        mean_nll=("per_trial_negative_log_likelihood", "mean"),
        brier_score=("squared_error", "mean"),
        observed_long_rate=("choice_long", "mean"),
        predicted_long_rate=("model_probability_long", "mean"),
        mouse_accuracy=("mouse_correct", "mean"),
        model_expected_accuracy=("model_probability_correct", "mean"),
    ).reset_index()
    pooled["session_id"] = "ALL_SESSIONS"
    return pd.concat([pooled, groups], ignore_index=True)
