"""Leak-safe raw-data assembly for the independently runnable M0 fitter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import prepare_dataset
from .model.session import SessionTrials, session_trials_from_frame
from .registry import ModelRegistry


@dataclass(frozen=True)
class FitInputAudit:
    """Serializable provenance for the train/validation data opened by a run."""

    mouse_id: str
    split_version: str
    training_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    opened_source_files: tuple[str, ...]
    opened_source_sha256: tuple[str, ...]
    training_timeline_rows: int
    validation_timeline_rows: int
    training_valid_decisions: int
    validation_valid_decisions: int
    omitted_or_invalid_rows_retained: int
    reward_naive_rows: int
    punish_naive_rows: int
    locked_test_session_count: int
    locked_test_files_opened: int

    def as_dict(self) -> dict:
        """Return JSON-safe audit primitives."""

        return asdict(self)


@dataclass(frozen=True)
class PreparedFitData:
    """Train/validation tensors, canonical timelines, and input audit."""

    training_sessions: tuple[SessionTrials, ...]
    validation_sessions: tuple[SessionTrials, ...]
    validation_block_labels: dict[str, tuple[object, ...]]
    canonical_trials: pd.DataFrame
    audit: FitInputAudit


def discover_registered_session_paths(
    raw_data_root: str | Path,
    registry: ModelRegistry,
    *,
    roles: Iterable[str] = ("train", "validation"),
) -> tuple[Path, ...]:
    r"""Resolve frozen session IDs without opening locked-test MATLAB contents.

    Purpose
    -------
    Match the frozen split table to raw files before behavioral trial loading.

    Mathematical definition
    -----------------------
    For requested roles :math:`\mathcal R`, return the unique path satisfying

    .. math::

       \operatorname{stem}(path)=session\_id
       \quad\forall\ session\text{ with }split\_role\in\mathcal R.

    Parameter interpretation
    ------------------------
    Roles are immutable split metadata. They are unrelated to behavioral task
    block labels.

    Input units and shapes
    ----------------------
    ``raw_data_root`` is a directory tree. The registry split has one row per
    session.

    Returned quantities and shapes
    ------------------------------
    Returns one ordered path per requested frozen session.

    Constraints
    -----------
    Only ``train`` and ``validation`` are accepted. Every session stem must
    resolve exactly once; duplicate matches raise.

    Numerical stability
    -------------------
    Matching is exact string equality and does not depend on timestamps parsed
    from floating or locale-sensitive values.

    Causal restrictions
    -------------------
    Directory entries may be indexed, but only selected train/validation paths
    are returned and subsequently opened. Test MATLAB contents are never read.

    Important edge cases
    --------------------
    Missing roots, missing sessions, duplicate filenames, and test requests all
    fail before fitting begins.

    Related tests
    -------------
    ``test_discovery_returns_only_registered_non_test_paths``.
    """

    selected_roles = tuple(roles)
    if not selected_roles or set(selected_roles).difference({"train", "validation"}):
        raise ValueError("Raw fitting may discover only train/validation roles")
    root = Path(raw_data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw data root does not exist: {root}")
    assignments = pd.read_csv(registry.split.assignment_path)
    selected = assignments.loc[
        assignments["split_role"].isin(selected_roles)
    ].sort_values(["mouse_id", "chronological_rank"])
    requested_ids = set(selected["session_id"].astype(str))
    matches: dict[str, list[Path]] = {session_id: [] for session_id in requested_ids}
    for path in root.rglob("*.mat"):
        if path.stem in matches:
            matches[path.stem].append(path.resolve())
    missing = sorted(session_id for session_id, paths in matches.items() if not paths)
    duplicated = sorted(session_id for session_id, paths in matches.items() if len(paths) > 1)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} registered session files; first: {missing[0]}"
        )
    if duplicated:
        raise ValueError(
            f"Duplicate raw paths for registered session; first: {duplicated[0]}"
        )
    return tuple(matches[str(session_id)][0] for session_id in selected["session_id"])


def prepare_registered_fit_data(
    raw_data_root: str | Path,
    registry: ModelRegistry,
) -> PreparedFitData:
    r"""Load canonical train/validation timelines and never open locked test.

    Purpose
    -------
    Build the exact session tensors consumed by fitting directly from selected
    raw MATLAB files, while retaining omissions and original naive labels.

    Mathematical definition
    -----------------------
    Each session retains the full timeline

    .. math::

       \{(s_{d,t},y_{d,t})\}_{t=1}^{n_d},

    where non-decisions have :math:`y_{d,t}=\mathrm{NaN}` and are masked only
    inside the model/likelihood.

    Parameter interpretation
    ------------------------
    ``RewardNaive`` and ``PunishNaive`` reconstruct choices exactly like reward
    and punishment while their original labels remain in the input audit data.

    Input units and shapes
    ----------------------
    Raw files contain session MATLAB structs. Output tensors are one-dimensional
    ``float64`` durations in seconds and binary/NaN choices.

    Returned quantities and shapes
    ------------------------------
    Returns ordered train and validation session tuples, validation block labels
    for diagnostics, and a complete serializable provenance audit.

    Constraints
    -----------
    Every selected session must be in the frozen split and pass the legacy
    five-short/five-long session inclusion rule. Test paths are not selected.

    Numerical stability
    -------------------
    Canonical preparation normalizes nullable values before Torch conversion;
    model calculations later index out nonfinite durations and choices.

    Causal restrictions
    -------------------
    True block is returned only for post-prediction validation diagnostics. It
    never enters :class:`SessionTrials` or any probability calculation.

    Important edge cases
    --------------------
    Any missing, duplicate, excluded, or unexpected session raises rather than
    silently altering the frozen sample.

    Related tests
    -------------
    ``test_real_raw_pipeline_smoke_preserves_timeline_and_locked_test``.
    """

    source_paths = discover_registered_session_paths(raw_data_root, registry)
    prepared = prepare_dataset(source_paths, require_legacy_isi_bins=True)
    assignments = pd.read_csv(registry.split.assignment_path)
    selected_assignments = assignments.loc[
        assignments["split_role"].isin(["train", "validation"])
    ].sort_values(["mouse_id", "chronological_rank"])
    expected_ids = tuple(selected_assignments["session_id"].astype(str))
    observed_ids = tuple(prepared.sessions["session_id"].astype(str))
    if set(observed_ids) != set(expected_ids):
        raise ValueError("Prepared sessions disagree with frozen train/validation IDs")
    excluded = prepared.sessions.loc[
        ~prepared.sessions["included_in_model_trials"].astype(bool),
        "session_id",
    ].astype(str).tolist()
    if excluded:
        raise ValueError(f"Frozen session failed legacy inclusion rule: {excluded[0]}")

    by_id = {
        str(session_id): frame.sort_values("trial_index").reset_index(drop=True)
        for session_id, frame in prepared.canonical_trials.groupby(
            "session_id",
            sort=False,
            observed=False,
        )
    }
    training: list[SessionTrials] = []
    validation: list[SessionTrials] = []
    validation_blocks: dict[str, tuple[object, ...]] = {}
    train_rows = validation_rows = train_valid = validation_valid = 0
    for assignment in selected_assignments.itertuples(index=False):
        session_id = str(assignment.session_id)
        frame = by_id[session_id]
        if registry.model_id in {"M1_hmm_true_v1", "M2_hmm_static_v1"}:
            if registry.model_id == "M1_hmm_true_v1":
                from .model.m1_session import m1_session_trials_from_frame as dynamic_session_from_frame
            else:
                from .model.m2_session import m2_session_trials_from_frame as dynamic_session_from_frame

            session = dynamic_session_from_frame(
                frame, split_role=assignment.split_role
            )
        else:
            session = session_trials_from_frame(frame, split_role=assignment.split_role)
        valid_count = int(
            (
                session.choice_long.isfinite()
                & ((session.choice_long == 0.0) | (session.choice_long == 1.0))
                & session.duration_s.isfinite()
                & (session.duration_s > 0.0)
            ).sum()
        )
        if assignment.split_role == "train":
            training.append(session)
            train_rows += len(frame)
            train_valid += valid_count
        else:
            validation.append(session)
            validation_rows += len(frame)
            validation_valid += valid_count
            validation_blocks[session_id] = tuple(frame["true_block"].tolist())

    mouse_ids = set(selected_assignments["mouse_id"].astype(str))
    if len(mouse_ids) != 1:
        raise ValueError("Current M0 runner requires exactly one mouse per run")
    audit = FitInputAudit(
        mouse_id=next(iter(mouse_ids)),
        split_version=registry.split.version,
        training_session_ids=tuple(session.session_id for session in training),
        validation_session_ids=tuple(session.session_id for session in validation),
        opened_source_files=tuple(prepared.sessions["source_file"].astype(str)),
        opened_source_sha256=tuple(prepared.sessions["source_sha256"].astype(str)),
        training_timeline_rows=train_rows,
        validation_timeline_rows=validation_rows,
        training_valid_decisions=train_valid,
        validation_valid_decisions=validation_valid,
        omitted_or_invalid_rows_retained=int(
            len(prepared.canonical_trials)
            - prepared.canonical_trials["choice_valid"].fillna(0).astype(int).sum()
        ),
        reward_naive_rows=int(prepared.canonical_trials["outcome"].eq("RewardNaive").sum()),
        punish_naive_rows=int(prepared.canonical_trials["outcome"].eq("PunishNaive").sum()),
        locked_test_session_count=int(assignments["split_role"].eq("test").sum()),
        locked_test_files_opened=0,
    )
    return PreparedFitData(
        training_sessions=tuple(training),
        validation_sessions=tuple(validation),
        validation_block_labels=validation_blocks,
        canonical_trials=prepared.canonical_trials.merge(
            selected_assignments[["session_id", "split_role"]],
            on="session_id",
            how="left",
            validate="many_to_one",
        ).sort_values(["session_index", "trial_index"]).reset_index(drop=True),
        audit=audit,
    )


def prepare_registered_m1_fit_data(
    raw_data_root: str | Path,
    registry,
) -> PreparedFitData:
    """Prepare M1 timelines with feedback/category fields and locked test data.

    This explicit entry point prevents accidentally passing an M0 registry to
    an M1 run while sharing the same audited raw-data discovery and canonical
    preparation path.
    """

    if registry.model_id != "M1_hmm_true_v1":
        raise ValueError("prepare_registered_m1_fit_data requires an M1 registry")
    return prepare_registered_fit_data(raw_data_root, registry)


def prepare_registered_m2_fit_data(
    raw_data_root: str | Path,
    registry,
) -> PreparedFitData:
    """Prepare full M2 timelines while preserving the locked-test boundary."""

    if registry.model_id != "M2_hmm_static_v1":
        raise ValueError("prepare_registered_m2_fit_data requires an M2 registry")
    return prepare_registered_fit_data(raw_data_root, registry)


def prepare_locked_test_analysis_trials(
    raw_data_root: str | Path,
    registry: ModelRegistry,
    *,
    unlock_test: bool = False,
) -> pd.DataFrame:
    """Load the frozen test timelines only after an explicit unlock request.

    This function is intentionally separate from fitting. Merely constructing
    the normal train/validation pipeline cannot open a test MATLAB file.
    """

    if not unlock_test:
        raise PermissionError(
            "Test analysis is locked; pass unlock_test=True only after the model "
            "and analysis specification have been frozen"
        )
    root = Path(raw_data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw data root does not exist: {root}")
    assignments = pd.read_csv(registry.split.assignment_path)
    selected = assignments.loc[assignments["split_role"].eq("test")].sort_values(
        ["mouse_id", "chronological_rank"]
    )
    requested_ids = set(selected["session_id"].astype(str))
    matches: dict[str, list[Path]] = {session_id: [] for session_id in requested_ids}
    for path in root.rglob("*.mat"):
        if path.stem in matches:
            matches[path.stem].append(path.resolve())
    missing = sorted(key for key, paths in matches.items() if not paths)
    duplicated = sorted(key for key, paths in matches.items() if len(paths) > 1)
    if missing:
        raise FileNotFoundError(f"Missing locked-test session file: {missing[0]}")
    if duplicated:
        raise ValueError(f"Duplicate locked-test session path: {duplicated[0]}")
    paths = tuple(matches[str(row.session_id)][0] for row in selected.itertuples())
    prepared = prepare_dataset(paths, require_legacy_isi_bins=True)
    if set(prepared.sessions["session_id"].astype(str)) != requested_ids:
        raise ValueError("Prepared test sessions disagree with the frozen split")
    excluded = prepared.sessions.loc[
        ~prepared.sessions["included_in_model_trials"].astype(bool), "session_id"
    ]
    if not excluded.empty:
        raise ValueError(f"Frozen test session failed inclusion rule: {excluded.iloc[0]}")
    return prepared.canonical_trials.assign(split_role="test").sort_values(
        ["session_index", "trial_index"]
    ).reset_index(drop=True)


def save_fit_input_audit(audit: FitInputAudit, path: str | Path) -> None:
    """Atomically save the leak-safe fitting-input audit as JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(audit.as_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
