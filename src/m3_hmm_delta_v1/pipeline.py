"""Leak-safe, additive data assembly for M3 fitting and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m0_sensory_v1.data import prepare_dataset
from m0_sensory_v1.pipeline import FitInputAudit, discover_registered_session_paths

from .registry import M3DeltaRegistry
from .session import M3SessionTrials, m3_session_trials_from_frame


@dataclass(frozen=True)
class M3PreparedFitData:
    """M3 tensors, canonical timelines, diagnostic labels, and provenance."""

    training_sessions: tuple[M3SessionTrials, ...]
    validation_sessions: tuple[M3SessionTrials, ...]
    validation_block_labels: dict[str, tuple[object, ...]]
    canonical_trials: pd.DataFrame
    audit: FitInputAudit


def prepare_registered_m3_fit_data(
    raw_data_root: str | Path,
    registry: M3DeltaRegistry,
) -> M3PreparedFitData:
    """Prepare complete causal timelines without opening locked-test files."""

    if registry.model_id != "M3_hmm_delta_v1":
        raise ValueError("prepare_registered_m3_fit_data requires an M3 registry")
    source_paths = discover_registered_session_paths(raw_data_root, registry)
    prepared = prepare_dataset(source_paths, require_legacy_isi_bins=True)
    assignments = pd.read_csv(registry.split.assignment_path)
    selected = assignments.loc[
        assignments["split_role"].isin(["train", "validation"])
    ].sort_values(["mouse_id", "chronological_rank"])
    expected_ids = tuple(selected["session_id"].astype(str))
    observed_ids = tuple(prepared.sessions["session_id"].astype(str))
    if set(observed_ids) != set(expected_ids):
        raise ValueError("Prepared sessions disagree with frozen M3 split IDs")
    excluded = prepared.sessions.loc[
        ~prepared.sessions["included_in_model_trials"].astype(bool), "session_id"
    ].astype(str)
    if not excluded.empty:
        raise ValueError(f"Frozen M3 session failed inclusion rule: {excluded.iloc[0]}")

    by_id = {
        str(session_id): frame.sort_values("trial_index").reset_index(drop=True)
        for session_id, frame in prepared.canonical_trials.groupby(
            "session_id", sort=False, observed=False
        )
    }
    training: list[M3SessionTrials] = []
    validation: list[M3SessionTrials] = []
    validation_blocks: dict[str, tuple[object, ...]] = {}
    train_rows = validation_rows = train_valid = validation_valid = 0
    for assignment in selected.itertuples(index=False):
        session_id = str(assignment.session_id)
        frame = by_id[session_id]
        session = m3_session_trials_from_frame(
            frame, split_role=str(assignment.split_role)
        )
        valid = (
            session.choice_long.isfinite()
            & ((session.choice_long == 0.0) | (session.choice_long == 1.0))
            & session.duration_s.isfinite()
            & (session.duration_s > 0.0)
        )
        if assignment.split_role == "train":
            training.append(session)
            train_rows += len(frame)
            train_valid += int(valid.sum())
        else:
            validation.append(session)
            validation_rows += len(frame)
            validation_valid += int(valid.sum())
            validation_blocks[session_id] = tuple(frame["true_block"].tolist())

    mouse_ids = set(selected["mouse_id"].astype(str))
    if len(mouse_ids) != 1:
        raise ValueError("An M3 fit operates on exactly one mouse")
    canonical = prepared.canonical_trials.merge(
        selected[["session_id", "split_role"]],
        on="session_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["session_index", "trial_index"]).reset_index(drop=True)
    audit = FitInputAudit(
        mouse_id=next(iter(mouse_ids)),
        split_version=registry.split.version,
        training_session_ids=tuple(item.session_id for item in training),
        validation_session_ids=tuple(item.session_id for item in validation),
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
        reward_naive_rows=int(canonical["outcome"].eq("RewardNaive").sum()),
        punish_naive_rows=int(canonical["outcome"].eq("PunishNaive").sum()),
        locked_test_session_count=int(assignments["split_role"].eq("test").sum()),
        locked_test_files_opened=0,
    )
    return M3PreparedFitData(
        training_sessions=tuple(training),
        validation_sessions=tuple(validation),
        validation_block_labels=validation_blocks,
        canonical_trials=canonical,
        audit=audit,
    )
