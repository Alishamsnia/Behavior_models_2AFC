"""Canonical trial preparation for ``M0_sensory_v1``.

The extraction rules intentionally mirror ``Model_v24_validation.ipynb``:

* trial outcome is selected using the same state priority;
* reverse contingencies swap trial and block codes ``1 <-> 2``;
* duration is read from ``ProcessedSessionData.trial_isi.PostMeanISI``;
* rewarded/punished outcomes reconstruct the binary response;
* the legacy analysis view retains only Reward/Punish and naive variants; and
* the optional session filter requires exactly five rounded short and five
  rounded long ISI levels.

Unlike the legacy notebook, the canonical table retains invalid-choice and
omission trials for stable indexing and audit. ``predict_m0_trials`` boolean-
indexes these rows out before any model or gradient computation.
``select_legacy_model_trials`` remains available only as a compact,
legacy-compatible comparison view.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .matlab import load_session_mat


OUTCOME_PRIORITY: tuple[str, ...] = (
    "ChangingMindReward",
    "WrongInitiation",
    "Punish",
    "Reward",
    "PunishNaive",
    "RewardNaive",
    "EarlyChoice",
    "DidNotChoose",
)

VALID_CHOICE_OUTCOMES = frozenset(
    {"Reward", "Punish", "RewardNaive", "PunishNaive"}
)
REWARDED_OUTCOMES = frozenset({"Reward", "RewardNaive"})
PUNISHED_OUTCOMES = frozenset({"Punish", "PunishNaive"})

CANONICAL_TRIAL_COLUMNS: tuple[str, ...] = (
    "mouse_id",
    "session_id",
    "session_index",
    "session_date",
    "trial_index",
    "source_trial_index",
    "block_index",
    "true_block",
    "true_run_length",
    "category_long",
    "duration_s",
    "choice_long",
    "rewarded",
    "omission",
    "choice_valid",
    "feedback_valid",
    "outcome",
    "outcome_time_s",
    "warm_up",
    "jitter_flag",
    "opto_tag",
    "task_version",
    "contingency_index",
    "source_file",
    "source_sha256",
)


@dataclass(frozen=True)
class SessionFileMetadata:
    """Stable identifiers parsed from a behavioral filename.

    Attributes
    ----------
    mouse_id:
        Mouse identifier at the beginning of the filename, for example
        ``MC08``. This corrects the legacy regex that captured ``SChR2``.
    session_id:
        Stable identifier equal to the complete filename stem.
    task_version:
        Version token such as ``V_1``.
    session_date:
        Recording timestamp parsed from ``YYYYMMDD_HHMMSS`` when available.
    """

    mouse_id: str
    session_id: str
    task_version: str
    session_date: datetime


@dataclass(frozen=True)
class DataPreparationResult:
    """Prepared canonical data and its legacy-compatible model view.

    Attributes
    ----------
    canonical_trials:
        One row per raw task trial, including omissions and invalid choices.
    model_trials:
        Valid binary-choice rows from sessions that passed the configured ISI
        criterion. This compact view is retained for compatibility checks; the
        omission-safe M0 wrapper can operate directly on canonical rows.
    sessions:
        One audit row per input session, including inclusion decisions, counts,
        file checksum, and observed ISI levels.
    """

    canonical_trials: pd.DataFrame
    model_trials: pd.DataFrame
    sessions: pd.DataFrame


def _as_sequence(value: Any) -> list[Any]:
    """Return a MATLAB-derived scalar or container as a Python list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _indexed(value: Any, index: int, default: Any = None) -> Any:
    """Safely retrieve an indexed MATLAB-derived value.

    Scalars are treated as one-element sequences. This is necessary because
    SciPy's ``squeeze_me=True`` removes singleton dimensions.
    """

    values = _as_sequence(value)
    return values[index] if 0 <= index < len(values) else default


def _first_finite(value: Any) -> float | None:
    """Return the first finite numeric scalar from a MATLAB-derived value."""

    for candidate in _as_sequence(value):
        try:
            scalar = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            return scalar
    return None


def _optional_float(value: Any) -> float | None:
    """Convert a scalar-like value to a finite float or return ``None``."""

    scalar = _first_finite(value)
    return scalar if scalar is not None else None


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a stable SHA-256 checksum without loading the file at once."""

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_behavior_file_path(path: str | Path) -> SessionFileMetadata:
    """Parse stable mouse, session, version, and date identifiers.

    Purpose
    -------
    Provide identifiers needed for per-mouse pooling and chronological session
    splitting while retaining a stable session ID.

    Parameters
    ----------
    path:
        Behavioral filename. The expected form includes ``<mouse>_..._block``,
        a version token such as ``V_1``, and ``YYYYMMDD_HHMMSS``.

    Returns
    -------
    SessionFileMetadata
        Parsed identifiers. ``session_id`` is the full filename stem.

    Constraints
    -----------
    The mouse identifier must be the first underscore-delimited filename token.

    Important edge cases
    --------------------
    A missing time component is accepted and interpreted as midnight. Invalid
    names raise ``ValueError`` rather than silently receiving unstable IDs.

    Related tests
    -------------
    ``test_parse_behavior_file_path`` checks the task's current naming scheme.
    """

    source = Path(path)
    stem = source.stem
    pattern = re.compile(
        r"^(?P<mouse>[A-Za-z0-9]+)_.*?_block.*?"
        r"(?P<version>V_\d+).*?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?$"
    )
    match = pattern.search(stem)
    if match is None:
        raise ValueError(f"Filename does not match the behavioral convention: {source.name}")

    time_text = match.group("time") or "000000"
    session_date = datetime.strptime(match.group("date") + time_text, "%Y%m%d%H%M%S")
    return SessionFileMetadata(
        mouse_id=match.group("mouse"),
        session_id=stem,
        task_version=match.group("version"),
        session_date=session_date,
    )


def states_labeling(trial_states: dict[str, Any]) -> tuple[str, float | None]:
    """Select a trial outcome using the legacy outcome priority.

    Purpose
    -------
    Reproduce the previous notebook's deterministic state-priority rule.

    Mathematical definition
    -----------------------
    The first state in :data:`OUTCOME_PRIORITY` with a finite first timestamp
    defines the outcome. This is an ordered rule, not a probabilistic model.

    Parameters
    ----------
    trial_states:
        Mapping from Bpod state names to scalar or sequence timestamps.

    Returns
    -------
    tuple
        ``(outcome_name, first_timestamp_seconds)``. If no recognized finite
        state exists, the result is ``("Other", None)``.

    Input units and shapes
    ----------------------
    State timestamps are seconds and may be MATLAB scalars or arrays.

    Causal restrictions
    -------------------
    Outcome is post-choice information and must not enter an M0 prediction.

    Important edge cases
    --------------------
    ``Punish`` precedes ``Reward`` exactly as in the previous notebook.

    Related tests
    -------------
    ``test_states_labeling_preserves_priority``.
    """

    for state in OUTCOME_PRIORITY:
        if state not in trial_states:
            continue
        timestamp = _first_finite(trial_states[state])
        if timestamp is not None:
            return state, timestamp
    return "Other", None


def _canonical_code(raw_code: Any, contingency_index: int) -> int | None:
    """Canonicalize task code ``1/2`` under normal or reversed contingency."""

    scalar = _optional_float(raw_code)
    if scalar is None:
        return None
    code = int(scalar)
    if contingency_index == 2 and code in (1, 2):
        return 2 if code == 1 else 1
    return code


def _extract_duration(processed_trial: Any) -> float | None:
    """Read ``trial_isi.PostMeanISI`` in seconds from one processed trial."""

    if not isinstance(processed_trial, dict):
        return None
    trial_isi = processed_trial.get("trial_isi")
    if not isinstance(trial_isi, dict):
        return None
    return _optional_float(trial_isi.get("PostMeanISI"))


def _contiguous_block_metadata(blocks: Sequence[object]) -> tuple[list[int], list[int]]:
    """Compute one-based contiguous block IDs and within-block run lengths."""

    block_indices: list[int] = []
    run_lengths: list[int] = []
    previous: object = object()
    block_index = 0
    run_length = 0

    for current in blocks:
        key = current if pd.notna(current) else "<missing>"
        if key != previous:
            block_index += 1
            run_length = 1
        else:
            run_length += 1
        block_indices.append(block_index)
        run_lengths.append(run_length)
        previous = key
    return block_indices, run_lengths


def prepare_session(
    session_data: dict[str, Any],
    metadata: SessionFileMetadata,
    *,
    source_file: str | Path,
    source_sha256: str = "",
) -> pd.DataFrame:
    r"""Convert one MATLAB session into the canonical one-row-per-trial table.

    Purpose
    -------
    Preserve the exact task timeline while reproducing the previous notebook's
    outcome, contingency, duration, reward, and response reconstruction rules.

    Mathematical definition
    -----------------------
    On valid deterministic-feedback trials, the long-associated choice is

    .. math::

       Y = C \quad\text{if rewarded},\qquad Y = 1-C \quad\text{if punished}.

    Equivalently, this is the inversion of ``C = W Y + (1-W)(1-Y)``.

    Parameters
    ----------
    session_data:
        Converted MATLAB ``SessionData`` dictionary.
    metadata:
        Stable filename-derived identifiers.
    source_file:
        Original raw-file path retained for traceability.
    source_sha256:
        Optional checksum of the raw file.

    Returns
    -------
    pandas.DataFrame
        Canonical columns listed in :data:`CANONICAL_TRIAL_COLUMNS`. Trial,
        block, and run-length indices are one-based. ``source_trial_index`` is
        retained as the zero-based raw index.

    Constraints
    -----------
    Durations are stored in seconds. Binary scientific columns use pandas'
    nullable ``Int64`` dtype so invalid trials remain explicit rather than
    being coerced into a category.

    Numerical stability
    -------------------
    Nonfinite durations and timestamps become missing values and are rejected
    from the model likelihood view.

    Causal restrictions
    -------------------
    ``true_block``, category, reward, and outcome are stored for audit and
    post-choice processing only. M0 prediction may use only ``duration_s`` and
    the pooled parameter vector.

    Important edge cases
    --------------------
    The default contingency is normal (index 1), matching the legacy notebook.
    Missing or unknown trial/block codes remain missing or explicit rather than
    being silently relabeled as long.

    Related tests
    -------------
    Synthetic normal/reverse-contingency and omission tests exercise this
    function; real-session smoke checks confirm compatibility with raw files.
    """

    if not isinstance(session_data, dict):
        raise TypeError("session_data must be a converted SessionData dictionary")

    trial_settings = _as_sequence(session_data.get("TrialSettings"))
    if not trial_settings or not isinstance(trial_settings[0], dict):
        raise KeyError("SessionData.TrialSettings[0] is missing or invalid")
    settings = trial_settings[0].get("GUI")
    if not isinstance(settings, dict):
        raise KeyError("SessionData.TrialSettings[0].GUI is missing or invalid")

    number_of_trials_value = session_data.get("nTrials", 0)
    number_of_trials = int(_optional_float(number_of_trials_value) or 0)
    if number_of_trials < 0:
        raise ValueError(f"nTrials must be nonnegative, got {number_of_trials}")

    contingency_index = int(_optional_float(settings.get("Contingency")) or 1)
    warm_up_trials = int(_optional_float(settings.get("NumNaiveWarmup")) or 0)

    trial_types = session_data.get("TrialTypes", [])
    block_types = session_data.get("BlockTypes", [])
    processed_trials = session_data.get("ProcessedSessionData", [])
    jitter_flags = session_data.get("JitterFlag", [])
    opto_types = session_data.get("OptoType", [])

    raw_events = session_data.get("RawEvents")
    raw_trials = raw_events.get("Trial", []) if isinstance(raw_events, dict) else []

    rows: list[dict[str, Any]] = []
    canonical_blocks: list[object] = []

    for raw_index in range(number_of_trials):
        trial_event = _indexed(raw_trials, raw_index, {})
        states = trial_event.get("States", {}) if isinstance(trial_event, dict) else {}
        states = states if isinstance(states, dict) else {}
        outcome, outcome_time = states_labeling(states)

        trial_code = _canonical_code(_indexed(trial_types, raw_index), contingency_index)
        block_code = _canonical_code(_indexed(block_types, raw_index), contingency_index)

        category_long: int | None
        if trial_code == 1:
            category_long = 0
        elif trial_code == 2:
            category_long = 1
        else:
            category_long = None

        if block_code == 0:
            true_block: str | None = "N"
        elif block_code == 1:
            true_block = "S"
        elif block_code == 2:
            true_block = "L"
        else:
            true_block = None
        canonical_blocks.append(true_block)

        choice_valid = outcome in VALID_CHOICE_OUTCOMES and category_long is not None
        rewarded: int | None = None
        choice_long: int | None = None
        if choice_valid:
            rewarded = int(outcome in REWARDED_OUTCOMES)
            choice_long = category_long if rewarded else 1 - category_long

        rows.append(
            {
                "mouse_id": metadata.mouse_id,
                "session_id": metadata.session_id,
                "session_index": pd.NA,
                "session_date": metadata.session_date,
                "trial_index": raw_index + 1,
                "source_trial_index": raw_index,
                "block_index": pd.NA,
                "true_block": true_block,
                "true_run_length": pd.NA,
                "category_long": category_long,
                "duration_s": _extract_duration(_indexed(processed_trials, raw_index)),
                "choice_long": choice_long,
                "rewarded": rewarded,
                "omission": int(outcome == "DidNotChoose"),
                "choice_valid": int(choice_valid),
                # This matches the previous valid-response logic. The task audit
                # must revise it if category feedback occurs after omissions.
                "feedback_valid": int(choice_valid),
                "outcome": outcome,
                "outcome_time_s": outcome_time,
                "warm_up": int(raw_index < warm_up_trials),
                "jitter_flag": int(_optional_float(_indexed(jitter_flags, raw_index)) == 1.0),
                "opto_tag": _indexed(opto_types, raw_index),
                "task_version": metadata.task_version,
                "contingency_index": contingency_index,
                "source_file": str(Path(source_file).expanduser().resolve()),
                "source_sha256": source_sha256,
            }
        )

    block_indices, run_lengths = _contiguous_block_metadata(canonical_blocks)
    for row, block_index, run_length in zip(rows, block_indices, run_lengths):
        row["block_index"] = block_index
        row["true_run_length"] = run_length

    frame = pd.DataFrame(rows, columns=CANONICAL_TRIAL_COLUMNS)
    for column in (
        "session_index",
        "trial_index",
        "source_trial_index",
        "block_index",
        "true_run_length",
        "category_long",
        "choice_long",
        "rewarded",
        "omission",
        "choice_valid",
        "feedback_valid",
        "warm_up",
        "jitter_flag",
        "contingency_index",
    ):
        frame[column] = pd.array(frame[column], dtype="Int64")
    frame["duration_s"] = pd.to_numeric(frame["duration_s"], errors="coerce")
    frame["outcome_time_s"] = pd.to_numeric(frame["outcome_time_s"], errors="coerce")
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    return frame


def select_legacy_model_trials(canonical_trials: pd.DataFrame) -> pd.DataFrame:
    """Select the valid-choice view used by the previous fitting notebook.

    Purpose
    -------
    Match the legacy valid-outcome selection while emitting the binary names
    required by M0: ``duration_s`` and ``choice_long``.

    Mathematical definition
    -----------------------
    Rows are retained only for Reward/Punish and their naive variants, with a
    finite duration, known binary category, known block, and reconstructed
    binary choice.

    Parameters
    ----------
    canonical_trials:
        One-row-per-task-trial canonical table from :func:`prepare_session`.

    Returns
    -------
    pandas.DataFrame
        Filtered rows in chronological order with a one-based
        ``model_trial_index`` within each session.

    Causal restrictions
    -------------------
    Category, block, and reward are retained for audit/diagnostics but must not
    enter the M0 pre-choice probability.

    Important edge cases
    --------------------
    Invalid trials remain in the canonical table even though they are absent
    here. Therefore their raw temporal locations are never lost.

    Related tests
    -------------
    ``test_prepare_session_reconstructs_legacy_choices_and_keeps_omission``.
    """

    required = {
        "choice_valid",
        "duration_s",
        "category_long",
        "choice_long",
        "true_block",
        "outcome",
        "mouse_id",
        "session_id",
        "trial_index",
    }
    missing = sorted(required.difference(canonical_trials.columns))
    if missing:
        raise KeyError(f"Canonical trial table is missing columns: {missing}")

    valid = (
        canonical_trials["choice_valid"].eq(1)
        & canonical_trials["outcome"].isin(VALID_CHOICE_OUTCOMES)
        & canonical_trials["duration_s"].notna()
        & canonical_trials["category_long"].isin([0, 1])
        & canonical_trials["choice_long"].isin([0, 1])
        & canonical_trials["true_block"].isin(["N", "S", "L"])
    )
    model_trials = canonical_trials.loc[valid].copy()
    model_trials["model_trial_index"] = (
        model_trials.groupby(["mouse_id", "session_id"], sort=False).cumcount() + 1
    ).astype("Int64")
    return model_trials.reset_index(drop=True)


def has_required_isi_bins(
    model_trials: pd.DataFrame,
    *,
    expected_bins: int = 5,
    decimals: int = 3,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Apply the previous notebook's exact short/long ISI-level criterion.

    Purpose
    -------
    Reproduce ``has_required_isi_bins`` from the previous notebook using the
    M0 binary category encoding.

    Parameters
    ----------
    model_trials:
        Valid-choice rows for exactly one session.
    expected_bins:
        Required number of unique duration levels per category. The legacy
        default is exactly five, not at least five.
    decimals:
        Decimal places used before uniqueness is evaluated.

    Returns
    -------
    tuple
        ``(passes, short_levels, long_levels)`` with sorted NumPy arrays in
        seconds.

    Constraints
    -----------
    ``category_long`` must use 0 for short and 1 for long.

    Related tests
    -------------
    ``test_has_required_isi_bins_requires_exact_counts``.
    """

    if expected_bins <= 0:
        raise ValueError("expected_bins must be positive")
    short_levels = np.unique(
        np.round(
            model_trials.loc[model_trials["category_long"].eq(0), "duration_s"].dropna(),
            decimals,
        )
    )
    long_levels = np.unique(
        np.round(
            model_trials.loc[model_trials["category_long"].eq(1), "duration_s"].dropna(),
            decimals,
        )
    )
    passes = len(short_levels) == expected_bins and len(long_levels) == expected_bins
    return passes, short_levels, long_levels


def _session_audit_row(
    trials: pd.DataFrame,
    model_trials: pd.DataFrame,
    *,
    passes_isi_filter: bool,
    included: bool,
    short_levels: np.ndarray,
    long_levels: np.ndarray,
) -> dict[str, Any]:
    """Build one serializable session-level audit record."""

    first = trials.iloc[0]
    invalid_choice_trials = int(len(trials) - int(trials["choice_valid"].sum()))
    return {
        "mouse_id": first["mouse_id"],
        "session_id": first["session_id"],
        "session_index": int(first["session_index"]),
        "session_date": first["session_date"],
        "task_version": first["task_version"],
        "source_file": first["source_file"],
        "source_sha256": first["source_sha256"],
        "n_task_trials": int(len(trials)),
        "n_choice_valid_trials": int(trials["choice_valid"].sum()),
        "n_model_eligible_trials": int(len(model_trials)),
        "n_omissions": int(trials["omission"].sum()),
        "n_invalid_choice_trials": invalid_choice_trials,
        "n_feedback_valid": int(trials["feedback_valid"].sum()),
        "n_reward": int(trials["outcome"].eq("Reward").sum()),
        "n_punish": int(trials["outcome"].eq("Punish").sum()),
        "n_reward_naive": int(trials["outcome"].eq("RewardNaive").sum()),
        "n_punish_naive": int(trials["outcome"].eq("PunishNaive").sum()),
        "short_isi_levels_s": ";".join(f"{value:.3f}" for value in short_levels),
        "long_isi_levels_s": ";".join(f"{value:.3f}" for value in long_levels),
        "n_short_isi_levels": int(len(short_levels)),
        "n_long_isi_levels": int(len(long_levels)),
        "passes_legacy_isi_filter": bool(passes_isi_filter),
        "included_in_model_trials": bool(included),
        "exclusion_reason": "" if included else "legacy_isi_bin_requirement",
    }


def prepare_dataset(
    paths: Iterable[str | Path],
    *,
    require_legacy_isi_bins: bool = True,
    expected_isi_bins: int = 5,
    isi_decimals: int = 3,
) -> DataPreparationResult:
    """Load, canonicalize, audit, and combine behavioral sessions.

    Purpose
    -------
    Create the immutable input tables needed before chronological splitting and
    M0 fitting, while preserving the legacy model-trial inclusion rules.

    Parameters
    ----------
    paths:
        Iterable of MATLAB session paths. Directory order is ignored.
    require_legacy_isi_bins:
        If true, only sessions with exactly ``expected_isi_bins`` short and
        long duration levels enter ``model_trials``. All sessions remain in the
        canonical and session audit tables.
    expected_isi_bins:
        Exact legacy duration-level count per category.
    isi_decimals:
        Rounding precision used for the level count.

    Returns
    -------
    DataPreparationResult
        Canonical trials, filtered model trials, and the session audit table.

    Constraints
    -----------
    Sessions are sorted separately within each mouse by recording timestamp and
    stable session ID. ``session_index`` is one-based within mouse.

    Numerical stability
    -------------------
    Nonfinite durations never enter ``model_trials``. Raw checksums are stored
    so future fit results can identify their exact source data.

    Causal restrictions
    -------------------
    This function prepares diagnostic task fields but does not expose them as
    predictive inputs. The later M0 evaluator must use only duration and pooled
    parameters.

    Important edge cases
    --------------------
    Failed files raise an exception rather than being silently skipped. Empty
    input and zero-trial sessions also raise because a pooled fit would not be
    reproducible if sessions disappeared implicitly.

    Related tests
    -------------
    Synthetic tests cover sorting, exact bin filtering, and legacy choice
    reconstruction. Real-file smoke tests verify observed dataset counts.
    """

    sources = [Path(path).expanduser().resolve() for path in paths]
    if not sources:
        raise ValueError("At least one behavioral MATLAB path is required")
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate behavioral MATLAB paths were provided")

    indexed_sources: list[tuple[SessionFileMetadata, Path]] = []
    for source in sources:
        metadata = parse_behavior_file_path(source)
        indexed_sources.append((metadata, source))

    indexed_sources.sort(
        key=lambda item: (item[0].mouse_id, item[0].session_date, item[0].session_id)
    )
    per_mouse_counts: dict[str, int] = {}
    canonical_frames: list[pd.DataFrame] = []

    # Decode one MATLAB file at a time. Some SessionData structs are large, so
    # retaining every decoded session until sorting is unnecessarily expensive.
    for metadata, source in indexed_sources:
        per_mouse_counts[metadata.mouse_id] = per_mouse_counts.get(metadata.mouse_id, 0) + 1
        session_index = per_mouse_counts[metadata.mouse_id]
        checksum = _sha256_file(source)
        session_data = load_session_mat(source)
        frame = prepare_session(
            session_data,
            metadata,
            source_file=source,
            source_sha256=checksum,
        )
        if frame.empty:
            raise ValueError(f"Session contains no task trials: {source}")
        frame["session_index"] = pd.array(
            [session_index] * len(frame), dtype="Int64"
        )
        canonical_frames.append(frame)

    canonical = pd.concat(canonical_frames, ignore_index=True)
    audit_rows: list[dict[str, Any]] = []
    included_model_frames: list[pd.DataFrame] = []

    for (_, _), session_trials in canonical.groupby(
        ["mouse_id", "session_id"], sort=False, observed=False
    ):
        session_trials = session_trials.sort_values("trial_index").reset_index(drop=True)
        session_model = select_legacy_model_trials(session_trials)
        passes, short_levels, long_levels = has_required_isi_bins(
            session_model,
            expected_bins=expected_isi_bins,
            decimals=isi_decimals,
        )
        included = passes or not require_legacy_isi_bins
        if included:
            included_model_frames.append(session_model)
        audit_rows.append(
            _session_audit_row(
                session_trials,
                session_model,
                passes_isi_filter=passes,
                included=included,
                short_levels=short_levels,
                long_levels=long_levels,
            )
        )

    model_trials = (
        pd.concat(included_model_frames, ignore_index=True)
        if included_model_frames
        else canonical.iloc[0:0].assign(model_trial_index=pd.Series(dtype="Int64"))
    )
    sessions = pd.DataFrame(audit_rows).sort_values(
        ["mouse_id", "session_index", "session_id"]
    ).reset_index(drop=True)
    return DataPreparationResult(
        canonical_trials=canonical.reset_index(drop=True),
        model_trials=model_trials.reset_index(drop=True),
        sessions=sessions,
    )
