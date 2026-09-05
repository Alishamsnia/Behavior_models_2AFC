"""Build the per-model raw-behavior input CSVs from the single-session
model-comparison winner assignments.

For each model (M0-M4): filter every contributing subject's raw trial CSV
down to just the sessions that model won, write one filtered per-subject CSV
per model, then pool them with the unmodified, reused
``pool_subject_csvs.pool_subject_csvs`` (same function the original
Behavior/New_version pipeline uses). A model with zero winning sessions gets
an explicit "no data" manifest and no pooled CSV -- the downstream notebook
runner skips it rather than failing on an empty pool.

Reads the original ``<SUBJECT>_final_data.csv`` files read-only from
``session_mapping.DEFAULT_RAW_BEHAVIOR_ROOT``; writes only under
``results_raw_behavior_by_winner/<MODEL>/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import session_mapping as sm
from pool_subject_csvs import pool_subject_csvs

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results_raw_behavior_by_winner"


def build_model_input(
    model_id: str,
    assignments: pd.DataFrame,
    raw_behavior_root: Path,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    """Write the filtered per-subject CSVs + pooled CSV + manifest for one
    model. Returns the manifest dict. Always writes a manifest, even when
    there are zero winning sessions for this model.
    """
    model_dir = results_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    model_rows = assignments.loc[assignments["model_id"] == model_id]
    manifest = {
        "model_id": model_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "raw_behavior_root": str(raw_behavior_root),
        "n_sessions_won": int(len(model_rows)),
        "subjects": {},
        "pooled_csv": None,
        "pooled_trial_count": None,
        "pooled_session_count": None,
    }

    if model_rows.empty:
        manifest["status"] = "no_data"
        _write_manifest(model_dir, manifest)
        return manifest

    subject_csv_paths = []
    for subject, subject_rows in model_rows.groupby("subject", sort=True):
        raw_csv = raw_behavior_root / f"{subject}_final_data.csv"
        source_df = pd.read_csv(raw_csv)
        winning_idx = set(subject_rows["raw_session_idx"].tolist())
        filtered = source_df.loc[
            pd.to_numeric(source_df["session_idx"], errors="raise").isin(winning_idx)
        ].copy()

        found_idx = set(pd.to_numeric(filtered["session_idx"]).unique().tolist())
        missing_idx = winning_idx - found_idx
        if missing_idx:
            raise ValueError(
                f"{model_id}/{subject}: raw_session_idx {sorted(missing_idx)} not "
                f"found in {raw_csv}."
            )

        out_path = model_dir / f"{subject}_final_data.csv"
        filtered.to_csv(out_path, index=False)
        subject_csv_paths.append(out_path)

        manifest["subjects"][subject] = {
            "source_csv": str(raw_csv),
            "source_trial_count": int(len(source_df)),
            "n_sessions_won": int(len(subject_rows)),
            "sessions": sorted(
                subject_rows[["raw_session_idx", "session_date", "chronological_rank", "session_id"]]
                .drop_duplicates()
                .to_dict("records"),
                key=lambda r: r["raw_session_idx"],
            ),
            "filtered_trial_count": int(len(filtered)),
            "filtered_csv": str(out_path),
        }

    pooled_path = model_dir / "all_subjects_final_data.csv"
    pooled = pool_subject_csvs(subject_csv_paths, pooled_path)
    manifest["status"] = "ok"
    manifest["pooled_csv"] = str(pooled_path)
    manifest["pooled_trial_count"] = int(len(pooled))
    manifest["pooled_session_count"] = int(pooled["session_idx"].nunique())
    manifest["pooled_subjects"] = sorted(pooled["subject"].unique().tolist())
    _write_manifest(model_dir, manifest)
    return manifest


def _write_manifest(model_dir: Path, manifest: dict) -> None:
    (model_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(raw_behavior_root: Path = sm.DEFAULT_RAW_BEHAVIOR_ROOT) -> dict[str, dict]:
    assignments = sm.all_winner_assignments(raw_behavior_root)
    manifests = {}
    for model_id in sm.MODEL_ORDER:
        manifests[model_id] = build_model_input(model_id, assignments, raw_behavior_root)
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-behavior-root", type=Path, default=sm.DEFAULT_RAW_BEHAVIOR_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = run(args.raw_behavior_root)
    for model_id, manifest in manifests.items():
        if manifest["status"] == "no_data":
            print(f"{model_id}: no winning sessions -- no data")
        else:
            print(
                f"{model_id}: {manifest['n_sessions_won']} sessions, "
                f"{manifest['pooled_trial_count']} pooled trials, "
                f"subjects={manifest['pooled_subjects']} -> {manifest['pooled_csv']}"
            )


if __name__ == "__main__":
    main()
