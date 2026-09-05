from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_FILES = [
    "YH24LG_final_data.csv",
    "MC06_final_data.csv",
    "MC07_final_data.csv",
    "MC08_final_data.csv",
]


def subject_id_from_filename(path: Path) -> str:
    suffix = "_final_data.csv"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def load_subject_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "session_idx" not in df.columns:
        raise ValueError(f"{path} is missing required column: session_idx")

    subject_id = subject_id_from_filename(path)
    subject_session_idx = pd.to_numeric(df["session_idx"], errors="raise").astype(int)
    raw_subject = df["subject"] if "subject" in df.columns else pd.NA

    df = df.drop(columns=[column for column in ["session_idx", "subject"] if column in df.columns])
    df.insert(0, "source_file", path.name)
    df.insert(0, "raw_subject", raw_subject)
    df.insert(0, "subject_session_idx", subject_session_idx)
    df.insert(0, "subject", subject_id)
    df.insert(0, "session_uid", [f"{subject_id}_S{idx:03d}" for idx in subject_session_idx])

    return df


def pool_subject_csvs(source_paths: list[Path], output_path: Path) -> pd.DataFrame:
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing source CSV file(s):\n{missing_list}")

    pooled = pd.concat([load_subject_csv(path) for path in source_paths], ignore_index=True)

    session_lookup = pooled[["subject", "subject_session_idx"]].drop_duplicates().reset_index(drop=True)
    session_lookup["session_idx"] = range(1, len(session_lookup) + 1)
    pooled = pooled.merge(session_lookup, on=["subject", "subject_session_idx"], how="left", sort=False)
    pooled.insert(0, "session_idx", pooled.pop("session_idx"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_path, index=False)
    return pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pool processed per-subject 2AFC CSV files into one analysis-ready CSV."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="all_subjects_final_data.csv",
        help="Path for the pooled output CSV.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="Per-subject CSV files to pool. Defaults to the four known subject files in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = [Path(path) for path in (args.sources or DEFAULT_SOURCE_FILES)]
    output_path = Path(args.output)

    pooled = pool_subject_csvs(source_paths, output_path)

    summary = (
        pooled.groupby("subject", sort=False)
        .agg(sessions=("subject_session_idx", "nunique"), trials=("subject_session_idx", "size"))
        .reset_index()
    )
    print(f"Wrote {output_path}")
    print(f"Total subjects: {pooled['subject'].nunique()}")
    print(f"Total sessions: {pooled['session_idx'].nunique()}")
    print(f"Total trials: {len(pooled)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
