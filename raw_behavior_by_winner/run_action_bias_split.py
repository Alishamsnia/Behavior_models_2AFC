"""MC08 raw-behavior analysis split by whether an action-bias model wins the session.

Same six notebooks as ``run_model_winner_analysis`` (reused unchanged), but the
sessions are grouped by the single-session held-out winner among the 13
expanded models instead of by M0-M4:

  action_bias_wins    -- the best model is sensory_actionHistory_lapse
                         (m1_action_kernel) or the
                         ..._learnDynamicBlockSwitch_actionSingle_lapse model
                         (m2_dynamic_repetition_bias)
  action_bias_not     -- any of the other 11 models wins

Each analysis PDF is written to ``<out>/raw_comparison/`` with the
action_bias_wins pages first and the action_bias_not pages after, each page
stamped with its group.  Nothing under results_raw_behavior_by_winner/ or
results_single_session/ is touched.

Stages (they need different interpreters -- notebooks run in the
Behavior/New_version venv, the rest in the repo's python):

    python raw_behavior_by_winner/run_action_bias_split.py --stage assign
    <New_version venv>/bin/python3 raw_behavior_by_winner/run_action_bias_split.py --stage run
    python raw_behavior_by_winner/run_action_bias_split.py --stage merge
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(REPO_ROOT))

SUBJECT = "MC08"
OUT_DIR = (
    REPO_ROOT / "results_post_analysis" / "expanded_model_comparison" / SUBJECT / "raw_comparison"
)
WORK_DIR = OUT_DIR / "_work"
GROUPS_CSV = OUT_DIR / "session_groups.csv"
ACTION_MODELS = ("m1_action_kernel", "m2_dynamic_repetition_bias")
GROUPS = {
    "action_bias_wins": "Action-bias model wins",
    "action_bias_not": "Action-bias model does NOT win",
}


def assign() -> None:
    from post_analysis.expanded_model_comparison.config import MODEL_SPECS
    from session_mapping import DEFAULT_RAW_BEHAVIOR_ROOT, _raw_session_idx_by_date, date_from_session_id

    rows = []
    for spec in MODEL_SPECS:
        root = REPO_ROOT / "results_single_session" / spec.directory / SUBJECT / "full_session_matched_v1"
        for session_dir in sorted(root.glob("session_*")):
            path = session_dir / "prefix_and_heldout_metrics.csv"
            if not path.is_file():
                path = session_dir / "split_and_session_metrics.csv"
            metrics = pd.read_csv(path)
            test = metrics[(metrics["split_role"] == "test") & (metrics["session_id"] == "ALL_SESSIONS")].iloc[0]
            rows.append({
                "chronological_rank": int(session_dir.name.split("_")[1]),
                "session_id": session_dir.name.split("_", 2)[2],
                "model_id": spec.model_id, "mean_nll": float(test["mean_nll"]),
                "mouse_accuracy": float(test["mouse_accuracy"]),
            })
    table = pd.DataFrame(rows)
    if table.groupby("chronological_rank")["model_id"].nunique().ne(len(MODEL_SPECS)).any():
        raise ValueError("every session must be scored by all 13 models")
    winners = table.loc[table.groupby("chronological_rank")["mean_nll"].idxmin()].copy()
    winners = winners.rename(columns={"model_id": "winning_model"})
    winners["group"] = winners["winning_model"].map(
        lambda model: "action_bias_wins" if model in ACTION_MODELS else "action_bias_not"
    )
    date_to_idx = _raw_session_idx_by_date(SUBJECT, DEFAULT_RAW_BEHAVIOR_ROOT)
    winners["raw_session_idx"] = [
        int(date_to_idx.loc[date_from_session_id(sid)]) for sid in winners["session_id"]
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    winners.sort_values("chronological_rank").to_csv(GROUPS_CSV, index=False)
    print(winners["group"].value_counts().to_string())
    print(f"wrote {GROUPS_CSV}")


def run(timeout: int) -> None:
    import run_model_winner_analysis as runner
    from pool_subject_csvs import pool_subject_csvs
    from session_mapping import DEFAULT_RAW_BEHAVIOR_ROOT

    groups = pd.read_csv(GROUPS_CSV)
    source = pd.read_csv(DEFAULT_RAW_BEHAVIOR_ROOT / f"{SUBJECT}_final_data.csv")
    for group in GROUPS:
        indices = set(groups.loc[groups["group"] == group, "raw_session_idx"])
        group_dir = WORK_DIR / group
        group_dir.mkdir(parents=True, exist_ok=True)
        filtered = source[pd.to_numeric(source["session_idx"]).isin(indices)]
        if set(pd.to_numeric(filtered["session_idx"]).unique()) != indices:
            raise ValueError(f"{group}: raw sessions missing from the MC08 CSV")
        subject_csv = group_dir / f"{SUBJECT}_final_data.csv"
        filtered.to_csv(subject_csv, index=False)
        pooled = group_dir / "all_subjects_final_data.csv"
        pool_subject_csvs([subject_csv], pooled)
        manifest = {"model_id": group, "status": "ok", "pooled_csv": str(pooled), "pooled_subjects": [SUBJECT]}
        (group_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        # run_model_winner_analysis.run_model keys results off RESULTS_ROOT/<id>;
        # point that at our work dir so the group folders are used as-is.
        runner.RESULTS_ROOT = WORK_DIR
        runner.run_model(group, manifest, runner.ANALYSIS_NOTEBOOKS, timeout)


def merge() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pypdf import PdfReader, PdfWriter

    groups = pd.read_csv(GROUPS_CSV)
    counts = groups["group"].value_counts().to_dict()
    names = {group: {p.name for p in (WORK_DIR / group).glob("*.pdf")} for group in GROUPS}
    common = sorted(set.intersection(*names.values()))
    for group, found in names.items():
        for missing in sorted(found - set(common)):
            print(f"only in {group}: {missing}")

    def stamp(page, label: str) -> None:
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        fig = plt.figure(figsize=(width / 72, height / 72))
        fig.patch.set_alpha(0)
        fig.text(0.01, 0.995, label, ha="left", va="top", fontsize=11, fontweight="bold",
                 color="#9C0006", bbox={"facecolor": "white", "edgecolor": "#9C0006", "pad": 3})
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", transparent=True)
        plt.close(fig)
        page.merge_page(PdfReader(buffer).pages[0])

    for name in common:
        writer = PdfWriter()
        for group, title in GROUPS.items():
            for page in PdfReader(WORK_DIR / group / name).pages:
                stamp(page, f"{SUBJECT} - {title} ({counts[group]} sessions)")
                writer.add_page(page)
        with (OUT_DIR / name).open("wb") as handle:
            writer.write(handle)
    print(f"merged {len(common)} PDFs into {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--stage", choices=("assign", "run", "merge"), required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    {"assign": assign, "merge": merge}.get(args.stage, lambda: run(args.timeout))()


if __name__ == "__main__":
    main()
