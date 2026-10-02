"""Write a concise, auditable summary of the expanded comparison."""

from __future__ import annotations

import argparse

import pandas as pd

from post_analysis import common

from .config import ANALYSIS_NAME, MODEL_BY_ID, MODEL_ORDER, SPLIT_ROLES, validate_mouse_id


def run(mouse_id: str):
    mouse_id = validate_mouse_id(mouse_id)
    tables_dir, _ = common.ensure_output_dirs(ANALYSIS_NAME, mouse_id)
    status = pd.read_csv(tables_dir / "model_completion_status.csv")
    rankings = pd.read_csv(tables_dir / "figure00_rankings.csv")
    eligible = status[status["eligible_for_comparison"].astype(str).str.lower().eq("true")]
    incomplete = status[~status.index.isin(eligible.index)]
    provisional = len(eligible) != len(MODEL_ORDER)
    lines = [
        f"# Expanded sliding-window model comparison: {mouse_id}", "",
        f"**Status:** {'PROVISIONAL' if provisional else 'FINAL CONFIGURED SET'} — "
        f"{len(eligible)}/{len(MODEL_ORDER)} configured models are complete and eligible.", "",
        "All included results passed the `full_session_matched_v1` directory and "
        "`full_session_matched` manifest checks, the canonical 9/3/1 schedule check, "
        "a converged-winner check, exact evaluated-trial matching, and metric recomputation.", "",
        "## Completion status", "",
        "| Model | Status | Complete windows | Included? |", "|---|---:|---:|---:|",
    ]
    for _, row in status.iterrows():
        lines.append(f"| {row['model_label']} | {row['status']} | {row['complete_window_count']}/{row['expected_windows']} | {'yes' if row['eligible_for_comparison'] else 'no'} |")
    if not incomplete.empty:
        lines += ["", "Incomplete models are listed for progress only. Their partial windows are never used in rankings, pairwise tests, win counts, or information criteria."]
    lines += ["", "## Figure 0 ranking", ""]
    for split_role in SPLIT_ROLES:
        all_best = rankings[(rankings["split_role"] == split_role) & (rankings["ranking_scope"] == "all_models")].sort_values("rank").iloc[0]
        deployable_best = rankings[(rankings["split_role"] == split_role) & (rankings["ranking_scope"] == "deployable_only")].sort_values("rank").iloc[0]
        lines.append(
            f"- **{split_role.capitalize()}:** best overall = {MODEL_BY_ID[all_best['model_id']].label} "
            f"(mean window NLL {all_best['mean_window_nll']:.6f}); best non-oracle = "
            f"{MODEL_BY_ID[deployable_best['model_id']].label} ({deployable_best['mean_window_nll']:.6f})."
        )
    lines += [
        "", "## Interpretation limits", "",
        "- The test split is the primary descriptive comparison. Train and validation summarize fitting/model-selection behavior.",
        "- Sliding windows overlap, so windows are not independent experimental replicates. Bootstrap intervals and all p-values are exploratory within-mouse summaries, not population inference.",
        "- M1/M2 known-switch models use true switch timing and are oracle benchmarks. The tables therefore include both all-model and deployable-only rankings.",
        f"- Provisional pairwise p-values are explicitly marked `provisional_descriptive_do_not_interpret`; final inferential interpretation waits for all {len(MODEL_ORDER)} models.",
        "- Every Figure 1–23 panel is sliding-window-only. Legacy single-session slots were replaced by explicitly labeled sliding-window diagnostics rather than silently reusing fine-tuned results.",
        "", "## Outputs", "",
        "Figure 0 ranks models separately for train, validation, and test. Figures 1–23 form the complete sliding-window suite; the original Figure 1–6 artifacts remain available, and Figures 7–23 add subset, stability, parameter, availability, agreement, event, psychometric, prior, hazard, and rare-trial views.", "",
    ]
    path = tables_dir.parent / "summary.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC08")
    args = parser.parse_args()
    print(f"wrote {run(args.mouse)}")


if __name__ == "__main__":
    main()
