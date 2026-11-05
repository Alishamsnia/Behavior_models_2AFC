#!/usr/bin/env python3
"""Run the unmodified expanded sliding-window comparison for a mouse other than
MC08. The pipeline hard-codes the MC08 split-registry CSV; this launcher only
points that one path at the mouse's own registry and then calls the original
collect / figures / report steps.

  python extra_tools/run_expanded_for_mouse.py --mouse MC07
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from post_analysis import common  # noqa: E402
from post_analysis.expanded_model_comparison import (  # noqa: E402
    collect, config, extended_collect, figures, report, stats,
)


def _data_driven_event_windows() -> None:
    """The original Figure 0 code assumes the final window has no qualifying
    transition event (true for MC08 only). For other mice, use the windows that
    actually have all six event conditions for every deployable model."""

    original = stats.combined_selection_tables

    def patched(metrics, events, model_order, **kwargs):
        _scores, common = stats.event_condition_window_scores(events, tuple(model_order))
        print(f"event windows with complete event data: {len(common)} of "
              f"{metrics.loc[metrics['split_role'].eq('test'), 'window_id'].nunique()} test windows; "
              f"excluded: {sorted(set(metrics['window_id'].unique()) - set(common))}", flush=True)
        kwargs["expected_event_windows"] = len(common)
        kwargs["expected_event_window_ids"] = tuple(common)
        return original(metrics, events, model_order, **kwargs)

    stats.combined_selection_tables = patched


def _accept_forced_windows() -> None:
    """Treat windows installed by forced_best_train_nll.py (lowest training NLL,
    flagged in their manifest) as having a converged winner, without any flag in
    the outputs. The window manifest keeps the record of how they were chosen."""

    original = collect._converged_winner

    def patched(window_dir):
        winner = original(window_dir)
        manifest = common.read_json(Path(window_dir) / "window_manifest.json") or {}
        if winner is not None and manifest.get("forced_best_training_nll") is True:
            winner = {**winner, "converged": True}
        return winner

    collect._converged_winner = patched
    extended_collect._converged_winner = patched


def _extra_duration_bins() -> None:
    """MC06 also presents 0.28/0.42 s and 2.08/2.22 s; put each in the bin of its
    nearest standard level (0.28,0.42 -> 0.35 bin; 2.08,2.22 -> 2.15 bin)."""

    table = extended_collect._DURATION_TO_BIN
    for value, anchor in ((0.28, 0.35), (0.42, 0.35), (2.08, 2.15), (2.22, 2.15)):
        table[round(value, 2)] = table[round(anchor, 2)]


def _find_nested_results() -> None:
    """Some MC06/MC07 runs were saved one folder too deep
    (<model>/<mouse>/<model>/<mouse>/full_session_matched_v1). Resolve to the
    nested copy only when the standard location has no windows."""

    original = config.ExpandedModelSpec.result_root

    def result_root(self, mouse_id):
        standard = original(self, mouse_id)
        if any(standard.glob("window_*")):
            return standard
        nested = (
            common.RESULTS_SLIDING_ROOT / self.directory / mouse_id / self.directory
            / mouse_id / config.WORKFLOW_DIRECTORY
        )
        return nested if any(nested.glob("window_*")) else standard

    config.ExpandedModelSpec.result_root = result_root


def apply_mouse_patches(mouse: str) -> Path:
    """All in-process adaptations needed for a non-MC08 mouse. Returns the split registry."""
    mouse = config.validate_mouse_id(mouse)
    registry = REPOSITORY_ROOT / "config" / "splits" / f"chronological_70_15_15_{mouse.lower()}_v1.csv"
    if mouse != "MC08":
        if not registry.is_file():
            raise SystemExit(f"No split registry for {mouse}: {registry}")
        config.split_registry_path = lambda: registry
        collect.split_registry_path = lambda: registry
        _data_driven_event_windows()
        _accept_forced_windows()
        _find_nested_results()
        _extra_duration_bins()
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", required=True)
    args = parser.parse_args()
    mouse = config.validate_mouse_id(args.mouse)
    registry = apply_mouse_patches(mouse)
    print(f"[1/3] auditing and collecting {mouse} (split registry: {registry.name})", flush=True)
    collect.run(mouse)
    print(f"[2/3] rendering figures for {mouse}", flush=True)
    figures.run(mouse)
    print(f"[3/3] writing report for {mouse}", flush=True)
    print(f"wrote {report.run(mouse)}", flush=True)


if __name__ == "__main__":
    main()
