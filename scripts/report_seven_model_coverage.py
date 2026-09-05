#!/usr/bin/env python3
"""Report exactly what result coverage exists for each of the seven
implemented models (M0-M4 plus M1_dynamic_block_length_v1 and
M2_dynamic_block_length_v1), per mouse, for both the sliding-window and
single-session pipelines.

This does not run anything -- it only inspects what is already on disk --
and is meant to answer one question precisely: "is a true seven-model
comparison possible yet, and if not, exactly what is missing?" As of this
writing the answer is no: M2_dynamic_block_length_v1 has no results of any
kind, M1_dynamic_block_length_v1 has partial sliding-window results and no
single-session results, and even the five nested M0-M4 models have unequal
single-session coverage (see post_analysis.common.MODEL_ORDER for why the
main comparison pipeline is scoped to those five models only).

Usage:
    python scripts/report_seven_model_coverage.py [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SLIDING_ROOT = REPOSITORY_ROOT / "results_sliding_window"
SINGLE_ROOT = REPOSITORY_ROOT / "results_single_session"

# (model_id, directory name under results_sliding_window / results_single_session)
MODELS = [
    ("M0", "M0_sensory_v1"),
    ("M1", "M1_hmm_true_v1"),
    ("M2", "M2_hmm_static_v1"),
    ("M3", "M3_hmm_delta_v1"),
    ("M4", "M4_hsmm_delta_v1"),
    ("M1_dynamic", "M1_dynamic_block_length_v1"),
    ("M2_dynamic", "M2_dynamic_block_length_v1"),
]


def _manifest_status(directory: Path, manifest_name: str) -> str | None:
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
    except (json.JSONDecodeError, OSError):
        return "unreadable"


def _count_dirs(root: Path, prefix: str, manifest_name: str) -> dict[str, dict[str, int]]:
    """{mouse_id: {"count": N, "complete": M}} for every child of `root`
    whose name matches an existing mouse directory containing subdirectories
    named `prefix*`."""

    per_mouse: dict[str, dict[str, int]] = {}
    if not root.is_dir():
        return per_mouse
    for mouse_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # Sliding-window layout nests under a workflow directory
        # (full_session_matched_v1); single-session layouts vary similarly.
        candidates = list(mouse_dir.rglob(f"{prefix}*"))
        run_dirs = [p for p in candidates if p.is_dir()]
        if not run_dirs:
            continue
        complete = sum(
            1 for d in run_dirs if _manifest_status(d, manifest_name) == "complete"
        )
        per_mouse[mouse_dir.name] = {"count": len(run_dirs), "complete": complete}
    return per_mouse


def collect_coverage() -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for model_id, dir_name in MODELS:
        sliding = _count_dirs(SLIDING_ROOT / dir_name, "window_", "window_manifest.json")
        single = _count_dirs(SINGLE_ROOT / dir_name, "session_", "single_session_manifest.json")
        coverage[model_id] = {
            "dir_name": dir_name,
            "sliding_window": sliding,
            "single_session": single,
            "has_any_results": bool(sliding or single),
        }
    return coverage


def _fmt_per_mouse(per_mouse: dict[str, dict[str, int]]) -> str:
    if not per_mouse:
        return "none"
    return ", ".join(
        f"{mouse}={info['complete']}/{info['count']}" for mouse, info in per_mouse.items()
    )


def print_report(coverage: dict[str, dict[str, Any]]) -> None:
    print(f"{'model':<12} {'sliding-window (complete/total per mouse)':<45} single-session")
    print("-" * 100)
    for model_id, info in coverage.items():
        print(
            f"{model_id:<12} {_fmt_per_mouse(info['sliding_window']):<45} "
            f"{_fmt_per_mouse(info['single_session'])}"
        )
    print()
    missing = [m for m, info in coverage.items() if not info["has_any_results"]]
    partial_sw_only = [
        m for m, info in coverage.items()
        if info["has_any_results"] and not info["single_session"]
    ]
    if missing:
        print(f"No results of any kind: {', '.join(missing)}")
    if partial_sw_only:
        print(f"Sliding-window results only (no single-session): {', '.join(partial_sw_only)}")
    print(
        "\nA true seven-model comparison additionally requires post_analysis.common."
        "MODEL_ORDER to include the dynamic-block-length models, and a decision on how "
        "nested/delta-IC comparisons work across the two non-nested branches -- see "
        "post_analysis/common.py's comment on MODEL_SPECS for the two dynamic entries."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead")
    args = parser.parse_args()
    coverage = collect_coverage()
    if args.json:
        print(json.dumps(coverage, indent=2))
    else:
        print_report(coverage)


if __name__ == "__main__":
    main()
