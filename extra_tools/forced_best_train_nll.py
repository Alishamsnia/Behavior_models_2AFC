#!/usr/bin/env python3
"""Produce the normal per-window analysis for `no_winner` sliding windows by
forcing the start with the lowest TRAINING NLL, without retraining and without
modifying any existing code or result.

Two stages, both explicit:

  stage   (default)  Copies each target window's saved fit into a STAGING tree,
                     marks the lowest-training-NLL start as winner there, and
                     runs the unmodified runner's own post-fit path on it
                     (predictions, metrics, figures, manifest). Nothing under
                     the real results directory is touched.
  install (--install) Copies the staged files into the real window folder.
                     Original fit_result.json / window_manifest.json are kept
                     next to them as *_original_no_winner.json. Refuses to
                     overwrite an existing backup.

Stage usage (runner arguments go after `--`; do NOT pass --sliding-output-dir,
the tool sets it; pass the same optimizer/registry/raw-data/workflow arguments
used for the original fit):

  python extra_tools/forced_best_train_nll.py \
      --runner m1_dynamic --windows 5 8 11 13 \
      --original-sliding-dir "$PWD/results_sliding_window" \
      --staging-dir "$PWD/extra_tools_output/forced_best_staging" \
      -- --raw-data-root .../SCHR_MC06 --registry config/..._mc06.json \
         --workflow full_session_matched --sliding-only --resume

Install usage:

  python extra_tools/forced_best_train_nll.py --install \
      --original-sliding-dir "$PWD/results_sliding_window" \
      --staging-dir "$PWD/extra_tools_output/forced_best_staging"

Runners: known_switch, m1_dynamic, m2_dynamic, action_bias, wsls, m0, extended_wsls, m1_stimulus,
pure_action, extended_action, previous_action_bias, previous_wsls_bias, action_kernel_bias.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

RUNNERS = {
    "known_switch": "run_m1_known_switch_sliding_window",
    "m1_dynamic": "run_m1_dynamic_block_length_sliding_window",
    "m2_dynamic": "run_m2_dynamic_block_length_sliding_window",
    "action_bias": "run_m2_dynamic_previous_action_bias_sliding_window",
    "wsls": "run_m2_dynamic_previous_wsls_bias_sliding_window",
    "m0": "run_m0_sliding_window",
    "extended_wsls": "run_extended_wsls_sliding_window",
    "m1_stimulus": "run_m1_stimulus_kernel_sliding_window",
    "pure_action": "run_pure_action_sliding_window",
    "extended_action": "run_extended_action_sliding_window",
    "previous_action_bias": "run_m1_previous_action_bias_sliding_window",
    "previous_wsls_bias": "run_m1_previous_wsls_bias_sliding_window",
    "action_kernel_bias": "run_m2_dynamic_action_kernel_bias_sliding_window",
}
MARKER_NAME = "NOT_STANDARD_WINDOW_forced_best_training_nll.txt"
FORCED_STATUS = "forced_best_training_nll_not_converged"
BACKUP_RESULT = "fit_result_original_no_winner.json"
BACKUP_MANIFEST = "window_manifest_original_no_winner.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_nll(run: dict[str, Any]) -> float:
    for key in ("recalculated_best_training_mean_nll", "best_fresh_training_mean_nll"):
        value = run.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    raise RuntimeError(f"start {run.get('start_id')} has no finite training NLL")


def _forced_start(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("runs", [])
    if len(runs) != 8:
        raise RuntimeError(f"expected 8 saved starts, found {len(runs)}")
    if summary.get("winner_start_id") is not None:
        raise RuntimeError("window already has a winner; refusing to force one")
    if any(run.get("exception_message") for run in runs):
        raise RuntimeError("a start recorded an exception; not a clean no-winner window")
    return min(runs, key=lambda run: (_training_nll(run), int(run["start_id"])))


def _saved_summary_path(window_dir: Path) -> Path | None:
    """fit_result.json if the runner saved one, else the multistart summary
    (the WSLS runner raises on a no-winner window before writing fit_result.json)."""

    for candidate in (window_dir / "fit_result.json", window_dir / "starts" / "multistart_summary.json"):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- stage
def _stage(args: argparse.Namespace, runner_args: list[str]) -> int:
    if "--sliding-output-dir" in runner_args or "--single-output-dir" in runner_args:
        raise SystemExit("Do not pass --sliding-output-dir/--single-output-dir; the tool sets them")
    if "--sliding-only" not in runner_args:
        raise SystemExit("Pass --sliding-only after `--`")
    original_base = args.original_sliding_dir.resolve()
    staging_base = args.staging_dir.resolve()
    if staging_base == original_base or original_base in staging_base.parents:
        raise SystemExit("--staging-dir must be outside the real results directory")
    targets = set(args.windows)

    runner = importlib.import_module(RUNNERS[args.runner])
    # Hard guarantee against retraining: any attempt to fit raises.
    def _no_training(*_a, **_k):
        raise RuntimeError("forced-best tool must never retrain; a staleness check triggered a refit")

    for name in dir(runner):
        if re.fullmatch(r"fit_(.*_)?all_initializations", name):
            setattr(runner, name, _no_training)
    runner.manifest_fingerprint_is_stale = lambda *_a, **_k: False
    runner._generate_sliding_global_figures = lambda *_a, **_k: print(
        "global sliding figures skipped by forced-best tool", flush=True
    )
    done: set[int] = set()

    def _run_forced_windows(prepared, windows, registry, run_args, root):
        root = Path(root)
        original_root = original_base / root.relative_to(staging_base)
        parameters: dict[int, list[float]] = {}
        for window in windows:
            if window.window_id not in targets:
                continue
            source = original_root / window.directory_name
            stage_dir = root / window.directory_name
            result_path = _saved_summary_path(source)
            if result_path is None:
                raise SystemExit(f"window {window.window_id}: no saved fit summary at {source}")
            summary = json.loads(result_path.read_text(encoding="utf-8"))
            if runner.result_registry_is_stale(summary, registry):
                raise SystemExit(
                    f"window {window.window_id}: saved fit does not match the current registry "
                    "identity (or has none); refusing to force-analyse it"
                )
            forced = _forced_start(summary)
            manifest_path = source / "window_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
            if manifest.get("status") == "complete":
                raise SystemExit(f"window {window.window_id}: already complete; not a no-winner window")
            if hasattr(runner, "_optimizer_provenance"):
                saved_opt = manifest.get("optimizer_configuration")
                if saved_opt is not None and saved_opt != runner._optimizer_provenance(run_args):
                    raise SystemExit(
                        f"window {window.window_id}: optimizer settings differ from the original fit; "
                        "pass the same optimizer arguments"
                    )
                saved_src = manifest.get("source_sha256")
                if saved_src is not None and saved_src != list(prepared.source_sha256):
                    raise SystemExit(f"window {window.window_id}: raw data changed since the fit")
            if stage_dir.exists():
                shutil.rmtree(stage_dir)  # our own staging copy only
            shutil.copytree(source, stage_dir)

            staged_summary = dict(summary)
            staged_summary["winner_start_id"] = int(forced["start_id"])
            staged_summary["selection_status"] = FORCED_STATUS
            staged_summary["original_selection_status"] = summary.get("selection_status")
            staged_summary["forced_selection"] = {
                "rule": "lowest recalculated training mean NLL across all eight starts, regardless of convergence",
                "forced_start_id": int(forced["start_id"]),
                "forced_start_training_mean_nll": _training_nll(forced),
                "forced_start_converged": forced.get("converged"),
                "forced_start_tail_occupancy_acceptable": forced.get("tail_occupancy_acceptable"),
                "all_start_training_mean_nll": {
                    str(r["start_id"]): _training_nll(r) for r in summary["runs"]
                },
            }
            (stage_dir / "fit_result.json").write_text(
                json.dumps(staged_summary, indent=2), encoding="utf-8"
            )
            # Make the runner's own input-freshness check pass for this already-verified fit.
            staged_manifest = dict(manifest)
            staged_manifest.setdefault("status", "no_winner")
            if hasattr(runner, "_runner_sha256"):
                staged_manifest["source_sha256"] = list(prepared.source_sha256)
                staged_manifest["optimizer_configuration"] = runner._optimizer_provenance(run_args)
                staged_manifest["runner_sha256"] = runner._runner_sha256()
            (stage_dir / "window_manifest.json").write_text(
                json.dumps(staged_manifest, indent=2), encoding="utf-8"
            )

            runner._fit_window(prepared, window, registry, run_args, root)

            final_path = stage_dir / "window_manifest.json"
            final = json.loads(final_path.read_text(encoding="utf-8"))
            if final.get("status") != "complete":
                raise RuntimeError(f"window {window.window_id}: staged analysis did not complete")
            final["forced_best_training_nll"] = True
            final["selection_status"] = FORCED_STATUS
            final["forced_winner_start_id"] = int(forced["start_id"])
            final["forced_selection_note"] = (
                "NOT a standard window: no start met the converged + tail-acceptable criteria; "
                "the start with the lowest training NLL was used. No retraining was performed."
            )
            final["original_fit_result_sha256"] = _sha256(result_path)
            final["original_manifest_status"] = manifest.get("status")
            final_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
            (stage_dir / MARKER_NAME).write_text(
                "NOT A STANDARD WINDOW\n"
                f"model: {registry.model_id}  mouse: {prepared.mouse_id}  window: {window.window_id}\n"
                "No start passed the converged + tail-acceptable acceptance gate.\n"
                f"Winner forced to start {forced['start_id']} (lowest training mean NLL "
                f"{_training_nll(forced):.6f}); converged={forced.get('converged')}, "
                f"tail_ok={forced.get('tail_occupancy_acceptable')}.\n"
                "All outputs were generated from the saved fit; nothing was retrained.\n"
                f"Original saved fit sha256: {_sha256(result_path)}\n",
                encoding="utf-8",
            )
            parameters[window.window_id] = list(forced["best_constrained_parameters"])
            done.add(window.window_id)
            print(
                f"window={window.window_id:03d} status=forced_best_staged start={forced['start_id']}",
                flush=True,
            )
        return parameters

    runner._run_windows = _run_forced_windows
    sys.argv = [
        RUNNERS[args.runner] + ".py",
        *runner_args,
        "--sliding-output-dir", str(staging_base),
        "--single-output-dir", str(staging_base / "_unused_single_session"),
    ]
    code = runner.main()
    missing = targets - done
    if missing:
        print(f"ERROR: requested windows not staged: {sorted(missing)}", file=sys.stderr)
        return 1
    print(f"Staged windows {sorted(done)} under {staging_base}. Inspect, then run with --install.")
    return code


# ------------------------------------------------------------------------- install
def _install(args: argparse.Namespace) -> int:
    original_base = args.original_sliding_dir.resolve()
    staging_base = args.staging_dir.resolve()
    markers = sorted(staging_base.rglob(MARKER_NAME))
    if not markers:
        raise SystemExit(f"No staged windows found under {staging_base}")
    plans = []
    for marker in markers:
        stage_dir = marker.parent
        target = original_base / stage_dir.relative_to(staging_base)
        original_result = _saved_summary_path(target)
        if original_result is None:
            raise SystemExit(f"{target}: original fit summary missing")
        if json.loads(original_result.read_text(encoding="utf-8")).get("winner_start_id") is not None:
            raise SystemExit(f"{target}: already has a winner (already installed?)")
        manifest = target / "window_manifest.json"
        if (target / BACKUP_RESULT).exists() or (target / BACKUP_MANIFEST).exists():
            raise SystemExit(f"{target}: backup already exists; refusing to overwrite")
        plans.append((stage_dir, target, manifest))
    for stage_dir, target, manifest in plans:
        if (target / "fit_result.json").exists():
            shutil.move(str(target / "fit_result.json"), str(target / BACKUP_RESULT))
        if manifest.exists():
            shutil.move(str(manifest), str(target / BACKUP_MANIFEST))
        shutil.copytree(
            stage_dir, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("starts")
        )
        print(f"installed: {target}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    runner_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, runner_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runner", choices=sorted(RUNNERS))
    parser.add_argument("--windows", type=int, nargs="+")
    parser.add_argument("--original-sliding-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    if args.install:
        if runner_args or args.runner or args.windows:
            parser.error("--install takes only --original-sliding-dir and --staging-dir")
        return _install(args)
    if not args.runner or not args.windows:
        parser.error("stage mode requires --runner and --windows")
    return _stage(args, runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
