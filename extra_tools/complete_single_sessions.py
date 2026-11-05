#!/usr/bin/env python3
"""Fill in missing full_session_matched single-session fine-tunes for one
model/mouse, using each rolling window's INSTALLED winner -- including windows
whose winner was forced by extra_tools/forced_best_train_nll.py.

Why a separate tool: every runner fingerprints all of src/, so after new model
packages were added a ``--resume`` of an older model would treat every rolling
window as stale and refit it, and ``--single-only`` refuses to run. This tool
drives the model's own, unmodified runner in-process instead:

* rolling windows are never trained: every ``fit_*_all_initializations`` is
  replaced by a function that raises, and ``_run_windows`` only reads the
  installed winner from the REAL rolling folder (status complete, matched
  workflow, same raw data);
* the runner's own rolling-root writes (experiment manifest, session table,
  block audit) go to a scratch folder; the real rolling folder is checksummed
  before and after and must be unchanged;
* only the missing sessions (no session folder, or a manifest that is not
  complete/ineligible) are handed to the runner's ``_fine_tune_all``, so its
  spawned workers never see -- and can never refit -- an existing session;
* the rebuilt session->window initialization table must equal the existing
  one for every session that already had a fit (same winners read the same
  way); the previous table and run summary are kept next to the new ones.

``--evaluate-failed-best`` is for sessions whose fine-tune stopped with a
numerical failure (an L-BFGS step sent a parameter to infinity): the session is
scored at the best point that fine-tune reached before failing (its recorded
lowest training NLL), with zero further optimizer steps -- the single-session
counterpart of the forced lowest-training-NLL window winners. The failed
result is kept under ``original_failed_fit/`` and the manifest is flagged.

``--reproduce RANK`` refits one already-finished session into scratch with the
current code and compares parameters and NLL with the stored fit.

  python extra_tools/complete_single_sessions.py --model M2_dynamic_block_length_v1 --mouse MC06 --workers 8
  python extra_tools/complete_single_sessions.py --model Extended_WSLS_v1 --mouse MC08 --reproduce 20
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA_ROOT = Path("/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior")
WORKFLOW = "full_session_matched"
WORKFLOW_DIR = "full_session_matched_v1"
FITTED_STATUSES = ("complete", "ineligible")  # ineligible = fitted, excluded only by that model's own acceptance gate
# results/model directory -> (runner module, registry config stem)
MODELS = {
    "Pure_WSLS_v1": ("run_pure_wsls_sliding_window", "pure_wsls_v1"),
    "Extended_WSLS_v1": ("run_extended_wsls_sliding_window", "extended_wsls_v1"),
    "Pure_action_v1": ("run_pure_action_sliding_window", "pure_action_v1"),
    "Extended_action_v1": ("run_extended_action_sliding_window", "extended_action_v1"),
    "M0_sensory_v1": ("run_m0_sliding_window", "m0_sensory_v1"),
    "M1_action_kernel_v1": ("run_m1_action_kernel_sliding_window", "m1_action_kernel_v1"),
    "M1_stimulus_kernel_v1": ("run_m1_stimulus_kernel_sliding_window", "m1_stimulus_kernel_v1"),
    "M1_previous_action_bias_v1": ("run_m1_previous_action_bias_sliding_window", "m1_previous_action_bias_v1"),
    "M1_previous_wsls_bias_v1": ("run_m1_previous_wsls_bias_sliding_window", "m1_previous_wsls_bias_v1"),
    "M2_hmm_static_v1": ("run_m2_sliding_window", "m2_hmm_static_v1"),
    "M2_dynamic_block_length_v1": ("run_m2_dynamic_block_length_sliding_window", "m2_dynamic_block_length_v1"),
    "M2_dynamic_previous_wsls_bias_v1": ("run_m2_dynamic_previous_wsls_bias_sliding_window", "m2_dynamic_previous_wsls_bias_v1"),
    "M2_dynamic_previous_action_bias_v1": ("run_m2_dynamic_previous_action_bias_sliding_window", "m2_dynamic_previous_action_bias_v1"),
}


def _results_root(base: str, model: str, mouse: str) -> Path:
    """Standard <base>/<model>/<mouse>/full_session_matched_v1, or the copy some
    MC06/MC07 runs saved one level deeper (same rule as run_expanded_for_mouse)."""
    standard = REPOSITORY_ROOT / base / model / mouse / WORKFLOW_DIR
    nested = REPOSITORY_ROOT / base / model / mouse / model / mouse / WORKFLOW_DIR
    if not any(standard.glob("window_*")) and not any(standard.glob("session_*")) and \
            (any(nested.glob("window_*")) or any(nested.glob("session_*"))):
        return nested
    return standard


def _tree_snapshot(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _session_rank(directory: Path) -> int:
    return int(re.match(r"session_(\d+)_", directory.name).group(1))


def _fitted_ranks(single_root: Path) -> set[int]:
    ranks = set()
    for directory in single_root.glob("session_*"):
        manifest_path = directory / "single_session_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("workflow") == WORKFLOW and manifest.get("status") in FITTED_STATUSES \
                and (directory / "trial_predictions.csv").is_file():
            ranks.add(_session_rank(directory))
    return ranks


def _window_table(sliding_root: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(sliding_root.glob("window_*/window_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        rows.append({"window_id": int(manifest["window_id"]), "test_rank": int(manifest["test_rank"]),
                     "test_session_id": manifest["test_session_id"], "directory": manifest_path.parent,
                     "forced": bool(manifest.get("forced_best_training_nll")), "manifest": manifest})
    return pd.DataFrame(rows)


def _winner_parameters(window_dir: Path) -> list[float]:
    summary = json.loads((window_dir / "fit_result.json").read_text())
    winner_id = summary.get("winner_start_id")
    if winner_id is None:
        raise RuntimeError(f"{window_dir.name}: no winner installed (run the forced-best tool first)")
    runs = [run for run in summary["runs"] if run.get("start_id") == winner_id]
    if len(runs) != 1:
        raise RuntimeError(f"{window_dir.name}: winner_start_id {winner_id} matches {len(runs)} runs")
    return [float(value) for value in runs[0]["best_constrained_parameters"]]


def _parameter_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith("initial_") and pd.api.types.is_numeric_dtype(table[c])]


def _failed_best(single_root: Path, ranks: set[int]) -> dict[int, list[float]]:
    best = {}
    for rank in ranks:
        directory = next(single_root.glob(f"session_{rank:03d}_*"), None)
        if directory is None:
            raise SystemExit(f"--evaluate-failed-best: no failed fit for rank {rank}")
        manifest = json.loads((directory / "single_session_manifest.json").read_text())
        fit = json.loads((directory / "fine_tune_fit_result.json").read_text())
        values = fit.get("best_constrained_parameters")
        if manifest.get("status") != "failed" or not values or not np.isfinite(values).all():
            raise SystemExit(f"--evaluate-failed-best: rank {rank} is not a failed fit with a finite best point")
        best[rank] = [float(v) for v in values]
    return best


def run(model: str, mouse: str, workers: int, reproduce: int | None, dry_run: bool,
        evaluate_failed_best: bool = False) -> int:
    runner_name, config_stem = MODELS[model]
    registry = REPOSITORY_ROOT / "config" / (f"{config_stem}.json" if mouse == "MC08" else f"{config_stem}_{mouse.lower()}.json")
    real_sliding = _results_root("results_sliding_window", model, mouse)
    real_single = _results_root("results_single_session", model, mouse)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scratch = REPOSITORY_ROOT / "extra_tools_output" / "single_session_completion" / f"{model}_{mouse}_{stamp}"

    windows = _window_table(real_sliding)
    bad = windows[[m.get("status") != "complete" or m.get("workflow") != WORKFLOW for m in windows.manifest]]
    if windows.empty or not bad.empty:
        raise SystemExit(f"{model} {mouse}: rolling windows missing or not complete/matched: {bad.window_id.tolist()}")
    fitted = _fitted_ranks(real_single) if real_single.is_dir() else set()
    if reproduce is not None:
        if reproduce not in fitted:
            raise SystemExit(f"--reproduce {reproduce}: no finished session at that rank to compare against")
        targets = {reproduce}
    else:
        targets = set(windows.test_rank) - fitted
    forced_ranks = set(windows.loc[windows.forced, "test_rank"])
    print(f"{model} {mouse}: windows={len(windows)} fitted_sessions={len(fitted)} "
          f"targets={sorted(targets)} (from forced windows: {sorted(targets & forced_ranks)})", flush=True)
    if not targets or dry_run:
        return 0
    failed_best = _failed_best(real_single, targets) if evaluate_failed_best else {}

    runner = importlib.import_module(runner_name)

    def _no_training(*_args, **_kwargs):
        raise RuntimeError("complete_single_sessions must never train a rolling window")

    for name in dir(runner):
        if re.fullmatch(r"fit_(.*_)?all_initializations", name):
            setattr(runner, name, _no_training)
    runner._generate_sliding_global_figures = lambda *_a, **_k: None
    runner._generate_single_global_figures = lambda *_a, **_k: None

    by_window = windows.set_index("window_id")

    def _installed_winners(prepared, selected, registry_obj, run_args, root):
        parameters = {}
        for window in selected:
            row = by_window.loc[window.window_id]
            if row.directory.name != window.directory_name:
                raise RuntimeError(f"window {window.window_id}: folder {row.directory.name} != {window.directory_name}")
            if row.manifest.get("source_sha256") not in (None, list(prepared.source_sha256)):
                raise RuntimeError(f"window {window.window_id}: rolling fit used different raw data")
            parameters[window.window_id] = _winner_parameters(row.directory)
        return parameters

    runner._run_windows = _installed_winners
    original_fine_tune = runner._fine_tune_all
    single_out = (scratch / "single") if reproduce is not None else (REPOSITORY_ROOT / "results_single_session")
    expected_single_root = (single_out / model / mouse / WORKFLOW_DIR).resolve()
    seen: dict = {}

    def _fine_tune_targets(prepared, registry_obj, run_args, single_root, initializations):
        if Path(single_root).resolve() != expected_single_root:
            raise RuntimeError(f"unexpected single-session root {single_root}")
        subset = initializations[initializations["chronological_rank"].astype(int).isin(targets)]
        if set(subset["chronological_rank"].astype(int)) != targets:
            raise RuntimeError(f"initialization table lacks target ranks {sorted(targets - set(subset.chronological_rank))}")
        seen["initializations"] = initializations.copy()
        if failed_best:
            subset = subset.copy()
            columns = [f"initial_{name}" for name in runner.PARAMETER_NAMES]
            for rank, values in failed_best.items():
                subset.loc[subset["chronological_rank"].astype(int).eq(rank), columns] = values
        return original_fine_tune(prepared, registry_obj, run_args, single_root, subset)

    runner._fine_tune_all = _fine_tune_targets

    keep = {}
    if reproduce is None and real_single.is_dir():
        backup = scratch / "backup"
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("matched_window_test_initializations.csv", "all_session_block_cut_audit.csv",
                     "single_session_run_summary.json"):
            if (real_single / name).is_file():
                shutil.copy2(real_single / name, backup / name)
                keep[name] = backup / name

    for rank in failed_best:
        directory = next(real_single.glob(f"session_{rank:03d}_*"))
        archive = directory / "original_failed_fit"
        archive.mkdir(exist_ok=False)
        for name in ("fine_tune_fit_result.json", "single_session_manifest.json"):
            shutil.move(directory / name, archive / name)

    rolling_before = _tree_snapshot(real_sliding)
    sys.argv = [
        f"{runner_name}.py", "--raw-data-root", str(DATA_ROOT / f"SCHR_{mouse}"), "--registry", str(registry),
        "--optimizer", "lbfgs", "--resume",
        # Older runners default to the legacy partitioned protocol; never let that apply here.
        *(["--workflow", WORKFLOW] if '"--workflow"' in Path(runner.__file__).read_text() else []),
        "--single-session-workers", str(max(1, min(workers, len(targets)))),
        "--single-session-threads-per-worker", "1",
        "--sliding-output-dir", str(scratch / "sliding_unused"),
        "--single-output-dir", str(single_out),
        # Zero steps is not allowed; one step of size 1e-12 with a single line-search
        # evaluation scores the given point (verified below to within 1e-9).
        *(["--lbfgs-max-steps", "1", "--lbfgs-learning-rate", "1e-12", "--lbfgs-max-evaluations", "1"]
          if failed_best else []),
    ]
    code = runner.main()
    if _tree_snapshot(real_sliding) != rolling_before:
        raise RuntimeError("real rolling-window folder changed; this must never happen")

    initializations = seen["initializations"]
    if reproduce is not None:
        return _compare_reproduction(real_single, expected_single_root, reproduce)

    # Same winners for sessions that already had a fit: the old table must be a subset of the new one.
    if "matched_window_test_initializations.csv" in keep:
        old = pd.read_csv(keep["matched_window_test_initializations.csv"]).set_index("session_id")
        new = initializations.set_index("session_id").loc[old.index]
        columns = _parameter_columns(old)
        if not columns or not np.allclose(old[columns].to_numpy(float), new[columns].to_numpy(float), rtol=0, atol=1e-12):
            raise RuntimeError("rebuilt initialization table disagrees with the existing one for already-fitted sessions")
        shutil.copy2(keep["matched_window_test_initializations.csv"],
                     real_single / f"matched_window_test_initializations_before_completion_{stamp}.csv")
    if "all_session_block_cut_audit.csv" in keep and \
            not pd.read_csv(keep["all_session_block_cut_audit.csv"]).equals(pd.read_csv(real_single / "all_session_block_cut_audit.csv")):
        raise RuntimeError("block-cut audit changed; refusing to continue")
    if (real_single / "single_session_run_summary.json").is_file():
        shutil.move(real_single / "single_session_run_summary.json",
                    real_single / f"single_session_completion_summary_{stamp}.json")
    if "single_session_run_summary.json" in keep:
        shutil.copy2(keep["single_session_run_summary.json"], real_single / "single_session_run_summary.json")

    rows = []
    for directory in sorted(real_single.glob("session_*")):
        rank = _session_rank(directory)
        if rank not in targets:
            continue
        manifest_path = directory / "single_session_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        manifest["completed_by"] = {"tool": "extra_tools/complete_single_sessions.py", "run": stamp,
                                    "source_window_forced_best_training_nll": rank in forced_ranks}
        if rank in failed_best:
            scored = json.loads((directory / "fine_tune_fit_result.json").read_text())["best_constrained_parameters"]
            if not np.allclose(scored, failed_best[rank], rtol=0, atol=1e-9):
                raise RuntimeError(f"rank {rank}: scored parameters moved away from the failed fit's best point")
            manifest["completed_by"]["evaluated_at_failed_fit_best"] = {
                "note": "fine-tune failed numerically; scored at its lowest-training-NLL point with zero optimizer steps",
                "parameters": failed_best[rank], "original_result": "original_failed_fit/"}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        rows.append({"rank": rank, "status": manifest.get("status"), "converged": manifest.get("converged"),
                     "forced_window": rank in forced_ranks})
    report = pd.DataFrame(rows)
    print(report.to_string(index=False), flush=True)
    missing = targets - set(report["rank"]) if len(report) else targets
    failed = report[~report["status"].isin(FITTED_STATUSES)] if len(report) else report
    if missing or len(failed):
        print(f"NOT COMPLETE: missing={sorted(missing)} failed={failed['rank'].tolist() if len(failed) else []}", flush=True)
        return 1
    print(f"{model} {mouse}: all {len(targets)} target sessions fitted", flush=True)
    return code or 0


def _compare_reproduction(real_single: Path, scratch_single: Path, rank: int) -> int:
    real = next(real_single.glob(f"session_{rank:03d}_*"))
    fresh = next(scratch_single.glob(f"session_{rank:03d}_*"))
    result = {}
    for name, directory in (("stored", real), ("fresh", fresh)):
        fit = json.loads((directory / "fine_tune_fit_result.json").read_text())
        metrics_path = next(p for p in (directory / "prefix_and_heldout_metrics.csv",
                                        directory / "split_and_session_metrics.csv") if p.is_file())
        metrics = pd.read_csv(metrics_path).query("session_id == 'ALL_SESSIONS'")
        result[name] = (np.asarray(fit["best_constrained_parameters"], float),
                        metrics.set_index("split_role")["mean_nll"].to_dict())
    parameter_diff = float(np.max(np.abs(result["stored"][0] - result["fresh"][0])))
    nll_diff = max(abs(result["stored"][1][k] - result["fresh"][1][k]) for k in result["stored"][1])
    ok = parameter_diff < 1e-6 and nll_diff < 1e-8
    print(f"REPRODUCTION rank {rank}: max |param diff|={parameter_diff:.3g}  max |NLL diff|={nll_diff:.3g}  "
          f"stored NLL={result['stored'][1]}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--mouse", required=True, choices=("MC06", "MC07", "MC08"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reproduce", type=int, default=None, metavar="RANK")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-failed-best", action="store_true")
    args = parser.parse_args()
    return run(args.model, args.mouse, args.workers, args.reproduce, args.dry_run, args.evaluate_failed_best)


if __name__ == "__main__":
    raise SystemExit(main())
