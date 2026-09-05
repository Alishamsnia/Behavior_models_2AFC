"""Run the raw-behavior analysis notebooks separately for each of the five
Bayesian models' single-session winner sessions.

Bundle: the same six notebooks run_control_analyses.py runs for its MC16/MC17
cohort, minus the GLM regression notebook (Regression_model.ipynb) by
explicit request:
  performance_pooled_grand_average, psychometric_analysis, learning_curve,
  licking_analysis, rare_trials_analysis, meta_learning_transition_analysis.

Reuses the original Behavior/New_version notebooks and pool_subject_csvs.py
verbatim (copied into this folder, never modified) plus its
prelude-injection + nbclient execution technique, generalized via
notebook_patches.py to any subject cohort instead of one fixed pair -- so
adding a new subject later (once it has model results in
results_post_analysis/model_comparison/<SUBJECT>/) needs no change here.

Reads Behavior/New_version read-only. Writes only under
results_raw_behavior_by_winner/.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import build_model_inputs
import notebook_patches as patches
import session_mapping as sm

SOURCE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = build_model_inputs.RESULTS_ROOT
KERNEL_NAME = "model_winner_analysis"

ANALYSIS_NOTEBOOKS = [
    "performance_pooled_grand_average.ipynb",
    "psychometric_analysis.ipynb",
    "learning_curve.ipynb",
    "licking_analysis.ipynb",
    "rare_trials_analysis.ipynb",
    "meta_learning_transition_analysis.ipynb",
]


def _prelude(results_dir: Path, pooled_path: Path) -> str:
    source_dir = str(SOURCE_DIR)
    result_path = str(results_dir)
    csv_path = str(pooled_path)
    return f"""
import builtins
import os
import sys
from pathlib import Path
import pandas as _winner_pd

_WINNER_SOURCE_DIR = Path({source_dir!r})
_WINNER_RESULTS_DIR = Path({result_path!r})
_WINNER_CSV = Path({csv_path!r})
if str(_WINNER_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_WINNER_SOURCE_DIR))
os.chdir(_WINNER_RESULTS_DIR)

def _winner_input(prompt=''):
    prompt_lower = str(prompt).lower()
    if 'csv' in prompt_lower:
        return str(_WINNER_CSV)
    if 'number of bins' in prompt_lower:
        return '4'
    if 'directory to save' in prompt_lower:
        return str(_WINNER_RESULTS_DIR)
    raise RuntimeError(f'Unexpected interactive prompt: {{prompt}}')

builtins.input = _winner_input
_winner_df = _winner_pd.read_csv(_WINNER_CSV)
print(
    'MODEL-WINNER INPUT:',
    len(_winner_df), 'trials,',
    _winner_df['session_uid'].nunique(), 'sessions, subjects',
    sorted(_winner_df['subject'].unique()),
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _winner_plt
_winner_plt.show = lambda *args, **kwargs: _winner_plt.close('all')
"""


def prepare_notebook_copy(
    notebook_name: str, results_dir: Path, pooled_path: Path, subjects: list[str],
):
    notebook_path = SOURCE_DIR / notebook_name
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(_prelude(results_dir, pooled_path)))

    for cell in notebook.cells[1:]:
        if cell.cell_type != "code":
            continue
        source = cell.source
        source = patches.patch_paths(source, str(pooled_path), str(results_dir))
        source = patches.patch_input_calls(source)
        source = patches.patch_subject_rows(notebook_name, source, subjects)
        if notebook_name == "licking_analysis.ipynb":
            source = patches.patch_licking_cohorts(source, subjects)
        if notebook_name == "meta_learning_transition_analysis.ipynb":
            source = patches.patch_meta_learning_import(source, subjects)
        cell.source = source

    notebook.metadata.setdefault("kernelspec", {})
    notebook.metadata["kernelspec"] = {
        "display_name": KERNEL_NAME, "language": "python", "name": KERNEL_NAME,
    }
    return notebook


def _install_local_kernelspec(results_dir: Path) -> None:
    kernel_dir = results_dir / ".jupyter" / "kernels" / KERNEL_NAME
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": KERNEL_NAME, "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(kernel_spec, indent=2) + "\n")
    existing = os.environ.get("JUPYTER_PATH")
    local_path = str(results_dir / ".jupyter")
    os.environ["JUPYTER_PATH"] = local_path if not existing else local_path + os.pathsep + existing


def execute_notebook(
    notebook_name: str, results_dir: Path, pooled_path: Path, subjects: list[str],
    timeout: int = 1800,
) -> dict:
    notebook = prepare_notebook_copy(notebook_name, results_dir, pooled_path, subjects)
    executed_dir = results_dir / "executed_notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)
    output_path = executed_dir / notebook_name

    before = {p.resolve() for p in results_dir.rglob("*") if p.is_file()}
    started = datetime.now().astimezone()
    print(f"EXECUTING {notebook_name}", flush=True)
    client = NotebookClient(
        notebook, timeout=timeout, kernel_name=KERNEL_NAME,
        allow_errors=False, resources={"metadata": {"path": str(results_dir)}},
    )
    status, error = "completed", None
    try:
        client.execute()
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        nbformat.write(notebook, output_path)

    after = {p.resolve() for p in results_dir.rglob("*") if p.is_file()}
    created = sorted(
        str(p.relative_to(results_dir.resolve()))
        for p in after - before
        if p != output_path.resolve()
    )
    finished = datetime.now().astimezone()
    record = {
        "status": status,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "executed_notebook": str(output_path),
        "created_files": created,
        "error": error,
    }
    suffix = f": {error}" if error else ""
    print(f"{status.upper()} {notebook_name} in {record['elapsed_seconds']:.1f}s{suffix}", flush=True)
    return record


def run_model(model_id: str, input_manifest: dict, notebooks: list[str], timeout: int) -> dict:
    """Run `notebooks` for one model. If a manifest from an earlier
    (possibly partial, e.g. --notebook-restricted) invocation already exists
    for this model, its ``notebook_runs`` entries are preserved and merged
    with this run's results, so the manifest always reflects every notebook
    ever run for this model rather than only the most recent invocation.
    """
    results_dir = RESULTS_ROOT / model_id
    run_manifest_path = results_dir / "notebook_run_manifest.json"
    previous_runs = {}
    if run_manifest_path.is_file():
        try:
            previous_runs = json.loads(run_manifest_path.read_text()).get("notebook_runs", {})
        except (json.JSONDecodeError, OSError):
            previous_runs = {}
    run_manifest = {
        "model_id": model_id,
        "status": input_manifest["status"],
        "started_at": datetime.now().astimezone().isoformat(),
        "notebooks_requested": sorted(set(previous_runs) | set(notebooks), key=ANALYSIS_NOTEBOOKS.index),
        "notebook_runs": dict(previous_runs),
    }
    if input_manifest["status"] == "no_data":
        run_manifest["note"] = "No sessions won by this model -- nothing to analyze."
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
        print(f"{model_id}: no data, skipping notebook execution")
        return run_manifest

    _install_local_kernelspec(results_dir)
    pooled_path = Path(input_manifest["pooled_csv"])
    subjects = input_manifest["pooled_subjects"]
    # A notebook's pooled input CSV is rebuilt from the current winner
    # assignments on every invocation (see build_model_inputs.run above), so
    # a "completed" status from an earlier run does not by itself mean this
    # run's input still matches -- winner assignments or upstream model
    # results can have changed since. Hash the input the notebook would
    # actually consume and only skip when both the status and that hash
    # still match what was previously recorded.
    current_input_checksum = sha256(pooled_path.read_bytes()).hexdigest()
    for notebook_name in notebooks:
        previous_record = previous_runs.get(notebook_name, {})
        if (
            previous_record.get("status") == "completed"
            and previous_record.get("input_pooled_csv_sha256") == current_input_checksum
        ):
            print(f"SKIP {notebook_name} (already completed, input unchanged)", flush=True)
            continue
        if previous_record.get("status") == "completed":
            print(
                f"RERUN {notebook_name} (input CSV changed since the completed run "
                "was recorded)",
                flush=True,
            )
        try:
            record = execute_notebook(
                notebook_name, results_dir, pooled_path, subjects, timeout=timeout,
            )
            record["input_pooled_csv_sha256"] = current_input_checksum
        except Exception as exc:
            # Never abort the whole model run because one notebook's
            # execution setup itself failed (as opposed to a cell raising,
            # which execute_notebook already catches internally).
            record = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "input_pooled_csv_sha256": current_input_checksum,
            }
        run_manifest["notebook_runs"][notebook_name] = record
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sm.MODEL_ORDER, action="append",
        help="Restrict to these models (default: all 5).",
    )
    parser.add_argument("--notebook", choices=ANALYSIS_NOTEBOOKS, action="append")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--raw-behavior-root", type=Path, default=sm.DEFAULT_RAW_BEHAVIOR_ROOT)
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Only build the filtered/pooled input CSVs; do not execute notebooks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifests = build_model_inputs.run(args.raw_behavior_root)
    if args.prepare_only:
        print(json.dumps(input_manifests, indent=2))
        return

    models = args.model or sm.MODEL_ORDER
    notebooks = args.notebook or ANALYSIS_NOTEBOOKS
    for model_id in models:
        run_model(model_id, input_manifests[model_id], notebooks, args.timeout)


if __name__ == "__main__":
    main()
