#!/usr/bin/env python3
"""Run scripts/run_m1_known_switch_sliding_window.py for ONLY the windows
that are not already complete, without touching any existing code.

Why this exists: the original runner's --resume refits (or re-post-processes)
a window whenever its saved fit lacks registry checksums or its figure
manifest is an older schema, which is true of the already-complete
M1_known_switch_v1 MC06 windows. Plain --resume would therefore overwrite
them. This wrapper imports the unmodified runner, replaces its per-window
function with a guard that returns an already-complete window's saved result
untouched, and otherwise calls the original function unchanged.

Requirements: pass --window-parallel-workers 1 (the guard lives in this
process; the runner's spawned window workers would not inherit it).
Global (root-level) sliding figures are skipped; generate them separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import run_m1_known_switch_sliding_window as runner  # noqa: E402

_original_fit_window = runner._fit_window


def _missing_only_fit_window(prepared, window, registry, args, root):
    output = Path(root) / window.directory_name
    manifest_path = output / "window_manifest.json"
    result_path = output / "fit_result.json"
    if manifest_path.is_file() and result_path.is_file():
        try:
            status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = None
        if status == "complete":
            print(f"window={window.window_id:03d} status=kept_complete_untouched", flush=True)
            return json.loads(result_path.read_text(encoding="utf-8"))
    return _original_fit_window(prepared, window, registry, args, root)


def _skip_global_figures(*_args, **_kwargs):
    print("global sliding figures skipped by missing-only wrapper", flush=True)


def main() -> int:
    argv = sys.argv[1:]
    if "--window-parallel-workers" not in argv or argv[argv.index("--window-parallel-workers") + 1] != "1":
        raise SystemExit("Pass --window-parallel-workers 1 (required by this wrapper)")
    if "--single-only" in argv or "--window-limit" in argv:
        raise SystemExit("This wrapper supports the full sliding stage only")
    if "--sliding-only" not in argv:
        raise SystemExit("Pass --sliding-only")
    runner._fit_window = _missing_only_fit_window
    runner._generate_sliding_global_figures = _skip_global_figures
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
