#!/usr/bin/env python3
"""Print (space-separated) the window ids of a model/mouse whose fit finished
all eight starts but has no accepted winner and no complete manifest -- the
windows forced_best_train_nll.py handles. Read-only.

  python extra_tools/list_no_winner.py --model M0_sensory_v1 --mouse MC06
Windows that are neither complete nor a clean no-winner (e.g. unfinished) are
reported on stderr and NOT printed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results_sliding_window"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mouse", required=True)
    args = parser.parse_args()
    root = ROOT / args.model / args.mouse / "full_session_matched_v1"
    ids = []
    for window in sorted(root.glob("window_*")):
        if not window.is_dir():
            continue
        wid = int(window.name.split("_")[1])
        manifest = window / "window_manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("status") == "complete":
            continue
        summary_path = next((p for p in (window / "fit_result.json", window / "starts" / "multistart_summary.json") if p.is_file()), None)
        summary = json.loads(summary_path.read_text()) if summary_path else None
        if summary and len(summary.get("runs", [])) == 8 and summary.get("winner_start_id") is None:
            ids.append(wid)
        else:
            print(f"window {wid}: not complete and not a clean no-winner (unfinished?)", file=sys.stderr)
    print(" ".join(map(str, ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
