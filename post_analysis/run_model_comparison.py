"""Run the full M0-M4 model-comparison post-analysis for one mouse.

Usage:

    python -u post_analysis/run_model_comparison.py --mouse MC06

This only reads ``results_sliding_window/`` and ``results_single_session/``
and writes CSV tables, figures, and a Markdown summary under
``results_post_analysis/model_comparison/<mouse>/``.
"""

from __future__ import annotations

import argparse

from post_analysis.model_comparison import collect, figures, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC06", help="Mouse/subject ID, e.g. MC06")
    args = parser.parse_args()

    print(f"[1/3] collecting tables for {args.mouse} ...")
    outputs = collect.run(args.mouse)
    for name, path in outputs.items():
        print(f"  wrote {path}")

    print(f"[2/3] rendering figures for {args.mouse} ...")
    figures.run(args.mouse)

    print(f"[3/3] writing summary report for {args.mouse} ...")
    report_path = report.run(args.mouse)
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()
