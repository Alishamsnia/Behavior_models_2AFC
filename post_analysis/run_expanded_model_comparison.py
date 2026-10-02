"""Run the expanded full-session-matched sliding-window comparison."""

from __future__ import annotations

import argparse

from post_analysis.expanded_model_comparison import collect, figures, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC08")
    args = parser.parse_args()
    print(f"[1/3] auditing and collecting {args.mouse}")
    collect.run(args.mouse)
    print(f"[2/3] rendering figures for {args.mouse}")
    figures.run(args.mouse)
    print(f"[3/3] writing report for {args.mouse}")
    print(f"wrote {report.run(args.mouse)}")


if __name__ == "__main__":
    main()

