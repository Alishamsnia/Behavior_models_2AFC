"""CLI for retrospective checkpoint-policy comparison."""

from __future__ import annotations

import argparse

from post_analysis.checkpoint_policy_comparison.analysis import run_analysis
from post_analysis.checkpoint_policy_comparison.figures import make_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse", default="MC08")
    args = parser.parse_args()
    output = run_analysis(args.mouse)
    make_outputs(output)
    print(output)


if __name__ == "__main__":
    main()
