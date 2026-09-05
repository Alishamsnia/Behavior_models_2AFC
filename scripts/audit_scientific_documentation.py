#!/usr/bin/env python3
"""Audit every registered M0, M1, and M2 scientific function against Part VI."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.documentation import (  # noqa: E402
    audit_scientific_docstrings,
    registered_scientific_functions,
)
from m4_hsmm_delta_v1.prior_state import capped_tail_hsmm_transition  # noqa: E402
from m4_hsmm_delta_v1.recovery import run_m4_design_preserving_recovery  # noqa: E402
from m4_hsmm_delta_v1.session import evaluate_m4_mouse_objective, evaluate_m4_session  # noqa: E402
from m4_hsmm_delta_v1.simulation import simulate_m4_choices  # noqa: E402


def main() -> int:
    """Print the documentation audit and return its shell status."""

    functions = registered_scientific_functions()
    functions.update({function.__name__: function for function in (
        capped_tail_hsmm_transition,
        evaluate_m4_session,
        evaluate_m4_mouse_objective,
        simulate_m4_choices,
        run_m4_design_preserving_recovery,
    )})
    violations = audit_scientific_docstrings(functions)
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print(f"Part VI documentation audit passed for {len(functions)} functions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
