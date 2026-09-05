"""Tests for the enforceable Part VI documentation contract."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.documentation import (  # noqa: E402
    REQUIRED_SCIENTIFIC_DOCSTRING_SECTIONS,
    audit_scientific_docstrings,
    registered_scientific_functions,
)


class PartSixDocumentationTests(unittest.TestCase):
    def test_every_registered_scientific_function_passes_contract(self) -> None:
        functions = registered_scientific_functions()
        self.assertGreaterEqual(len(functions), 49)
        required_m2 = {
            "prototypes",
            "decode_m2_raw_parameters",
            "encode_m2_constrained_parameters",
            "validate_m2_initialization_table",
            "m2_session_trials_from_frame",
            "evaluate_m2_session",
            "evaluate_m2_mouse_objective",
            "simulate_m2_choices",
            "simulate_m2_hmm_sessions",
            "simulate_m2_real_design_sessions",
            "run_m2_recovery_case",
            "run_m2_real_design_recovery_case",
            "evaluate_m2_hazard_contrast_surface",
            "load_m2_model_registry",
        }
        self.assertTrue(required_m2.issubset(functions))
        self.assertEqual(audit_scientific_docstrings(functions), ())

    def test_audit_reports_missing_sections_and_math(self) -> None:
        def undocumented() -> None:
            """Purpose only."""

        failures = audit_scientific_docstrings({"undocumented": undocumented})
        self.assertIn(
            "undocumented: missing Sphinx-compatible '.. math::' equation",
            failures,
        )
        for section in REQUIRED_SCIENTIFIC_DOCSTRING_SECTIONS[1:]:
            self.assertIn(
                f"undocumented: missing section {section!r}",
                failures,
            )


if __name__ == "__main__":
    unittest.main()
