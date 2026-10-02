"""Real-MATLAB-data structural check for Extended_WSLS_v1's data pipeline.

This is data loading and adapter conversion ONLY -- no optimizer, no
gradient computation, no model fitting is executed. It is skipped
automatically if the real raw data directory is not present on this
machine (e.g. in CI), so it never blocks the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

RAW_MC06 = Path("/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06")


@unittest.skipUnless(RAW_MC06.is_dir(), "Real MC06 raw data directory not present on this machine")
class ExtendedWslsRealDataPipelineTests(unittest.TestCase):
    def test_extended_wsls_pipeline_on_real_mc06(self) -> None:
        from extended_wsls_v1.pipeline import prepare_registered_extended_wsls_fit_data
        from extended_wsls_v1.registry import load_extended_wsls_model_registry

        registry = load_extended_wsls_model_registry("config/extended_wsls_v1_mc06.json")
        data = prepare_registered_extended_wsls_fit_data(RAW_MC06, registry)

        self.assertGreater(len(data.training_sessions), 0)
        self.assertGreater(len(data.validation_sessions), 0)
        self.assertEqual(data.audit.locked_test_files_opened, 0)
        for session in (*data.training_sessions, *data.validation_sessions):
            self.assertGreater(session.duration_s.numel(), 0)

    def test_extended_wsls_and_pure_wsls_agree_on_session_counts_from_the_same_real_split(self) -> None:
        from extended_wsls_v1.pipeline import prepare_registered_extended_wsls_fit_data
        from extended_wsls_v1.registry import load_extended_wsls_model_registry
        from pure_wsls_v1.pipeline import prepare_registered_pure_wsls_fit_data
        from pure_wsls_v1.registry import load_pure_wsls_model_registry

        extended_registry = load_extended_wsls_model_registry("config/extended_wsls_v1_mc06.json")
        pure_registry = load_pure_wsls_model_registry("config/pure_wsls_v1_mc06.json")
        extended_data = prepare_registered_extended_wsls_fit_data(RAW_MC06, extended_registry)
        pure_data = prepare_registered_pure_wsls_fit_data(RAW_MC06, pure_registry)

        self.assertEqual(len(extended_data.training_sessions), len(pure_data.training_sessions))
        self.assertEqual(len(extended_data.validation_sessions), len(pure_data.validation_sessions))


if __name__ == "__main__":
    unittest.main()
