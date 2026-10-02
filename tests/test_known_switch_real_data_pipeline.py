"""Real-MATLAB-data structural check for the known-switch data pipelines.

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
class KnownSwitchRealDataPipelineTests(unittest.TestCase):
    def test_m1_known_switch_pipeline_on_real_mc06(self) -> None:
        from m1_known_switch_v1.pipeline import prepare_registered_m1_known_switch_fit_data
        from m1_known_switch_v1.registry import load_m1_known_switch_model_registry

        registry = load_m1_known_switch_model_registry("config/m1_known_switch_v1_mc06.json")
        data = prepare_registered_m1_known_switch_fit_data(RAW_MC06, registry)

        self.assertGreater(len(data.training_sessions), 0)
        self.assertGreater(len(data.validation_sessions), 0)
        self.assertEqual(data.audit.locked_test_files_opened, 0)
        for session in (*data.training_sessions, *data.validation_sessions):
            self.assertFalse(bool(session.is_switch[0]))
            self.assertGreater(session.is_switch.numel(), 0)

    def test_m2_known_switch_pipeline_on_real_mc06(self) -> None:
        from m2_known_switch_v1.pipeline import prepare_registered_m2_known_switch_fit_data
        from m2_known_switch_v1.registry import load_m2_known_switch_model_registry

        registry = load_m2_known_switch_model_registry("config/m2_known_switch_v1_mc06.json")
        data = prepare_registered_m2_known_switch_fit_data(RAW_MC06, registry)

        self.assertGreater(len(data.training_sessions), 0)
        self.assertGreater(len(data.validation_sessions), 0)
        self.assertEqual(data.audit.locked_test_files_opened, 0)

    def test_m1_and_m2_agree_on_switch_timing_from_the_same_real_sessions(self) -> None:
        # Both adapters derive is_switch from the same true_block column;
        # for the same real sessions they must agree exactly.
        from m1_known_switch_v1.pipeline import prepare_registered_m1_known_switch_fit_data
        from m1_known_switch_v1.registry import load_m1_known_switch_model_registry
        from m2_known_switch_v1.pipeline import prepare_registered_m2_known_switch_fit_data
        from m2_known_switch_v1.registry import load_m2_known_switch_model_registry

        m1_registry = load_m1_known_switch_model_registry("config/m1_known_switch_v1_mc06.json")
        m2_registry = load_m2_known_switch_model_registry("config/m2_known_switch_v1_mc06.json")
        m1_data = prepare_registered_m1_known_switch_fit_data(RAW_MC06, m1_registry)
        m2_data = prepare_registered_m2_known_switch_fit_data(RAW_MC06, m2_registry)

        m1_by_id = {s.session_id: s for s in (*m1_data.training_sessions, *m1_data.validation_sessions)}
        m2_by_id = {s.session_id: s for s in (*m2_data.training_sessions, *m2_data.validation_sessions)}
        self.assertEqual(set(m1_by_id), set(m2_by_id))
        for session_id in m1_by_id:
            self.assertEqual(
                m1_by_id[session_id].is_switch.tolist(),
                m2_by_id[session_id].is_switch.tolist(),
            )


if __name__ == "__main__":
    unittest.main()
