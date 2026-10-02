"""Real-MATLAB-data structural check for M1_stimulus_kernel_v1's data
pipeline.

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
class StimulusKernelRealDataPipelineTests(unittest.TestCase):
    def test_m1_stimulus_kernel_pipeline_on_real_mc06(self) -> None:
        from m1_stimulus_kernel_v1.pipeline import prepare_registered_m1_stimulus_kernel_fit_data
        from m1_stimulus_kernel_v1.registry import load_m1_stimulus_kernel_model_registry

        registry = load_m1_stimulus_kernel_model_registry("config/m1_stimulus_kernel_v1_mc06.json")
        data = prepare_registered_m1_stimulus_kernel_fit_data(RAW_MC06, registry)

        self.assertGreater(len(data.training_sessions), 0)
        self.assertGreater(len(data.validation_sessions), 0)
        self.assertEqual(data.audit.locked_test_files_opened, 0)
        for session in (*data.training_sessions, *data.validation_sessions):
            self.assertGreater(session.duration_s.numel(), 0)

    def test_m1_stimulus_kernel_and_m1_agree_on_session_counts_from_the_same_real_split(self) -> None:
        # Both adapters delegate discovery/inclusion/leak-safety to the same
        # M0 registered pipeline and build sessions via the identical M1
        # frame adapter, so for the same subject and split they must produce
        # the same train/validation session counts.
        from m0_sensory_v1.model.m1_session import m1_session_trials_from_frame
        from m0_sensory_v1.pipeline import prepare_registered_fit_data
        from m1_stimulus_kernel_v1.pipeline import prepare_registered_m1_stimulus_kernel_fit_data
        from m1_stimulus_kernel_v1.registry import load_m1_stimulus_kernel_model_registry

        registry = load_m1_stimulus_kernel_model_registry("config/m1_stimulus_kernel_v1_mc06.json")
        stimulus_kernel_data = prepare_registered_m1_stimulus_kernel_fit_data(RAW_MC06, registry)
        base = prepare_registered_fit_data(RAW_MC06, registry.source_m0_registry)

        expected_train = 0
        expected_validation = 0
        for _, frame in base.canonical_trials.groupby("session_id", sort=False, observed=False):
            roles = frame["split_role"].dropna().unique().tolist()
            if len(roles) != 1:
                continue
            if roles[0] == "train":
                expected_train += 1
            elif roles[0] == "validation":
                expected_validation += 1
        self.assertEqual(len(stimulus_kernel_data.training_sessions), expected_train)
        self.assertEqual(len(stimulus_kernel_data.validation_sessions), expected_validation)


if __name__ == "__main__":
    unittest.main()
