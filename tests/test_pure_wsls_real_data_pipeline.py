"""Real-MATLAB-data structural check for Pure_WSLS_v1's data pipeline.

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
class PureWslsRealDataPipelineTests(unittest.TestCase):
    def test_pure_wsls_pipeline_on_real_mc06(self) -> None:
        from pure_wsls_v1.pipeline import prepare_registered_pure_wsls_fit_data
        from pure_wsls_v1.registry import load_pure_wsls_model_registry

        registry = load_pure_wsls_model_registry("config/pure_wsls_v1_mc06.json")
        data = prepare_registered_pure_wsls_fit_data(RAW_MC06, registry)

        self.assertGreater(len(data.training_sessions), 0)
        self.assertGreater(len(data.validation_sessions), 0)
        self.assertEqual(data.audit.locked_test_files_opened, 0)
        for session in (*data.training_sessions, *data.validation_sessions):
            self.assertGreater(session.duration_s.numel(), 0)

    def test_pure_wsls_and_m1_stimulus_kernel_agree_on_session_counts_from_the_same_real_split(self) -> None:
        # Both adapters delegate discovery/inclusion/leak-safety to the same
        # M0 registered pipeline and build sessions via the identical M1
        # frame adapter, so for the same subject and split they must produce
        # the same train/validation session counts.
        from m1_stimulus_kernel_v1.pipeline import prepare_registered_m1_stimulus_kernel_fit_data
        from m1_stimulus_kernel_v1.registry import load_m1_stimulus_kernel_model_registry
        from pure_wsls_v1.pipeline import prepare_registered_pure_wsls_fit_data
        from pure_wsls_v1.registry import load_pure_wsls_model_registry

        pure_registry = load_pure_wsls_model_registry("config/pure_wsls_v1_mc06.json")
        kernel_registry = load_m1_stimulus_kernel_model_registry("config/m1_stimulus_kernel_v1_mc06.json")
        pure_data = prepare_registered_pure_wsls_fit_data(RAW_MC06, pure_registry)
        kernel_data = prepare_registered_m1_stimulus_kernel_fit_data(RAW_MC06, kernel_registry)

        self.assertEqual(len(pure_data.training_sessions), len(kernel_data.training_sessions))
        self.assertEqual(len(pure_data.validation_sessions), len(kernel_data.validation_sessions))

    def test_pure_wsls_never_uses_duration_in_prediction(self) -> None:
        """Perturbing duration must never change any prediction (structural check)."""

        import torch

        from pure_wsls_v1.parameters import decode_pure_wsls_raw_parameters
        from pure_wsls_v1.pipeline import prepare_registered_pure_wsls_fit_data
        from pure_wsls_v1.registry import load_pure_wsls_model_registry
        from pure_wsls_v1.session import evaluate_pure_wsls_session
        from m0_sensory_v1.model.m1_session import M1SessionTrials

        registry = load_pure_wsls_model_registry("config/pure_wsls_v1_mc06.json")
        data = prepare_registered_pure_wsls_fit_data(RAW_MC06, registry)
        session = data.training_sessions[0]
        parameters = decode_pure_wsls_raw_parameters(
            registry.initializations[0].raw_tensor(lapse_max=registry.lapse_max)
        )
        baseline = evaluate_pure_wsls_session(session, parameters, registry.task_moments)
        perturbed_session = M1SessionTrials(
            mouse_id=session.mouse_id,
            session_id=session.session_id,
            duration_s=torch.rand_like(session.duration_s) * 1000.0,
            choice_long=session.choice_long,
            category_long=session.category_long,
            feedback_valid=session.feedback_valid,
            split_role=session.split_role,
        )
        perturbed = evaluate_pure_wsls_session(perturbed_session, parameters, registry.task_moments)
        torch.testing.assert_close(
            baseline.predictions.prior_long, perturbed.predictions.prior_long
        )
        torch.testing.assert_close(
            baseline.predictions.probability_long, perturbed.predictions.probability_long,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
