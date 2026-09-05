"""Cross-conformance test: M2_dynamic_block_length_v1 at delta0=log(4) must
equal M1_dynamic_block_length_v1 exactly (both use the fixed effective
prototypes (0.5,0.2,0.8) at that point).

This is the strongest guard against the two independently-written
``session.py`` loops drifting apart -- they share no runtime code, so if this
test passes it is because their per-trial math genuinely agrees, not because
one delegates to the other.
"""

from __future__ import annotations

import math
import unittest

import torch

from m0_sensory_v1.model.m1_session import M1SessionTrials
from m1_dynamic_block_length_v1.parameters import M1DynamicBlockLengthParameters
from m1_dynamic_block_length_v1.session import evaluate_m1_dynamic_block_length_session
from m2_dynamic_block_length_v1.parameters import M2DynamicBlockLengthParameters
from m2_dynamic_block_length_v1.session import evaluate_m2_dynamic_block_length_session

DTYPE = torch.float64


def _session(n, seed):
    g = torch.Generator().manual_seed(seed)
    duration = torch.where(
        torch.rand(n, generator=g) < 0.5,
        0.2 + 0.6 * torch.rand(n, generator=g),
        1.7 + 0.6 * torch.rand(n, generator=g),
    ).to(DTYPE)
    category = (torch.rand(n, generator=g) < 0.5).to(DTYPE)
    choice = (torch.rand(n, generator=g) < 0.5).to(DTYPE)
    feedback = torch.rand(n, generator=g) < 0.9
    return M1SessionTrials(
        mouse_id="mouse", session_id=f"sess{seed}", duration_s=duration,
        choice_long=choice, category_long=category, feedback_valid=feedback, split_role="train",
    )


class CrossConformanceTests(unittest.TestCase):
    def test_m2_dynamic_at_log4_equals_m1_dynamic_across_random_sessions_and_bh(self):
        from m0_sensory_v1.data.task_moments import TaskMoments

        moments = TaskMoments(
            mu_short_s=0.5, mu_long_s=2.0, var_short_s2=0.01, var_long_s2=0.01,
            n_short=100, n_long=100, min_short_s=0.2, max_short_s=0.8,
            min_long_s=1.7, max_long_s=2.3, variance_ddof=0, source="synthetic",
        )
        w = torch.tensor(0.03, dtype=DTYPE)
        s0 = torch.tensor(0.5, dtype=DTYPE)
        ell_s = torch.tensor(0.03, dtype=DTYPE)
        ell_l = torch.tensor(0.03, dtype=DTYPE)
        kappa = torch.tensor(1.0, dtype=DTYPE)
        ah = torch.tensor(-1.2, dtype=DTYPE)
        delta0 = torch.tensor(math.log(4.0), dtype=DTYPE)

        for seed in range(5):
            for bh_value in (0.01, 0.05, 0.15, 0.4):
                session = _session(n=80, seed=seed)  # >65 to exercise the tail
                bh = torch.tensor(bh_value, dtype=DTYPE)
                m2d_params = M2DynamicBlockLengthParameters(
                    w=w, s0=s0, ell_short=ell_s, ell_long=ell_l, kappa=kappa, ah=ah, delta0=delta0, bh=bh,
                )
                m1d_params = M1DynamicBlockLengthParameters(
                    w=w, s0=s0, ell_short=ell_s, ell_long=ell_l, kappa=kappa, ah=ah, bh=bh,
                )
                eval_m2d = evaluate_m2_dynamic_block_length_session(session, m2d_params, moments)
                eval_m1d = evaluate_m1_dynamic_block_length_session(session, m1d_params, moments)
                for name in (
                    "probability_long", "prior_long", "emission_normalizer",
                    "expected_run_length_pre", "expected_hazard_pre", "tail_probability_pre",
                    "pre_choice_block_probability", "post_feedback_block_probability",
                ):
                    left = getattr(eval_m2d.predictions, name)
                    right = getattr(eval_m1d.predictions, name)
                    torch.testing.assert_close(
                        left, right, rtol=0, atol=1e-13, equal_nan=True,
                        msg=f"seed={seed} bh={bh_value} field={name}",
                    )
                torch.testing.assert_close(
                    eval_m2d.negative_log_likelihood, eval_m1d.negative_log_likelihood,
                    rtol=0, atol=1e-12, msg=f"seed={seed} bh={bh_value} NLL",
                )


if __name__ == "__main__":
    unittest.main()
