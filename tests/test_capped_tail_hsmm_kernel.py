"""Independent verification of the shared capped-tail HSMM kernel
(``m0_sensory_v1.model.capped_tail_hsmm``).

This kernel is reused by M1_dynamic_block_length_v1 and
M2_dynamic_block_length_v1; it is NOT imported from ``m4_hsmm_delta_v1``, and
this file cross-checks it against that package's own kernel only as one more
piece of regression evidence, never as the primary proof of correctness.
"""

from __future__ import annotations

import unittest

import torch

from m0_sensory_v1.model.capped_tail_hsmm import (
    block_marginal,
    capped_tail_hazard,
    capped_tail_hsmm_transition,
    feedback_update,
    initialize_capped_tail_state,
    run_length_marginal,
)
from m0_sensory_v1.model.m1_session import constant_hazard_transition_matrix

DTYPE = torch.float64


def _reference_transition(state: torch.Tensor, hazard: torch.Tensor, omega: float) -> torch.Tensor:
    """Independent, explicit source-to-destination enumeration (not vectorized)."""

    rmax = state.shape[1]
    result = torch.zeros_like(state)
    for block in range(3):
        for r_index in range(rmax):
            mass = float(state[block, r_index])
            if mass == 0.0:
                continue
            r = r_index + 1
            h_biased = float(hazard[r_index])
            h = h_biased / 2.0 if block == 0 else h_biased
            survive = mass * (1.0 - h)
            switch = mass * h
            destination_r_index = min(r_index + 1, rmax - 1)
            result[block, destination_r_index] += survive
            if block == 0:
                result[1, 0] += switch * (1.0 - omega)
                result[2, 0] += switch * omega
            elif block == 1:
                result[2, 0] += switch
            else:
                result[1, 0] += switch
    return result


class CappedTailHSMMKernelTests(unittest.TestCase):
    def test_hazard_matches_closed_form(self):
        ah, bh = torch.tensor(-1.3, dtype=DTYPE), torch.tensor(0.07, dtype=DTYPE)
        run = torch.arange(1, 66, dtype=DTYPE)
        hazard = capped_tail_hazard(ah, bh, run, center=23)
        expected = torch.sigmoid(ah + bh * (run - 23.0))
        torch.testing.assert_close(hazard, expected, rtol=0, atol=1e-15)

    def test_reference_oracle_matches_vectorized_implementation(self):
        torch.manual_seed(0)
        for rmax in (2, 3, 65):
            for _ in range(5):
                raw = torch.rand(3, rmax, dtype=DTYPE)
                state = raw / raw.sum()
                hazard = torch.rand(rmax, dtype=DTYPE) * 0.9 + 0.02
                omega = float(torch.rand(()).item())
                expected = _reference_transition(state, hazard, omega)
                observed = capped_tail_hsmm_transition(state, hazard, first_long_probability=omega)
                torch.testing.assert_close(observed, expected, rtol=0, atol=1e-12)

    def test_basis_states_every_block_and_landmark_run_length(self):
        for rmax in (2, 3, 65):
            hazard = torch.linspace(0.05, 0.6, rmax, dtype=DTYPE)
            landmarks = sorted({0, rmax // 2, rmax - 2, rmax - 1} & set(range(rmax)))
            for block in range(3):
                for r_index in landmarks:
                    state = torch.zeros(3, rmax, dtype=DTYPE)
                    state[block, r_index] = 1.0
                    expected = _reference_transition(state, hazard, 0.5)
                    observed = capped_tail_hsmm_transition(state, hazard, first_long_probability=0.5)
                    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-13)

    def test_randomized_fuzz_mass_conservation_and_nonnegativity(self):
        torch.manual_seed(1)
        for rmax in (2, 3, 65):
            for omega in (0.0, 0.37, 1.0):
                for _ in range(10):
                    raw = torch.rand(3, rmax, dtype=DTYPE)
                    state = raw / raw.sum()
                    hazard = torch.rand(rmax, dtype=DTYPE)
                    out = capped_tail_hsmm_transition(state, hazard, first_long_probability=omega)
                    self.assertTrue(bool(torch.isclose(out.sum(), torch.tensor(1.0, dtype=DTYPE), atol=1e-12)))
                    self.assertTrue(bool((out >= -1e-14).all()))

    def test_constant_hazard_identity_matches_m1_transition_matrix(self):
        torch.manual_seed(2)
        raw = torch.rand(3, 65, dtype=DTYPE)
        state = raw / raw.sum()
        h_scalar = torch.tensor(0.13, dtype=DTYPE)
        h_const = h_scalar.expand(65).clone()
        out = capped_tail_hsmm_transition(state, h_const, first_long_probability=0.37)
        matrix = constant_hazard_transition_matrix(h_scalar, first_long_probability=0.37)
        expected_block = block_marginal(state) @ matrix
        torch.testing.assert_close(block_marginal(out), expected_block, rtol=0, atol=1e-13)

    def test_feedback_update_invariants(self):
        torch.manual_seed(3)
        raw = torch.rand(3, 65, dtype=DTYPE)
        state = raw / raw.sum()
        emission = torch.tensor([0.5, 0.3, 0.9], dtype=DTYPE)
        post, normalizer = feedback_update(state, emission)
        torch.testing.assert_close(normalizer, torch.dot(block_marginal(state), emission), rtol=0, atol=1e-13)
        for block in range(3):
            before = state[block] / state[block].sum()
            after = post[block] / post[block].sum()
            torch.testing.assert_close(after, before, rtol=0, atol=1e-12)

    def test_reachable_support_and_no_transition_after_final_trial(self):
        reference = torch.tensor(0.0, dtype=DTYPE)
        state = initialize_capped_tail_state(reference, 65)
        hazard = torch.sigmoid(torch.linspace(-3, 1, 65, dtype=DTYPE))
        for step in range(1, 10):
            state = capped_tail_hsmm_transition(state, hazard, first_long_probability=0.5)
            reachable = min(step + 1, 65)
            self.assertLessEqual(float(state[:, reachable:].sum()), 1e-12)

    def test_endpoint_h0_and_h1(self):
        reference = torch.tensor(0.0, dtype=DTYPE)
        state = initialize_capped_tail_state(reference, 65)
        zero = torch.zeros(65, dtype=DTYPE)
        out0 = capped_tail_hsmm_transition(state, zero, first_long_probability=0.5)
        expected0 = torch.zeros(3, 65, dtype=DTYPE)
        expected0[0, 1] = 1.0
        torch.testing.assert_close(out0, expected0, rtol=0, atol=1e-14)

        one = torch.ones(65, dtype=DTYPE)
        out1 = capped_tail_hsmm_transition(state, one, first_long_probability=0.5)
        expected1 = torch.zeros(3, 65, dtype=DTYPE)
        expected1[0, 1] = 0.5
        expected1[1, 0] = 0.25
        expected1[2, 0] = 0.25
        torch.testing.assert_close(out1, expected1, rtol=0, atol=1e-14)

    def test_gradients_and_dtype_device_preservation(self):
        ah = torch.tensor(-1.0, dtype=DTYPE, requires_grad=True)
        bh = torch.tensor(0.05, dtype=DTYPE, requires_grad=True)
        run = torch.arange(1, 66, dtype=DTYPE)
        hazard = capped_tail_hazard(ah, bh, run, center=23)
        state = initialize_capped_tail_state(ah, 65)
        out = capped_tail_hsmm_transition(state, hazard, first_long_probability=0.5)
        self.assertEqual(out.dtype, DTYPE)
        out.sum().backward()
        self.assertTrue(bool(torch.isfinite(ah.grad)))
        self.assertTrue(bool(torch.isfinite(bh.grad)))

    def test_invalid_inputs_raise(self):
        state = initialize_capped_tail_state(torch.tensor(0.0, dtype=DTYPE), 65)
        hazard = torch.full((65,), 0.1, dtype=DTYPE)
        with self.assertRaises(ValueError):
            capped_tail_hsmm_transition(state, hazard, first_long_probability=1.5)
        with self.assertRaises(ValueError):
            capped_tail_hsmm_transition(state, torch.full((64,), 0.1, dtype=DTYPE))
        with self.assertRaises(ValueError):
            capped_tail_hsmm_transition(state, torch.full((65,), float("nan"), dtype=DTYPE))
        with self.assertRaises(TypeError):
            capped_tail_hsmm_transition(state.numpy(), hazard)
        with self.assertRaises(ValueError):
            feedback_update(state, torch.tensor([0.5, 0.5], dtype=DTYPE))
        with self.assertRaises(ValueError):
            initialize_capped_tail_state(torch.tensor(0.0, dtype=DTYPE), 1)

    def test_cross_check_against_m4_hsmm_delta_v1_kernel(self):
        """Test-only regression evidence: this file is the ONLY place any of
        these two new models' test suites may import m4_hsmm_delta_v1."""

        from m4_hsmm_delta_v1.prior_state import capped_tail_hsmm_transition as m4_transition

        torch.manual_seed(4)
        for _ in range(20):
            raw = torch.rand(3, 65, dtype=DTYPE)
            state = raw / raw.sum()
            hazard = torch.rand(65, dtype=DTYPE) * 0.9 + 0.02
            omega = float(torch.rand(()).item())
            mine = capped_tail_hsmm_transition(state, hazard, first_long_probability=omega)
            theirs = m4_transition(state, hazard, first_long_probability=omega)
            torch.testing.assert_close(mine, theirs, rtol=0, atol=1e-13)


class ImportGraphIndependenceTests(unittest.TestCase):
    """Assert no production module under the two new models imports
    m4_hsmm_delta_v1 (this test file itself is exempt -- it is test code)."""

    def test_no_production_module_imports_m4_hsmm_delta_v1(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src"
        offenders = []
        for package in ("m1_dynamic_block_length_v1", "m2_dynamic_block_length_v1"):
            for path in (root / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module] if node.module else []
                    else:
                        continue
                    if any(name and name.startswith("m4_hsmm_delta_v1") for name in names):
                        offenders.append(str(path))
        self.assertEqual(offenders, [], f"Production files importing m4_hsmm_delta_v1: {offenders}")


if __name__ == "__main__":
    unittest.main()
