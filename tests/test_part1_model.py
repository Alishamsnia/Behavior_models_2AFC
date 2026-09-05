"""Mathematical and autograd tests for the M0 Part I implementation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.data import TaskMoments  # noqa: E402
from m0_sensory_v1.model import (  # noqa: E402
    DecodedParameters,
    LogOddsCoefficients,
    bayesian_choice_probability,
    category_variances,
    decode_raw_parameters,
    encode_constrained_parameters,
    fixed_neutral_prior,
    lapse_mixture,
    log_odds_coefficients,
    positive_decision_threshold,
    predict_m0_trials,
)


DTYPE = torch.float64
REFERENCE_MOMENTS = TaskMoments(
    mu_short_s=0.5,
    mu_long_s=2.0,
    var_short_s2=0.0296,
    var_long_s2=0.0296,
    n_short=1,
    n_long=1,
    min_short_s=0.5,
    max_short_s=0.5,
    min_long_s=2.0,
    max_long_s=2.0,
    source="mathematical reference test",
)


def _reference_raw(*, requires_grad: bool = False) -> torch.Tensor:
    constrained = torch.tensor([0.0271, 0.555, 0.03, 0.03, 1.0], dtype=DTYPE)
    return encode_constrained_parameters(constrained).detach().requires_grad_(requires_grad)


class PartOneModelTests(unittest.TestCase):
    def test_parameter_round_trip_and_constraints(self) -> None:
        constrained = torch.tensor(
            [
                [0.0271, 0.555, 0.030, 0.030, 1.0],
                [0.0050, 0.700, 0.135, 0.015, 0.5],
                [0.2000, 0.350, 0.098, 0.002, 2.0],
                [0.0500, 0.500, 0.800, 0.500, 0.75],
            ],
            dtype=DTYPE,
        )
        decoded = decode_raw_parameters(encode_constrained_parameters(constrained))
        recovered = decoded.as_tensor()
        torch.testing.assert_close(recovered, constrained, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.all(decoded.w > 0.0))
        self.assertTrue(torch.all(decoded.s0 > 0.0))
        self.assertTrue(torch.all((decoded.ell_short > 0.0) & (decoded.ell_short < 1.0)))
        self.assertTrue(torch.all((decoded.ell_long > 0.0) & (decoded.ell_long < 1.0)))
        self.assertTrue(torch.all(decoded.kappa > 0.0))
        with self.assertRaises(ValueError):
            decode_raw_parameters(torch.zeros(5, dtype=DTYPE), lapse_max=1.01)

    def test_reference_category_variances_coefficients_and_roots(self) -> None:
        parameters = decode_raw_parameters(_reference_raw())
        variances = category_variances(parameters, REFERENCE_MOMENTS)
        self.assertAlmostEqual(variances.short.item(), 0.344400, places=12)
        self.assertAlmostEqual(variances.long.item(), 0.446025, places=12)

        expected_roots = {0.20: 1.5848899681, 0.50: 1.2352680688, 0.80: 0.8632498719}
        for prior_value, expected_positive_root in expected_roots.items():
            prior = torch.tensor(prior_value, dtype=DTYPE)
            coefficients = log_odds_coefficients(
                variances,
                REFERENCE_MOMENTS,
                prior,
                parameters.log_kappa,
            )
            self.assertAlmostEqual(coefficients.a2.item(), 0.3307868362, places=9)
            self.assertAlmostEqual(coefficients.a1.item(), 3.0322533522, places=9)
            threshold = positive_decision_threshold(coefficients)
            self.assertAlmostEqual(threshold.item(), expected_positive_root, places=9)

    def test_positive_threshold_linear_limit_and_rejects_two_positive_roots(
        self,
    ) -> None:
        linear = LogOddsCoefficients(
            a2=torch.tensor(0.0, dtype=DTYPE),
            a1=torch.tensor(2.0, dtype=DTYPE),
            a0=torch.tensor(-3.0, dtype=DTYPE),
        )
        self.assertAlmostEqual(positive_decision_threshold(linear).item(), 1.5)

        positive_double_root = LogOddsCoefficients(
            a2=torch.tensor(1.0, dtype=DTYPE),
            a1=torch.tensor(-2.0, dtype=DTYPE),
            a0=torch.tensor(1.0, dtype=DTYPE),
        )
        self.assertAlmostEqual(
            positive_decision_threshold(positive_double_root).item(),
            1.0,
        )

        two_positive = LogOddsCoefficients(
            a2=torch.tensor(1.0, dtype=DTYPE),
            a1=torch.tensor(-3.0, dtype=DTYPE),
            a0=torch.tensor(2.0, dtype=DTYPE),
        )
        with self.assertRaisesRegex(ValueError, "exactly one positive"):
            positive_decision_threshold(two_positive)

    def test_fixed_neutral_prior_is_exactly_half(self) -> None:
        prior = fixed_neutral_prior(torch.empty(17, dtype=DTYPE))
        self.assertTrue(torch.equal(prior, torch.full((17,), 0.5, dtype=DTYPE)))
        self.assertTrue(torch.equal(torch.logit(prior), torch.zeros_like(prior)))

    def test_choice_probability_matches_normal_cdf(self) -> None:
        parameters = decode_raw_parameters(_reference_raw())
        durations = torch.tensor([0.2, 0.8, 1.7, 2.3], dtype=DTYPE)
        result = bayesian_choice_probability(
            durations,
            parameters,
            REFERENCE_MOMENTS,
            fixed_neutral_prior(durations),
        )
        expected = torch.distributions.Normal(0.0, 1.0).cdf(
            (durations - result.positive_threshold_s) / result.sensory_sd_s
        )
        torch.testing.assert_close(result.probability_long, expected, rtol=1e-13, atol=1e-13)
        self.assertTrue(torch.all(result.positive_threshold_s > 0.0))

    def test_kappa_is_neutral_at_one_and_has_the_registered_bias_direction(self) -> None:
        durations = torch.tensor([0.5, 1.0, 2.0], dtype=DTYPE)
        prior = fixed_neutral_prior(durations)

        def evaluate(kappa: float):
            constrained = torch.tensor(
                [0.0271, 0.555, 0.03, 0.03, kappa], dtype=DTYPE
            )
            parameters = decode_raw_parameters(
                encode_constrained_parameters(constrained)
            )
            return bayesian_choice_probability(
                durations, parameters, REFERENCE_MOMENTS, prior
            )

        favors_long = evaluate(0.5)
        neutral = evaluate(1.0)
        favors_short = evaluate(2.0)
        self.assertTrue(
            torch.all(
                favors_long.probability_long
                > neutral.probability_long
            )
        )
        self.assertTrue(
            torch.all(
                neutral.probability_long
                > favors_short.probability_long
            )
        )
        self.assertTrue(
            torch.all(
                favors_long.positive_threshold_s
                < neutral.positive_threshold_s
            )
        )
        self.assertTrue(
            torch.all(
                neutral.positive_threshold_s
                < favors_short.positive_threshold_s
            )
        )

    def test_category_specific_random_choice_lapse(self) -> None:
        parameters = decode_raw_parameters(_reference_raw())
        category = torch.tensor([0.0, 1.0], dtype=DTYPE)
        result = lapse_mixture(
            torch.tensor([0.0, 1.0], dtype=DTYPE), category, parameters
        )
        expected = torch.stack(
            (0.5 * parameters.ell_short, 1.0 - 0.5 * parameters.ell_long)
        )
        torch.testing.assert_close(result, expected)

        all_lapse = DecodedParameters(
            w=parameters.w,
            s0=parameters.s0,
            ell_short=torch.ones_like(parameters.ell_short),
            ell_long=torch.ones_like(parameters.ell_long),
            kappa=parameters.kappa,
        )
        torch.testing.assert_close(
            lapse_mixture(torch.tensor([0.1, 0.9], dtype=DTYPE), category, all_lapse),
            torch.full((2,), 0.5, dtype=DTYPE),
        )

    def test_masked_prediction_skips_nondecisions_before_computation(self) -> None:
        raw = _reference_raw(requires_grad=True)
        duration = torch.tensor([0.5, float("nan"), 2.0, 1.2], dtype=DTYPE)
        category = torch.tensor([0.0, float("nan"), 1.0, float("nan")], dtype=DTYPE)
        choice = torch.tensor([0.0, float("nan"), 1.0, float("nan")], dtype=DTYPE)
        result = predict_m0_trials(duration, category, choice, raw, REFERENCE_MOMENTS)
        self.assertEqual(result.valid_choice_mask.tolist(), [True, False, True, False])
        self.assertEqual(
            torch.isfinite(result.probability_long).tolist(),
            [True, False, True, False],
        )

        valid_probability = result.probability_long[result.valid_choice_mask]
        valid_choice = choice[result.valid_choice_mask]
        loss = -(
            valid_choice * torch.log(valid_probability)
            + (1.0 - valid_choice) * torch.log1p(-valid_probability)
        ).mean()
        gradient = torch.autograd.grad(loss, raw)[0]
        self.assertTrue(torch.all(torch.isfinite(gradient)))

        # Values on rows with missing choices are irrelevant because those rows
        # are indexed out before the scientific forward graph.
        alternative_duration = torch.tensor([0.5, 999.0, 2.0, 0.001], dtype=DTYPE)
        alternative = predict_m0_trials(
            alternative_duration,
            category,
            choice,
            raw.detach(),
            REFERENCE_MOMENTS,
        )
        torch.testing.assert_close(
            alternative.probability_long[alternative.valid_choice_mask],
            valid_probability.detach(),
        )

    def test_choice_values_do_not_change_predictions(self) -> None:
        duration = torch.tensor([0.5, 1.0, 2.0], dtype=DTYPE)
        category = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
        first_choice = torch.tensor([0.0, float("nan"), 1.0], dtype=DTYPE)
        second_choice = torch.tensor([1.0, float("nan"), 0.0], dtype=DTYPE)
        first = predict_m0_trials(duration, category, first_choice, _reference_raw(), REFERENCE_MOMENTS)
        second = predict_m0_trials(duration, category, second_choice, _reference_raw(), REFERENCE_MOMENTS)
        self.assertTrue(torch.equal(first.valid_choice_mask, second.valid_choice_mask))
        torch.testing.assert_close(
            first.probability_long[first.valid_choice_mask],
            second.probability_long[second.valid_choice_mask],
        )

    def test_masked_prediction_autograd_matches_finite_difference(self) -> None:
        duration = torch.tensor([0.35, 0.65, float("nan"), 1.85, 2.15], dtype=DTYPE)
        category = torch.tensor([0.0, 0.0, float("nan"), 1.0, 1.0], dtype=DTYPE)
        choice = torch.tensor([0.0, 1.0, float("nan"), 0.0, 1.0], dtype=DTYPE)

        def objective(raw: torch.Tensor) -> torch.Tensor:
            result = predict_m0_trials(duration, category, choice, raw, REFERENCE_MOMENTS)
            probability = result.probability_long[result.valid_choice_mask]
            observed = choice[result.valid_choice_mask]
            return -(
                observed * torch.log(probability)
                + (1.0 - observed) * torch.log1p(-probability)
            ).mean()

        raw = _reference_raw(requires_grad=True)
        analytical = torch.autograd.grad(objective(raw), raw)[0]
        step = 1e-6
        numerical = torch.empty_like(raw)
        for index in range(raw.numel()):
            plus = raw.detach().clone()
            minus = raw.detach().clone()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (objective(plus) - objective(minus)) / (2.0 * step)

        torch.testing.assert_close(analytical, numerical, rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
