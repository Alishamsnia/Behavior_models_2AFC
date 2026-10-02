from __future__ import annotations

import unittest

import numpy as np

from post_analysis.expanded_model_comparison.block_bootstrap_figures import (
    EQUIVALENCE_MARGIN,
    EQUIVALENCE_BLOCK_LENGTH,
    _document_pair_results,
    _equivalence_rows,
    _holm_adjust,
    _hybrid_pair_results,
    _load_scores,
    _moving_block_indices,
    _moving_block_weights,
)


class BlockBootstrapFigureTests(unittest.TestCase):
    def test_moving_blocks_are_deterministic_consecutive_and_trimmed(self):
        first = _moving_block_indices(7, 3, 20)
        second = _moving_block_indices(7, 3, 20)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (20, 7))
        self.assertTrue(((first >= 0) & (first < 7)).all())
        np.testing.assert_array_equal(np.diff(first, axis=1)[:, [0, 1, 3, 4]], 1)

    def test_holm_matches_hand_calculation(self):
        adjusted = _holm_adjust(np.asarray([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_exact_moving_block_weights_include_trimmed_draw(self):
        weights = _moving_block_weights(7, 3)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        blocks = [np.arange(start, start + 3) for start in range(5)]
        enumerated = np.mean([
            np.bincount(np.concatenate((first, second, third[:1])), minlength=7) / 7
            for first in blocks for second in blocks for third in blocks
        ], axis=0)
        np.testing.assert_allclose(weights, enumerated)

    def test_document_sign_positive_favors_first_model(self):
        matrix = np.asarray([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8]])
        rows, _ = _document_pair_results(matrix, ("a", "b"), 2, 2_000)
        self.assertAlmostEqual(rows[0].observed, 0.25)
        self.assertGreater(rows[0].observed, 0)
        self.assertLessEqual(rows[0].ci95_low, rows[0].ci95_high)
        self.assertEqual(rows[0].tail_lower + rows[0].tail_upper, 2_000 + rows[0].tail_zero)

    def test_hybrid_basic_interval_is_centered_on_observed_effect(self):
        matrix = np.asarray([
            [0.10, 0.14], [0.22, 0.25], [0.16, 0.21],
            [0.35, 0.39], [0.28, 0.30], [0.19, 0.24],
        ])
        rows, _ = _hybrid_pair_results(matrix, ("a", "b"), 2, 2_000)
        sampled = _hybrid_pair_results(matrix, ("a", "b"), 2, 2_000)[1]
        boot = sampled[:, 1] - sampled[:, 0]
        error = boot - boot.mean()
        expected_p = (np.count_nonzero(np.abs(error) >= abs(rows[0].observed)) + 1) / 2_001
        midpoint = (rows[0].ci95_low + rows[0].ci95_high) / 2
        self.assertAlmostEqual(midpoint, rows[0].observed, delta=0.01)
        self.assertAlmostEqual(rows[0].raw_p, expected_p)

    def test_constant_nonzero_difference_is_flagged_as_unavailable(self):
        matrix = np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        document, _ = _document_pair_results(matrix, ("a", "b"), 2, 2_000)
        hybrid, _ = _hybrid_pair_results(matrix, ("a", "b"), 2, 2_000)
        self.assertTrue(np.isnan(document[0].raw_p))
        self.assertTrue(np.isnan(hybrid[0].raw_p))

    def test_audited_mc08_inputs_and_tost_family(self):
        scores = _load_scores("MC08")
        self.assertEqual(scores.matrix.shape, (30, 13))
        np.testing.assert_array_equal(scores.test_ranks, np.arange(13, 43))
        leader = scores.model_ids[int(np.argmin(scores.matrix.mean(axis=0)))]
        _, sampled = _hybrid_pair_results(
            scores.matrix, scores.model_ids, EQUIVALENCE_BLOCK_LENGTH, 2_000,
        )
        rows = _equivalence_rows(scores, sampled, leader, 2_000)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["ci90_low"] <= row["ci90_high"] for row in rows))
        self.assertTrue(all(np.isfinite(row["holm_p"]) for row in rows))
        self.assertEqual(EQUIVALENCE_MARGIN, 0.005)
        self.assertEqual(EQUIVALENCE_BLOCK_LENGTH, 9)


if __name__ == "__main__":
    unittest.main()
