"""Scientific invariants of the final sliding-window clustering."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from post_analysis.session_clustering.features import normalize, matrix, transform_parameter
from post_analysis.session_clustering.cluster import mouse_checks, block_sample, fit_solutions, driver_manifest, BLOCK


class FeatureTests(unittest.TestCase):
    def test_mouse_location_and_scale_do_not_change_features(self):
        rng = np.random.default_rng(7)
        subjects = np.repeat(['A', 'B', 'C'], 30)
        x = rng.normal(size=(90, 5))
        shifted = x.copy()
        for i, subject in enumerate(['A', 'B', 'C']):
            shifted[subjects == subject] = x[subjects == subject] * (i+1) + 100*i
        np.testing.assert_allclose(normalize(x, subjects), normalize(shifted, subjects), atol=1e-12)

    def test_oracles_excluded_and_blocks_balanced(self):
        rng = np.random.default_rng(1)
        values = pd.DataFrame(rng.normal(size=(90, 5)), columns=list('abcde'))
        meta = pd.DataFrame(dict(feature=list('abcde'), block=['sensory']*3+['history']*2,
                                 model=['m0','m0','m1','m1','oracle'], oracle=[False]*4+[True]))
        x, used = matrix(values, meta, np.repeat(['A', 'B', 'C'], 30))
        self.assertEqual(x.shape, (90, 4))
        self.assertNotIn('oracle', used.model.tolist())
        for _, group in used.groupby('block'):
            self.assertAlmostEqual(float(np.sum(group.weight**2)), 1.)

    def test_invalid_parameters_fail_instead_of_being_clipped(self):
        for name, values in [('w', [-1]), ('s0', [0]), ('ell_short', [1.1]), ('ah', [np.nan])]:
            with self.assertRaises(ValueError):
                transform_parameter(name, values)

    def test_already_logit_parameters_not_transformed_twice(self):
        for name in ['a', 'ah', 'eta_logit', 'delta0', 'bh']:
            np.testing.assert_equal(transform_parameter(name, [-2, 0, 2]), [-2, 0, 2])
        self.assertTrue(np.isfinite(transform_parameter('ell_short', [0, 1])).all())


class MouseGateTests(unittest.TestCase):
    def test_clustering_the_mice_fails(self):
        subjects = np.repeat(['A', 'B', 'C'], 30)
        labels = np.repeat([1, 2, 3], 30)
        self.assertFalse(mouse_checks(labels, subjects)['mouse_gate'])

    def test_missing_mouse_in_cluster_fails(self):
        subjects = np.repeat(['A', 'B', 'C'], 30)
        labels = np.tile(np.repeat([1, 2], 15), 3)
        labels[:30] = 1
        self.assertFalse(mouse_checks(labels, subjects)['mouse_gate'])

    def test_balanced_groups_pass_mouse_gate(self):
        subjects = np.repeat(['A', 'B', 'C'], 30)
        labels = np.tile(np.repeat([1, 2], 15), 3)
        self.assertTrue(mouse_checks(labels, subjects)['mouse_gate'])

    def test_both_requested_solutions_render_even_when_diagnostics_fail(self):
        rng = np.random.default_rng(2)
        values = pd.DataFrame(rng.normal(size=(90, 2)), columns=['a', 'b'])
        meta = pd.DataFrame(dict(feature=['a', 'b'], block=['history', 'sensory'],
                                 description=['a', 's0'], model=['m1', 'm0'], oracle=[False, False]))
        failed = dict(bootstrap_ari_median=0., bootstrap_ari_p10=0., lomo_ari_min=0., lomo=[])
        with patch('post_analysis.session_clustering.cluster.stability', return_value=failed):
            _, _, labels, checks = fit_solutions(values, meta, np.repeat(['A', 'B', 'C'], 30), repeats=20)
        self.assertEqual(set(labels), {2, 3})
        for k in [2, 3]:
            self.assertEqual(set(labels[k]), set(range(1, k+1)))
        self.assertFalse(any(c['passes_diagnostics'] for c in checks))

    def test_bias_and_lapses_remain_display_only(self):
        meta = pd.DataFrame(dict(feature=['a', 'b', 'c', 'd', 'e'],
                                 block=['history', 'bias', 'sensory', 'sensory', 'prior'],
                                 description=['a', 'kappa', 'ell_short', 'ell_long', 'ah']))
        self.assertEqual(driver_manifest(meta).feature.tolist(), ['a', 'e'])
        self.assertEqual(len(meta), 5)

    def test_block_sampling_stays_within_subject_and_keeps_size(self):
        subjects = np.repeat(['A', 'B', 'C'], 30)
        ix = block_sample(subjects, np.random.default_rng(2))
        self.assertEqual(len(ix), 90)
        np.testing.assert_array_equal(subjects[ix], subjects)
        for mouse_start in (0, 30, 60):
            for offset in range(0, 30, BLOCK):
                length = min(BLOCK, 30-offset)
                start = mouse_start + offset
                np.testing.assert_equal(np.diff(ix[start:start+length]), np.ones(length-1))


if __name__ == '__main__':
    unittest.main()
