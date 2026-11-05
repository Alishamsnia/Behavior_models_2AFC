"""Contracts for the four-coordinate star geometry and report labels."""
import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from post_analysis.session_clustering import pattern_rbf_seven_features as base
from post_analysis.session_clustering import pattern_rbf_seven_features_star as star
from post_analysis.session_clustering import pattern_rbf_four_features_star as four


class FourStarTests(unittest.TestCase):
    def test_exact_signed_coordinates_and_scaling(self):
        raw = pd.DataFrame({'gain_bias__m0_sensory': [.12, -.03, .02, .05],
                            'nll__m0_sensory': [.6, .7, .5, .55],
                            'nll__m1_action_kernel': [.53, .65, .6, .52],
                            'nll__m1_stimulus_kernel': [.55, .72, .5, .51],
                            'nll__m2_dynamic': [.50, .60, .45, .57]})
        with patch.object(base, 'DEFINITIONS', four.DEFINITIONS), patch.object(base, 'FEATURES', four.FEATURES):
            gains, manifest = base.gain_features(raw)
            np.testing.assert_allclose(gains.iloc[:2], [[.12, .07, .05, .10], [-.03, .05, -.02, .10]], atol=1e-14)
            z, scaling = base.standardized(gains, ['A']*4)
        self.assertEqual(z.shape, (4, 4))
        self.assertEqual(list(gains), four.FEATURES)
        self.assertTrue(manifest.weight.eq(1).all())
        self.assertEqual(len(base.FEATURES), 7)
        self.assertEqual(len(star.FEATURES), 7)

    def test_methods_only_lists_four_coordinates(self):
        fig = four.figure_adapter(base.methods_page)()
        values = [t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)]
        coordinate = next(v for v in values if v.startswith('Coordinate ='))
        self.assertEqual(coordinate, 'Coordinate = ' + four.FORMULAS)
        self.assertFalse(any('seven clustering' in v for v in values))
        plt.close(fig)

    def test_heatmap_has_four_cells_and_ticks(self):
        raw = pd.DataFrame({'subject': ['MC06', 'MC07', 'MC08']*2, 'session_id': list(range(6))})
        gains = pd.DataFrame(np.arange(24).reshape(6, 4), columns=four.FEATURES)
        labels = raw.assign(cluster_label=[1, 1, 1, 2, 2, 2])
        with patch.object(base, 'DEFINITIONS', four.DEFINITIONS), patch.object(base, 'FEATURES', four.FEATURES), patch.object(base, 'COLUMN_LABELS', star.COLUMN_LABELS):
            fig, order = four.figure_adapter(base.session_heatmap_page)(raw, gains, labels, np.arange(6), 2)
        ax = fig.axes[1]
        self.assertEqual(len(ax.get_xticks()), 4)
        self.assertEqual(ax.get_xlim(), (-.5, 3.5))
        self.assertEqual(ax.collections[0].get_array().size, 24)
        plt.close(fig)

    def test_raw_panels_match_selected_model_pairs(self):
        raw = pd.DataFrame({'subject': ['MC06', 'MC07', 'MC08']*2, 'session_id': list(range(6))})
        gains = pd.DataFrame(np.arange(24).reshape(6, 4)*.001, columns=four.FEATURES)
        labels = raw.assign(cluster_label=[1, 1, 1, 2, 2, 2])
        with patch.object(base, 'DEFINITIONS', four.DEFINITIONS), patch.object(base, 'FEATURES', four.FEATURES), patch.object(base, 'COLUMN_LABELS', star.COLUMN_LABELS):
            fig = four.figure_adapter(base.gain_detail_page)(raw, gains, labels, 2)
        titles = [ax.get_title().split('\n')[-1] for ax in fig.axes[:4]]
        self.assertEqual(titles, ['L(constant choice) - L(S)', 'L(S) - L(S+AH)',
                                  'L(S) - L(S+WH)', 'L(S) - L(S+Bdynamic)'])
        self.assertTrue(all(not ax.axison for ax in fig.axes[4:7]))
        plt.close(fig)

    def test_choice_behavior_supports_four_clusters(self):
        fig, axes = plt.subplots(3, 3)
        labels = pd.DataFrame({'cluster_label': [1, 2, 3, 4]})
        trials = labels.assign(model_evaluated=True, choice_long=1, warm_up=0, true_block='N')
        with patch.object(base, 'behavior_figure', return_value=(fig, None)), patch.object(base.behavior, 'attach_labels', return_value=trials):
            actual = base.choice_behavior_page(labels, trials, trials, 4)
        panels = actual.axes[-4:]
        self.assertEqual([ax.get_title() for ax in panels], [f'Cluster {c}: short vs long block' for c in range(1, 5)])
        self.assertEqual(len(actual.axes), 13)
        for first, second in zip(panels, panels[1:]):
            self.assertLess(first.get_position().x1, second.get_position().x0)
        plt.close(fig)
