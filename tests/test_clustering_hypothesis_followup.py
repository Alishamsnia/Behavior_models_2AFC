"""Checks for actual failure modes: chronology, matching and sparse uncertainty."""
import unittest
import numpy as np
import pandas as pd
from post_analysis.session_clustering.hypothesis_followup import (
    prepare_trials, shared_support, summarize_events, build_analyses, point_figure,
)
import matplotlib.pyplot as plt


class HypothesisFollowupTests(unittest.TestCase):
    def raw(self, choices, categories=None, blocks=None, outcomes=None):
        n = len(choices)
        categories = categories if categories is not None else [0]*n
        if outcomes is None:
            outcomes = ['Reward' if a == b else 'Punish' for a, b in zip(choices, categories)]
        return pd.DataFrame(dict(subject=['m']*n, session_id=['s']*n,
            cluster_label=[1]*n, source_trial_index=range(n), trial_index=range(n),
            true_block=blocks if blocks is not None else ['S']*n,
            category_long=categories, duration_s=[.5 if c == 0 else 2. for c in categories],
            choice_long=choices, outcome=outcomes, warm_up=[0]*n, model_evaluated=[True]*n))

    def test_omission_and_switch_do_not_bridge_history(self):
        d = prepare_trials(self.raw([0, 0, np.nan, 0, 0, 1, 1],
            blocks=['S']*5+['L']*2,
            outcomes=['Reward', 'Reward', 'DidNotChoose', 'Reward', 'Reward', 'Punish', 'Punish']))
        self.assertEqual(d.valid1.tolist(), [False, True, False, False, True, False, True])
        self.assertEqual(d.action_run.tolist(), [1, 2, 0, 1, 2, 1, 2])
        self.assertEqual(d.block_age.tolist(), [1, 2, 3, 4, 5, 1, 2])
        self.assertFalse(d.loc[4, 'valid2'])

    def test_never_matches_across_sessions(self):
        events = pd.DataFrame(dict(subject=['m', 'm'], session_id=['a', 'b'],
            cluster_label=[1, 1], condition=[0, 1], response=[0., 1.],
            cov=[1, 1], block_uid=['b1', 'b2']))
        table, retained = summarize_events(events, [0, 1], ['cov'])
        self.assertTrue(table.empty)
        self.assertTrue(retained.empty)
        self.assertIn('interval_supported', table.columns)
        fig = point_figure(1, 'block_age', table, [0, 1], 2)
        self.assertTrue(any('Insufficient' in t.get_text() for t in fig.axes[0].texts))
        plt.close(fig)

    def test_common_weights_remove_composition_artifact(self):
        # A and B have identical within-cell responses but opposite cell frequency.
        rows = []
        for cell, a, b, y in [('x', 10, 1, 0.), ('y', 1, 10, 1.)]:
            for condition, count in [(0, a), (1, b)]:
                rows += [dict(cov=cell, condition=condition, response=y, block_uid=f'{cell}{i}') for i in range(count)]
        result, retained = shared_support(pd.DataFrame(rows), [0, 1], ['cov'])
        self.assertAlmostEqual(result['mean_0'], .5)
        self.assertAlmostEqual(result['mean_1'], .5)
        self.assertAlmostEqual(result['contrast'], 0.)
        self.assertFalse(result['interval_supported'])
        self.assertEqual(len(retained), 22)

    def test_missing_third_condition_removes_cell(self):
        e = pd.DataFrame(dict(cov=['a', 'a', 'b', 'b', 'b'], condition=[1, 2, 1, 2, 3],
                              response=[0., 0., 1., 0., 1.], block_uid=list('abcde')))
        result, retained = shared_support(e, [1, 2, 3], ['cov'])
        self.assertEqual(result['n_cells'], 1)
        self.assertEqual(set(retained['cov']), {'b'})

    def test_age_refers_to_rare_anchor_and_not_target(self):
        cats = [0]*5+[1, 0]+[0]*13+[1, 0]+[0]*5+[1, 0]
        choices = cats.copy()
        d = prepare_trials(self.raw(choices, cats))
        events, _, _, _ = build_analyses(d)['block_age']
        self.assertEqual(events.anchor_age.tolist(), [6, 21])
        self.assertEqual(events.condition.tolist(), [0, 1])
        # Changing the target response cannot alter sequence eligibility.
        changed = self.raw(choices, cats)
        changed.loc[6, ['choice_long', 'outcome']] = [1, 'Punish']
        e2 = build_analyses(prepare_trials(changed))['block_age'][0]
        self.assertEqual(events.source_trial_index.tolist(), e2.source_trial_index.tolist())
        self.assertEqual(e2.response.tolist(), [1., 0.])

    def test_whole_blocks_supply_variance_not_individual_trials(self):
        rows = []
        for b in range(8):
            for cond in (0, 1):
                for j in range(4):
                    rows.append(dict(cov='x', condition=cond, response=float((b+j+cond) % 3 == 0), block_uid=f'b{b}'))
        result, _ = shared_support(pd.DataFrame(rows), [0, 1], ['cov'])
        self.assertTrue(result['interval_supported'])
        self.assertEqual(result['n_blocks'], 8)
        self.assertTrue(np.isfinite(result['se_block']))
        e = pd.DataFrame(rows)
        mu = e.groupby('condition').response.mean()
        e['score'] = np.where(e.condition.eq(1), 1., -1.) * (e.response-e.condition.map(mu)) / 32
        expected = np.sqrt(8/7 * (e.groupby('block_uid').score.sum()**2).sum())
        self.assertAlmostEqual(result['se_block'], expected)
        e['block_uid'] = [f'independent{i}' for i in range(len(e))]
        trial_result, _ = shared_support(e, [0, 1], ['cov'])
        self.assertNotAlmostEqual(result['se_block'], trial_result['se_block'])

    def test_run_matching_conditions_on_reference_choice(self):
        data = prepare_trials(self.raw([0, 0, 0, 1, 1, 1, 1]))
        strata = build_analyses(data)['action_run'][2]
        self.assertIn('prev1_choice_long', strata)

    def test_unequal_cell_weights_and_blocks_spanning_strata(self):
        rows = []
        specs = [('x', 0, [1, 0, 1, 0], [0, 1, 2, 0]),
                 ('x', 1, [0, 1], [0, 1]),
                 ('y', 0, [1, 1, 0], [0, 1, 2]),
                 ('y', 1, [0, 1, 0, 1, 1, 0], [0, 1, 2, 0, 1, 2])]
        for cell, condition, values, blocks in specs:
            rows += [dict(cov=cell, condition=condition, response=y, block_uid=f'b{b}')
                     for y, b in zip(values, blocks)]
        result, retained = shared_support(pd.DataFrame(rows), [0, 1], ['cov'])
        # Overlap weights 2/(2+3), 3/(2+3); block scores by hand:
        # b0=-1/6, b1=11/60, b2=-1/60 (sum zero).
        expected = np.sqrt(3/2 * ((-1/6)**2 + (11/60)**2 + (-1/60)**2))
        self.assertAlmostEqual(result['contrast'], -.1)
        self.assertAlmostEqual(result['se_block'], expected)
        self.assertAlmostEqual(retained[retained['cov'].eq('x')].cell_weight.iloc[0], .4)
        self.assertAlmostEqual(retained[retained['cov'].eq('y')].cell_weight.iloc[0], .6)


if __name__ == '__main__':
    unittest.main()
