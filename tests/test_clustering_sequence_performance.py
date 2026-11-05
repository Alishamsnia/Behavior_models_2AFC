"""Sequence definitions, block containment and pooling for the supplied E/F plots."""
import unittest

import numpy as np
import pandas as pd

from post_analysis.session_clustering.sequence_performance import (
    adjusted_block_start, collect_strict_common_events, source_module, summarize,
)


class SequencePerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = source_module()

    def trials(self, kinds, outcomes=None):
        """A complete short block and one next block to bound its endpoint."""
        n = len(kinds)
        return pd.DataFrame(dict(
            change_session=['mouse|session']*(n+1), subject=['mouse']*(n+1),
            _source_order=range(n+1), block_type=[1]*n+[2],
            is_rare=list(kinds)+[False],
            correct=(list(outcomes) if outcomes is not None else [1.]*n)+[1.]))

    def test_E_requires_three_commons_and_scores_final(self):
        data = self.trials([True, False, False, False, False], [1, 1, 0, 1, 0])
        events = collect_strict_common_events(data, self.source)
        self.assertEqual(events.anchor_position.tolist(), [3, 4])
        self.assertEqual(events.trace.tolist(), ['punished_common', 'rewarded_common'])
        self.assertEqual(events.performance.tolist(), [1., 0.])
        self.assertEqual(events.train_start_position.tolist(), [1, 2])

    def test_missing_outcome_does_not_bridge_sequence(self):
        data = self.trials([False]*5, [1, 1, np.nan, 0, 1])
        events = collect_strict_common_events(data, self.source)
        # Only final position 4: missing outcome is context only, not scored.
        self.assertEqual(events.anchor_position.tolist(), [4])
        f = self.source.collect_sequence_events(
            self.trials([False, False, True, False], [1, np.nan, 1, 1]), adjusted_block_start)
        self.assertTrue(f.empty)

    def test_windows_complete_train_containment_and_overlap(self):
        event = pd.DataFrame([dict(block_id='b', sequence_id='s', block_length=31,
                                   train_start_position=4, train_end_position=6)])
        result = self.source.collect_complete_train_sliding_events(event)
        self.assertEqual(result.window_center.tolist(), [4., 7.])
        self.source.validate_sliding_window_boundaries(result)
        event['train_start_position'] = 24
        event['train_end_position'] = 26
        result = self.source.collect_complete_train_sliding_events(event)
        self.assertEqual(result.window_center.tolist(), [25.])

    def test_session_and_block_boundaries(self):
        one = self.trials([False]*5)
        two = self.trials([False]*5)
        two['change_session'] = 'mouse|second_session'
        events = collect_strict_common_events(pd.concat([one, two]), self.source)
        self.assertEqual(len(events), 6)
        self.assertEqual(events.groupby('session').size().tolist(), [3, 3])
        # The appended final block has no following boundary and is omitted.
        self.assertTrue(events.block_context.eq('Short block').all())
        self.assertTrue(events.train_end_position.lt(events.block_length).all())

    def test_F_uses_paired_three_outcomes_in_four_trial_train(self):
        data = self.trials([False, False, True, False], [0, 1, 0, 1])
        events = self.source.collect_sequence_events(data, adjusted_block_start)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[['pre', 'rare', 'post']].iloc[0].tolist(), [1., 0., 1.])
        self.assertEqual(events[['train_start_position', 'train_end_position']].iloc[0].tolist(), [0, 3])

    def test_pooled_row_weights_events_not_subjects(self):
        events = pd.DataFrame(dict(
            cluster_label=[1]*4, subject=['MC06']*3+['MC07'], session=['a']*3+['b'],
            window_center=[4]*4, block_context=['Short block']*4,
            trace=['rewarded_common']*4, performance=[1., 1., 1., 0.]))
        result = summarize(events, 'E')
        row = result[result.subject.eq('All subjects pooled') & result.block_context.eq('All biased blocks')].iloc[0]
        self.assertEqual(row['mean'], .75)
        self.assertAlmostEqual(row['sem'], .25)
        self.assertEqual(row.n_subjects, 2)
        self.assertEqual(row.n_sessions, 2)
        self.assertEqual(row.n_sequences, 4)


if __name__ == '__main__':
    unittest.main()
