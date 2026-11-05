"""Session isolation and reuse of the existing pooled behaviour visualization."""
from pathlib import Path
import sys
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from post_analysis.session_clustering.behavior import attach_labels, cluster_frames, collect_events, make_page
import matplotlib.pyplot as plt


class SessionTests(unittest.TestCase):
    def test_same_session_name_in_different_mice_joins_correctly(self):
        labels = pd.DataFrame(dict(subject=['A', 'B'], session_id=['same', 'same'], cluster_label=[1, 2]))
        trials = pd.DataFrame(dict(subject=['A', 'A', 'B'], session_id=['same']*3, trial_index=[0, 1, 0]))
        tagged = attach_labels(trials, labels)
        self.assertEqual(tagged.cluster_label.tolist(), [1, 1, 2])

    def test_missing_or_extra_label_fails(self):
        trials = pd.DataFrame(dict(subject=['A'], session_id=['s']))
        for session in ['missing', 'extra']:
            labels = pd.DataFrame(dict(subject=['A'], session_id=[session], cluster_label=[1]))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                attach_labels(trials, labels)

    def test_duplicate_session_labels_fail(self):
        trials = pd.DataFrame(dict(subject=['A'], session_id=['s']))
        labels = pd.DataFrame(dict(subject=['A', 'A'], session_id=['s', 's'], cluster_label=[1, 2]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            attach_labels(trials, labels)

    def test_clusters_partition_trials_without_repetition(self):
        trials = pd.DataFrame(dict(subject=['A']*4, session_id=['s1', 's1', 's2', 's3'], trial_index=[0, 1, 0, 0]))
        labels = pd.DataFrame(dict(subject=['A']*3, session_id=['s1', 's2', 's3'], cluster_label=[1, 2, 2]))
        groups = list(cluster_frames(trials, labels, 2))
        self.assertEqual([c for c, _ in groups], [1, 2])
        combined = pd.concat([g for _, g in groups])
        self.assertEqual(len(combined), len(trials))
        self.assertFalse(combined.duplicated(['subject', 'session_id', 'trial_index']).any())

    def test_missing_requested_cluster_fails(self):
        trials = pd.DataFrame(dict(subject=['A'], session_id=['s']))
        labels = pd.DataFrame(dict(subject=['A'], session_id=['s'], cluster_label=[1]))
        with self.assertRaisesRegex(ValueError, 'Expected cluster labels'):
            list(cluster_frames(trials, labels, 2))

    def test_mouse_events_need_no_prediction_columns(self):
        rows=[]
        for c in [1, 2]:
            for i in range(120):
                block='S' if (i//30)%2==0 else 'L'
                category=0 if block=='S' else 1
                if i%7==0:category=1-category
                rows.append(dict(subject='A',session_id=f's{c}',trial_index=i,
                                 true_block=block,category_long=category,choice_long=category,
                                 duration_s=.5 if category==0 else 2.,model_evaluated=True,warm_up=0,
                                 cluster_label=c))
        data=pd.DataFrame(rows)
        trans, rare=collect_events(data)
        self.assertEqual(set(trans.cluster), {1,2})
        self.assertEqual(set(rare.cluster), {1,2})
        self.assertTrue(trans.mouse.eq(1).all())
        fig, _, _=make_page(data,2)
        try:
            self.assertEqual(len(fig.axes),9)
            for ax in fig.axes:
                labels=ax.get_legend_handles_labels()[1]
                self.assertTrue(any(t.startswith('Cluster 1') for t in labels))
                self.assertTrue(any(t.startswith('Cluster 2') for t in labels))
            texts=[t.get_text().lower() for ax in fig.axes for t in ax.texts]
            texts += [ax.get_ylabel().lower() for ax in fig.axes]
            self.assertFalse(any('model' in t or 'difference' in t for t in texts))
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
