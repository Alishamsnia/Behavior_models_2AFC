"""Latency filtering and session-weighted aggregation, independent of external files."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post_analysis.session_clustering.latency import filter_trials,binned_summary,verify_join


class LatencyTests(unittest.TestCase):
    def test_premature_lick_excludes_even_with_positive_saved_latency(self):
        d=pd.DataFrame(dict(warm_up=[0]*4,lick_reaction_time=[.2]*4,
                            all_lick_reaction_times=['[-0.01, 0.2]','[0.2]','[0.2]','[0.2]'],
                            all_lick_start_times_from_choice=['[0.2]','[-0.02, 0.2]','[0.2]','[]'],
                            outcome=['Reward']*4,true_block=['S']*4,category_long=[0]*4))
        self.assertEqual(filter_trials(d).latency_eligible.tolist(),[False,False,True,True])

    def test_warmup_negative_nonfinite_excluded_but_zero_included(self):
        d=pd.DataFrame(dict(warm_up=[1,0,0,0,0],lick_reaction_time=[.2,-.1,np.nan,np.inf,0],
                            all_lick_reaction_times=['[]']*5,all_lick_start_times_from_choice=['[]']*5,
                            outcome=['RewardNaive']*5,true_block=['S']*5,category_long=[0]*5))
        out=filter_trials(d)
        self.assertEqual(out.latency_eligible.tolist(),[False,False,False,False,True])
        self.assertTrue(out.outcome_strict.eq('Reward').all())

    def test_sessions_receive_equal_weight_even_with_unequal_trial_counts(self):
        d=pd.DataFrame(dict(subject=['A']*100+['B']*2,session_id=['same']*102,
                            category_long=[0]*102,duration_s=[.5]*102,latency_s=[1.]*100+[3.]*2))
        s=binned_summary(d,n_bins=1).iloc[0]
        self.assertAlmostEqual(s.latency_mean,2.)
        self.assertAlmostEqual(s.latency_sem,1.)
        self.assertEqual(s.n_sessions,2)
        self.assertEqual(s.n_trials,102)

    def test_short_and_long_stimuli_get_separate_four_bins(self):
        d=pd.DataFrame(dict(subject=['A']*16,session_id=['s']*16,
                            category_long=[0]*8+[1]*8,duration_s=np.linspace(.2,2.3,16),latency_s=[.2]*16))
        s=binned_summary(d)
        self.assertEqual(len(s),8)
        self.assertEqual(s.groupby('category_long').size().to_dict(),{0:4,1:4})

    def test_wrong_raw_trial_mapping_is_rejected(self):
        pred=pd.DataFrame(dict(duration_s=[.5],category_long=[0],true_block=['S'],warm_up=[0]))
        raw=pd.DataFrame(dict(trial_isi=[.6],trial_type=[1],block_type=[1],warm_up=[0]))
        with self.assertRaisesRegex(ValueError,'different ISIs'):
            verify_join(pred,raw)
        raw.trial_isi=.5
        verify_join(pred,raw)
        raw.trial_type=2
        with self.assertRaisesRegex(ValueError,'different stimulus'):
            verify_join(pred,raw)


if __name__=='__main__':unittest.main()
