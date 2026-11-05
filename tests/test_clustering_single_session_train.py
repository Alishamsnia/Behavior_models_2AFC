"""Single-session training isolation and chronological-prefix contracts."""
import tempfile
import json
from unittest.mock import patch
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from post_analysis.session_clustering import pattern_rbf_single_session_train as m


def session():
    t = pd.DataFrame(dict(session_id=['s']*8, source_trial_index=range(8), trial_index=range(1, 9),
                          block_index=np.repeat([1, 2, 3, 4], 2), split_role=['train']*4+['test']*4,
                          model_evaluated=[True]*8, choice_long=[0, 1]*4,
                          model_probability_long=[.2, .8]*4))
    p = t.model_probability_long.to_numpy(); y = t.choice_long.to_numpy()
    t['per_trial_negative_log_likelihood'] = -y*np.log(p)-(1-y)*np.log1p(-p)
    return t


class SingleSessionTrainingTests(unittest.TestCase):
    def test_heldout_loss_and_choices_cannot_change_training_score(self):
        t = session(); score, n = m.training_score(m.training_population(t))
        self.assertEqual(n, 4); self.assertAlmostEqual(score, -np.log(.8))
        altered = t.copy(); altered.loc[altered.split_role.eq('test'), 'choice_long'] = 0
        altered.loc[altered.split_role.eq('test'), 'model_probability_long'] = np.nan
        altered.loc[altered.split_role.eq('test'), 'per_trial_negative_log_likelihood'] = 1e20
        self.assertEqual(m.training_score(m.training_population(altered)), (score, n))
        with self.assertRaisesRegex(ValueError, 'Only training'): m.training_score(t)

    def test_prefix_retains_unevaluated_trials_without_scoring_them(self):
        t = session(); t.loc[1, 'model_evaluated'] = False
        t.loc[1, 'choice_long'] = -1; t.loc[1, 'per_trial_negative_log_likelihood'] = np.nan
        prefix = m.training_population(t)
        self.assertEqual(prefix.source_trial_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(m.training_score(prefix)[1], 3)

    def test_literal_final_two_blocks_and_full_identity_are_required(self):
        for modification in ('split', 'duplicate', 'gap'):
            t = session()
            if modification == 'split': t.loc[3, 'split_role'] = 'test'
            if modification == 'duplicate': t.loc[3, 'source_trial_index'] = 2
            if modification == 'gap': t.loc[7, 'source_trial_index'] = 9
            with self.subTest(modification=modification), self.assertRaises(ValueError): m.training_population(t)

    def test_same_prefix_baseline_is_training_choice_entropy(self):
        choices = [1, 1, 1, 0]
        q, loss = m.base.constant_choice_nll(choices, choices)
        self.assertEqual(q, .75)
        self.assertAlmostEqual(loss, -.75*np.log(.75)-.25*np.log(.25))

    def test_behavior_and_sequence_refuse_heldout_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'training-prefix'):
                m.observed_metrics(session(), pd.DataFrame(), Path(tmp))
            with self.assertRaisesRegex(ValueError, 'training-prefix'):
                m.sequence_reports(session(), pd.DataFrame(), Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_captions_explain_training_population(self):
        fig = plt.figure(); text = fig.text(.1, .1, 'L = mean test NLL. Parameters summarize training-window fits; evaluated on test trials in held-out test sessions.')
        m.adapt_figure(fig)
        self.assertIn('mean in-sample training NLL', text.get_text())
        self.assertNotIn('test', text.get_text()); self.assertNotIn('held-out', text.get_text())
        self.assertIn('single-session training-prefix fits', text.get_text())
        plt.close(fig)



    def test_successful_prefix_metrics_and_sequences_ignore_heldout_changes(self):
        n = 40
        t = pd.DataFrame(dict(subject=['A']*n, session_id=['s']*n, source_trial_index=range(n), trial_index=range(1,n+1),
                              block_index=np.repeat(range(1,6),8), split_role=['train']*24+['test']*16,
                              model_evaluated=[True]*n, warm_up=[0]*n, latency_eligible=[True]*n, latency_s=[.5]*n))
        t['true_block'] = t.block_index.map({1:'S',2:'L',3:'S',4:'L',5:'S'})
        t['category_long'] = t.true_block.eq('L').astype(int)
        t.loc[t.source_trial_index.mod(8).eq(4),'category_long'] = 1-t.loc[t.source_trial_index.mod(8).eq(4),'category_long']
        t['choice_long'] = t.category_long
        wrong = t.source_trial_index.mod(8).eq(1)
        t.loc[wrong,'choice_long'] = 1-t.loc[wrong,'choice_long']
        t['outcome'] = np.where(wrong,'Punish','Reward')
        t['duration_s'] = np.where(t.category_long.eq(0),.5,1.5)+t.source_trial_index.mod(4)*.05
        t.loc[10,['choice_long','outcome','model_evaluated']] = [-1,'Omission',False]
        labels = pd.DataFrame(dict(subject=['A'],session_id=['s'],K=[3],cluster_label=[1]))
        raw = pd.DataFrame(dict(subject=['A'],session_id=['s'],nll__m0_sensory=[.5],nll__m1_action_kernel=[.4],
                               nll__m1_stimulus_kernel=[.45],nll__m2_static=[.42],nll__m2_dynamic=[.41],nll__m2_dynamic_repetition_bias=[.40]))
        with tempfile.TemporaryDirectory() as tmp:
            prefix = m.training_population(t)
            observed = m.observed_metrics(prefix,raw,Path(tmp))
            self.assertAlmostEqual(observed.p_long.iloc[0],9/23)
            self.assertAlmostEqual(observed.rare_accuracy_cost.iloc[0],-.15)
            self.assertAlmostEqual(observed.adjusted_rapid_S2L_recovery.iloc[0],.5)
            self.assertAlmostEqual(observed.adjusted_rapid_L2S_recovery.iloc[0],.5)
            data,_ = m.sequence_population(prefix,labels)
            self.assertEqual(data._source_order.tolist(),list(range(24)))
            self.assertTrue(np.isnan(data.loc[data._source_order.eq(10),'correct']).all())
            source = m.seq.source_module()
            e = m.seq.collect_strict_common_events(data,source)
            self.assertFalse(e.empty)
            self.assertFalse(e.block_id.str.endswith('_B3').any())
            changed = t.copy(); heldout=changed.split_role.eq('test')
            changed.loc[heldout,'choice_long']=1-changed.loc[heldout,'choice_long']
            changed.loc[heldout,'outcome']='Omission'
            prefix2=m.training_population(changed)
            pd.testing.assert_frame_equal(observed,m.observed_metrics(prefix2,raw,Path(tmp)))
            data2,_=m.sequence_population(prefix2,labels)
            pd.testing.assert_frame_equal(e,m.seq.collect_strict_common_events(data2,source))

    def test_collector_reads_single_fit_and_checks_parameters_and_cache(self):
        from post_analysis.expanded_model_comparison.config import MODEL_BY_ID
        spec=MODEL_BY_ID['m0_sensory']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'output'; output.mkdir(); cache_root=root/'cache'; cache_root.mkdir()
            fit=root/spec.directory/'A/full_session_matched_v1/session_001_s';fit.mkdir(parents=True)
            t=session();t['chronological_rank']=1;t['warm_up']=0;t['category_long']=[0,1]*4
            t['duration_s']=[.8,1.7]*4;t['true_block']=np.repeat(['S','L','S','L'],2);t['outcome']='Reward'
            pars=[.1,.2,.1,.1,1.]
            for name,value in zip(spec.parameter_names,pars):t['fitted_'+name]=value
            t.to_csv(fit/'trial_predictions.csv',index=False)
            (fit/'single_session_manifest.json').write_text(json.dumps(dict(session_id='s',mouse_id='A',model_id=spec.directory,
                    chronological_rank=1,status='complete',best_constrained_parameters=pars)))
            pd.DataFrame([dict(split_role='train',session_id='ALL_SESSIONS',valid_trial_count=4,mean_nll=-np.log(.8))]).to_csv(fit/'prefix_and_heldout_metrics.csv',index=False)
            cache=t.copy();cache['subject']='A'
            for col,value in dict(lick_reaction_time=.5,negative_lick=False,latency_s=.5,latency_eligible=True,outcome_strict='Reward',is_common=1).items():cache[col]=value
            cache.to_csv(cache_root/'reaction_time_trials.csv',index=False)
            with patch.object(m,'MODEL_SPECS',(spec,)),patch.object(m.features,'SUBJECTS',('A',)),patch.object(m,'EXPECTED_COUNTS',{'A':1}),patch.object(m,'SINGLE_ROOT',root),patch.object(m.base,'OLD',cache_root):
                raw,_,_,prefix,_=m.collect(output)
                self.assertEqual(len(raw),1);self.assertEqual(len(prefix),4)
                self.assertAlmostEqual(raw.nll__m0_sensory.iloc[0],-np.log(.8))
                bad=t.copy();bad.loc[0,'fitted_w']=.9;bad.to_csv(fit/'trial_predictions.csv',index=False)
                with self.assertRaises(AssertionError):m.collect(output)
                t.to_csv(fit/'trial_predictions.csv',index=False)
                bad=cache.copy();bad.loc[0,'choice_long']=1;bad.to_csv(cache_root/'reaction_time_trials.csv',index=False)
                with self.assertRaises(AssertionError):m.collect(output)


if __name__ == '__main__': unittest.main()
