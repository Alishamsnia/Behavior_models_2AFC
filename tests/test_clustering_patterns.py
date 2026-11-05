"""Model-derived pattern features and nonlinear affinity invariants."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID
from post_analysis.session_clustering.pattern_exploration import functional_features,gaussian_affinity,affinity,align_labels,behavior_profiles,METRICS

class PatternTests(unittest.TestCase):
    def test_existing_history_parameter_is_a_logit_not_an_amplitude(self):
        # alpha=.25: impulse response weights are .25*.75**(lag-1).
        model=MODEL_BY_ID['m1_action_kernel']
        raw=pd.DataFrame({'param__m1_action_kernel__w':[.1], 'param__m1_action_kernel__s0':[.3],
                          'param__m1_action_kernel__a':[np.log(1/3)]})
        v=pd.DataFrame(index=raw.index);m=pd.DataFrame(columns=['feature','model','block','oracle','description'])
        with patch('post_analysis.session_clustering.pattern_exploration.MODEL_SPECS',(model,)):
            values,meta=functional_features(raw,v,m,rich=True)
        for lag in (1,3,8):self.assertAlmostEqual(values[f'history__m1_action_kernel__{lag}'].iloc[0],.25*.75**(lag-1))
        self.assertAlmostEqual(np.exp(values['noise__m1_action_kernel__0.8'].iloc[0]),np.sqrt(.1*.8**2+.3**2))
        self.assertNotIn('param__m1_action_kernel__a',meta.feature.tolist())

    def test_kernel_is_symmetric_positive_semidefinite_and_fixed_scale(self):
        x=np.random.default_rng(2).normal(size=(25,4));a,width=gaussian_affinity(x)
        np.testing.assert_allclose(a,a.T,atol=1e-14);np.testing.assert_allclose(np.diag(a),1.)
        self.assertGreaterEqual(np.linalg.eigvalsh(a).min(),-1e-12)
        aa,ww=gaussian_affinity(x*7);np.testing.assert_allclose(a,aa,atol=1e-12)
        self.assertAlmostEqual(ww/width,49.)

    def test_multiview_is_mean_of_block_kernels_not_concatenated_rbf(self):
        x=np.random.default_rng(8).normal(size=(15,4));x[:,2:]*=8
        used=pd.DataFrame({'block':['sensory','sensory','history','history']})
        a,_=affinity(x,used,'multiview');a0,_=gaussian_affinity(x[:,:2]);a1,_=gaussian_affinity(x[:,2:])
        np.testing.assert_allclose(a,(a0+a1)/2)
        self.assertGreaterEqual(np.linalg.eigvalsh(a).min(),-1e-12)

    def test_alignment_only_permutes_labels(self):
        ref=np.array([1,1,2,2,3,3]);labels=np.array([3,3,1,1,2,2]);aligned,mapping=align_labels(labels,ref,3)
        np.testing.assert_array_equal(aligned,ref);self.assertEqual(len(set(mapping.values())),3)

    def test_equal_mouse_profile_does_not_hide_a_missing_mouse(self):
        rows=[]
        for s,labels in [('A',[1,2]),('B',[1,2]),('C',[2])]:
            for label in labels:rows.append(dict(subject=s,cluster_label=label,**{m:float(label) for m in METRICS}))
        means,contrasts=behavior_profiles(pd.DataFrame(rows),2)
        self.assertTrue(means[means.cluster_label.eq(1)].equal_mouse_mean.isna().all())
        self.assertTrue(means[means.cluster_label.eq(1)].n_mice_available.eq(2).all())
        np.testing.assert_allclose(means[means.cluster_label.eq(2)].equal_mouse_mean,2.)

if __name__=='__main__':unittest.main()
