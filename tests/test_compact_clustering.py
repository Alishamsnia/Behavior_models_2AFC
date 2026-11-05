"""Scientific separation of historical-window drivers and target-session outcomes."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import expit
from post_analysis.session_clustering.features import compact_features, dynamic_annotations, select_features, matrix
from post_analysis.session_clustering.cluster import stability


def raw_fixture():
    rng = np.random.default_rng(42)
    n = 60
    r = pd.DataFrame(dict(subject=np.repeat(['A','B','C'],20), window_id=np.tile(np.arange(20),3),
                          test_rank=np.tile(np.arange(13,33),3), session_id=[str(i) for i in range(n)],
                          param__m0_sensory__w=rng.uniform(.01,.4,n), param__m0_sensory__s0=rng.uniform(.2,.6,n),
                          param__m1_action_kernel__a=rng.normal(-1, .4,n),
                          param__m2_static__delta0=rng.uniform(.4,2,n), param__m2_static__ah=rng.normal(-2,.4,n),
                          param__m2_dynamic__ah=rng.normal(-2,.4,n), param__m2_dynamic__bh=rng.uniform(0,.2,n)))
    for prefix in ('nll__','validation_nll__','train_nll__'):
        for model in ('m0_sensory','m1_action_kernel','m2_static','m2_dynamic'):
            r[prefix+model] = rng.uniform(.3,.65,n)
    return r


class CompactTests(unittest.TestCase):
    def test_target_and_dynamic_scores_cannot_change_primary_features(self):
        raw=raw_fixture(); before, meta=compact_features(raw)
        changed=raw.copy()
        for c in changed:
            if c.startswith('nll__') or 'm2_dynamic' in c:
                changed[c] = 123.
        after, _=compact_features(changed)
        pd.testing.assert_frame_equal(before,after)
        self.assertEqual(set(meta.model),{'m0_sensory','m1_action_kernel','m2_static'})

    def test_effective_noise_matches_model_variance_not_weber_squared(self):
        r=raw_fixture(); values,_=compact_features(r)
        np.testing.assert_allclose(np.exp(values.noise_long),np.sqrt(r.param__m0_sensory__w*1.7**2+r.param__m0_sensory__s0**2))
        np.testing.assert_allclose(values.history_alpha,expit(r.param__m1_action_kernel__a))

    def test_zero_dynamic_slope_equals_constant_hazard(self):
        r=raw_fixture();r.param__m2_dynamic__bh=0
        a=dynamic_annotations(r)
        np.testing.assert_allclose(a.dynamic_hazard_10,a.dynamic_hazard_35)
        np.testing.assert_allclose(a.dynamic_gain_test,r.nll__m2_static-r.nll__m2_dynamic)

    def test_small_spread_is_removed_before_standardization(self):
        s=np.repeat(['A','B','C'],20);v=pd.DataFrame({'tiny':np.tile(np.linspace(0,1e-5,20),3),'useful':np.tile(np.linspace(0,1,20),3)})
        m=pd.DataFrame(dict(feature=['tiny','useful'],block=['prior']*2,model=['m2']*2,oracle=[False]*2,min_spread=[.01]*2,redundancy_threshold=[.95]*2))
        audit=select_features(v,m,s)
        self.assertEqual(audit.selected.tolist(),[False,True])
        self.assertEqual(matrix(v,m,s)[1].feature.tolist(),['useful'])

    def test_duplicate_same_block_removed_but_cross_block_retained(self):
        s=np.repeat(['A','B','C'],20);v=pd.DataFrame({'a':np.tile(np.linspace(0,1,20),3)})
        v['b']=2*v.a;v['c']=v.a
        m=pd.DataFrame(dict(feature=list('abc'),block=['prior','prior','history'],model=['m2','m2','m1'],oracle=[False]*3,min_spread=[.01]*3,redundancy_threshold=[.95]*3))
        self.assertEqual(matrix(v,m,s)[1].feature.tolist(),['a','c'])

    def test_lomo_feature_selection_never_uses_excluded_mouse(self):
        r=raw_fixture();v,m=compact_features(r);s=r.subject.to_numpy();x,_=matrix(v,m,s)
        from post_analysis.session_clustering.cluster import partition
        calls=[]
        from post_analysis.session_clustering import cluster
        original=cluster.matrix
        def spy(values,meta,subjects):
            calls.append(set(subjects));return original(values,meta,subjects)
        with patch.object(cluster,'matrix',side_effect=spy):
            result=stability(v,m,s,partition(x,2),2,0,np.random.default_rng(1))
        self.assertEqual(calls,[{'B','C'},{'A','C'},{'A','B'}])
        self.assertEqual(len(result['lomo']),3)

if __name__=='__main__':unittest.main()
