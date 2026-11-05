"""Scale audit geometry and resampling diagnostics."""
import unittest
import numpy as np
import pandas as pd
from post_analysis.session_clustering.scale_audit import geometry, transform, matched_jaccard
from post_analysis.session_clustering.features import matrix
from test_compact_clustering import raw_fixture
from post_analysis.session_clustering.features import compact_features

class ScaleTests(unittest.TestCase):
    def test_own_z_reproduces_existing_matrix(self):
        raw=raw_fixture();v,m=compact_features(raw);a,_=matrix(v,m,raw.subject)
        b,_,_=geometry(v,m,raw.subject,'own_z');np.testing.assert_allclose(a,b,atol=1e-12)

    def test_pooled_scale_is_equal_mouse_rms_not_raw_between_mouse_variance(self):
        raw=raw_fixture();v,m=compact_features(raw);x,g,u=geometry(v,m,raw.subject,'pooled_z')
        expected=np.sqrt(np.mean([v.loc[raw.subject.eq(s),u.feature].var(ddof=1).to_numpy() for s in ['A','B','C']],axis=0))
        np.testing.assert_allclose(g['denominator'],expected)
        shifted=v.copy();shifted.loc[raw.subject.eq('B')]+=100
        xx,gg,_=geometry(shifted,m,raw.subject,'pooled_z')
        np.testing.assert_allclose(x,xx,atol=1e-11)

    def test_shared_scale_preserves_relative_variation_across_mice(self):
        subjects=np.repeat(['A','B'],20);v=pd.DataFrame({'a':np.r_[np.linspace(-1,1,20),np.linspace(-2,2,20)]})
        m=pd.DataFrame(dict(feature=['a'],model=['m0'],block=['sensory'],oracle=[False]))
        x,_,_=geometry(v,m,subjects,'pooled_z');self.assertAlmostEqual(x[20:].std()/x[:20].std(),2.)
        own,_,_=geometry(v,m,subjects,'own_z');self.assertAlmostEqual(own[20:].std()/own[:20].std(),1.)

    def test_heldout_data_cannot_refit_shared_denominator(self):
        raw=raw_fixture();v,m=compact_features(raw);keep=raw.subject.ne('C')
        _,g,_=geometry(v.loc[keep],m,raw.subject[keep],'pooled_z');before=g['denominator'].copy()
        transform(v.loc[~keep]*100,raw.subject[~keep],g)
        np.testing.assert_array_equal(before,g['denominator'])

    def test_missing_reference_group_is_not_falsely_counted_as_recovered(self):
        j=matched_jaccard(np.array([1,1,2,2]),np.array([2,2,1,1]),3)
        np.testing.assert_allclose(j[:2],[1,1]);self.assertTrue(np.isnan(j[2]))


class RobustMethodTests(unittest.TestCase):
    def test_pam_reaches_optimum_in_simple_separated_data(self):
        from itertools import combinations
        from scipy.spatial.distance import cdist
        from post_analysis.session_clustering.scale_audit import PAM
        x=np.array([[0.,0.],[0.,1.],[1.,0.],[10.,10.],[10.,11.],[11.,10.]])
        model=PAM(2).fit(x)
        optimal=min(cdist(x,x[list(m)]).min(axis=1).sum() for m in combinations(range(len(x)),2))
        self.assertAlmostEqual(model.cost_,optimal)
        self.assertEqual(len(np.unique(model.labels_)),2)
        np.testing.assert_array_equal(model.labels_,model.predict(x))

    def test_circular_resampling_has_no_endpoint_undercoverage(self):
        from post_analysis.session_clustering.scale_supplement import circular_sample
        s=np.repeat(['A','B','C'],30);counts=np.zeros(len(s));rng=np.random.default_rng(51)
        for _ in range(3000):
            ix=circular_sample(s,rng)
            self.assertEqual(len(ix),len(s));np.testing.assert_array_equal(s[ix],s)
            np.add.at(counts,ix,1)
        # Expected multiplicity is exactly one per row, including both endpoints.
        self.assertLess(np.max(np.abs(counts/3000-1)),.07)

if __name__=='__main__':unittest.main()
