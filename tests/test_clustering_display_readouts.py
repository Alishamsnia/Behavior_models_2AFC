"""Scientific units and weighting in the readable display, independent of plotting style."""
import unittest
import numpy as np
import pandas as pd
from post_analysis.session_clustering.pattern_visuals import readouts,balanced,MODELS

class DisplayReadoutTests(unittest.TestCase):
    def test_physical_units_and_percentage_loss_are_correct(self):
        raw=pd.DataFrame(dict(subject=['A'],session_id=['s'],window_id=[1],test_rank=[13],
                              param__m0_sensory__w=[.1],param__m0_sensory__s0=[.3],
                              param__m1_action_kernel__a=[0.],param__m2_static__delta0=[np.log(4)],
                              param__m2_static__ah=[np.log(1/9)],nll__m0_sensory=[.5]))
        for m,_,_ in MODELS:raw['nll__'+m]=.4
        original=raw.copy(deep=True);d,meta=readouts(raw)
        self.assertAlmostEqual(d.noise_short.iloc[0],np.sqrt(.1*.8**2+.3**2))
        self.assertAlmostEqual(d.memory.iloc[0],2.)
        self.assertAlmostEqual(d.contrast.iloc[0],30.)
        self.assertAlmostEqual(d.hazard.iloc[0],10.)
        self.assertAlmostEqual(d['gain__m1_action_kernel'].iloc[0],20.)
        self.assertEqual(len(meta),12)
        pd.testing.assert_frame_equal(raw,original)

    def test_display_gives_each_mouse_equal_weight_not_each_session(self):
        d=pd.DataFrame(dict(subject=['A']*10+['B','C'],session_id=[str(i) for i in range(12)],v=[0.]*10+[10.,20.]))
        labels=d[['subject','session_id']].assign(cluster_label=1)
        result,_=balanced(d,labels,['v'],1)
        self.assertAlmostEqual(result[0,0],10.)
        self.assertNotAlmostEqual(result[0,0],d.v.mean())

    def test_absent_mouse_is_not_filled_to_make_a_complete_group(self):
        d=pd.DataFrame(dict(subject=['A','A','B','B','C'],session_id=list('abcde'),v=[1.,2.,1.,2.,2.]))
        labels=d[['subject','session_id']].assign(cluster_label=[1,2,1,2,2])
        result,_=balanced(d,labels,['v'],2)
        self.assertTrue(np.isnan(result[0,0]));self.assertEqual(result[1,0],2.)

if __name__=='__main__':unittest.main()
