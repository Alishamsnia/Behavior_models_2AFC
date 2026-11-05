"""Numerical/scientific invariants for the independent four-gain Ward pipeline."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from post_analysis.zero_anchored_ward.core import normalize, ward_tree, partition, clustering
from post_analysis.zero_anchored_ward.data import constant_loss, assert_paired, IDENTITY, has_negative_array
from post_analysis.zero_anchored_ward.behavior import adjusted_start, event_tables
from post_analysis.zero_anchored_ward.followup import prepare, matched_estimates
from post_analysis.zero_anchored_ward.sequences import collect as sequence_events


def gains_fixture():
    return np.array([[1.,-.2,.1,0.],[2.,.2,.1,.3],[3.,.4,.3,.1],[.5,0.,.2,.2],
                     [1.5,.1,.4,.4],[.1,.6,.2,.5],[2.5,.3,.5,.1],[.2,.7,.1,.3]])


def test_exact_clamp_global_sample_sd_no_centering_and_row_formula():
    x=gains_fixture();g=normalize(x)
    positive=np.maximum(x,0)
    scale=np.std(positive,axis=0,ddof=1)
    expected_u=positive/scale
    expected_z=expected_u/np.sqrt((expected_u**2).sum(axis=1)+1e-24)[:,None]
    np.testing.assert_allclose(g.scale,scale)
    np.testing.assert_allclose(g.z,expected_z)
    assert g.z[0,1]==0 and g.z[0,3]==0
    assert (g.z>=0).all()
    np.testing.assert_allclose(np.linalg.norm(g.z,axis=1),1.)


def test_global_units_invariance_and_identical_profiles_across_mice():
    x=gains_fixture();x[1]=x[0]
    np.testing.assert_array_equal(normalize(x).z[0],normalize(x).z[1])
    np.testing.assert_allclose(normalize(x*np.array([5.,2.,.1,9.])).z,normalize(x).z)


def test_row_projection_discards_magnitude_with_fixed_global_scales():
    x=gains_fixture();x[1]=x[0]*7
    g=normalize(x)
    np.testing.assert_allclose(g.z[0],g.z[1])
    assert g.norm[1]==pytest.approx(7*g.norm[0])


def test_zero_column_zero_row_and_invalid_constants():
    x=gains_fixture();x[:,3]=0;x[0]=[-1,0,-2,0]
    g=normalize(x)
    assert g.scale[3]==1
    assert not g.eligible[0]
    np.testing.assert_array_equal(g.z[0],np.zeros(4))
    assert np.all(g.z[:,3]==0)
    x[:,3]=.5
    with pytest.raises(ValueError,match='Nonzero constant'):normalize(x)
    x[:,3]=1e-5
    with pytest.raises(ValueError,match='Nonzero constant'):normalize(x)
    with pytest.raises(ValueError):normalize([[np.nan]*4,[1.]*4])


def test_ward_receives_coordinates_and_chord_distance_matches_cosine():
    g=normalize(gains_fixture())
    expected=linkage(g.z,method='ward',metric='euclidean',optimal_ordering=True)
    np.testing.assert_allclose(ward_tree(g.z),expected)
    cosine=1-(g.z@g.z.T)
    squared=np.sum((g.z[:,None,:]-g.z[None,:,:])**2,axis=2)
    np.testing.assert_allclose(squared,2*cosine,atol=1e-14)
    tree=ward_tree(g.z)
    assert tree[0,2]==pytest.approx(pdist(g.z).min())


def test_cut_tree_exact_k_with_tied_heights_and_size_numbering():
    z=np.eye(4)
    for k in (2,3):
        lab=partition(ward_tree(z),k)
        assert set(lab)==set(range(1,k+1))
        sizes=np.bincount(lab)[1:]
        assert np.all(np.diff(sizes)<=0)


def test_zero_rows_retained_as_unresolved_and_all_candidates_exportable():
    x=gains_fixture();x=np.vstack([x,np.zeros(4)])
    g,tree,labels,selection,draws,loo,chosen=clustering(x,['A']*4+['B']*5,replicates=3,k_values=range(2,5))
    assert chosen in (2,3,4)
    assert len(tree)==7
    assert set(selection.K)=={2,3,4}
    for k,lab in labels.items():
        assert len(lab)==9 and lab[-1]==0
        assert set(lab[:-1])==set(range(1,k+1))
    assert not draws.empty and not loo.empty


def test_constant_baseline_probability_uses_training_only():
    train=np.array([1,0,0,0]);test=np.array([1,1])
    p,loss=constant_loss(train,test)
    q,other=constant_loss(train,[0,0])
    assert p==q==.25
    assert loss==pytest.approx(-np.log(.25))
    assert other==pytest.approx(-np.log(.75))
    with pytest.raises(ValueError):constant_loss([],test)


def test_prediction_pairing_rejects_changed_mask_or_trial_identity():
    a=pd.DataFrame({c:[0,1] for c in IDENTITY})
    assert_paired(a,a.copy())
    b=a.copy();b.loc[0,'model_evaluated']=1
    with pytest.raises(ValueError,match='Unpaired'):assert_paired(a,b)
    b=a.copy();b.loc[0,'source_trial_index']=2
    with pytest.raises(ValueError,match='Unpaired'):assert_paired(a,b)


def timeline(n=24):
    block=np.array(['S']*16+['L']*(n-16))
    category=np.array([0]*16+[1]*(n-16));category[4]=1
    return pd.DataFrame(dict(subject='MC06',session_id='test',source_trial_index=np.arange(n),trial_index=np.arange(n)+1,
        cluster_label=1,true_block=block,category_long=category,duration_s=np.where(category==0,.8,1.7),
        warm_up=0,choice_long=category,correct=1.,choice_eligible=True,model_evaluated=True,
        is_rare=category!=np.where(block=='S',0,1),latency_eligible=True,latency_s=.5,outcome_strict='Reward'))


def test_histories_do_not_jump_invalid_trials_or_block_boundaries():
    t=timeline();t.loc[3,'choice_eligible']=False
    g=prepare(t)
    assert not g.valid1.iloc[4]
    assert not g.valid2.iloc[5]
    assert not g.valid1.iloc[16]
    assert g.valid3.iloc[9]


def test_matched_support_excludes_unshared_cells_and_uses_equal_condition_weights():
    events=pd.DataFrame(dict(subject=['A']*7,session_id=['s']*7,cluster_label=[1]*7,
        stratum=['x','x','x','x','y','y','z'],condition=[0,0,1,1,0,1,0],
        response=[0.,1.,1.,1.,0.,0.,1.],block_uid=['b1','b1','b2','b2','b3','b4','b5']))
    estimates,retained=matched_estimates(events,[0,1],['stratum'],None)
    row=estimates.iloc[0]
    # x has min count 2, y has min count 1; z lacks condition 1 and is removed.
    assert row.mean_0==pytest.approx(1/3)
    assert row.mean_1==pytest.approx(2/3)
    assert row.contrast==pytest.approx(1/3)
    assert set(retained.stratum)=={'x','y'}
    assert not row.interval_supported
    assert np.isnan(row.ci_low)


def test_complete_sequence_trains_and_incomplete_last_block_exclusion():
    events,audit=sequence_events(timeline())
    f=events['F']
    assert set(f.trace)=={'pre','rare','post'}
    assert f.block_id.str.endswith('B0').all()
    assert np.all(f.train_start_position>=f.window_start)
    assert np.all(f.train_end_position<f.window_start+7)
    assert f.window_center.max()<=25
    assert audit.n_complete_blocks.iloc[0]==1
    assert not events['E'].empty


def test_transition_adjustment_and_raw_lick_screening():
    assert adjusted_start(np.array([False,True,True,False]),np.array(['S','S','S','L']),3)==1
    assert has_negative_array('[0.2, -0.01]')
    assert has_negative_array('[nan, -0.1]')
    assert not has_negative_array('[0.0, 0.5]')
    trans,rare=event_tables(timeline())
    assert set(trans.kind)=={'S2L'}
    assert set(rare.kind)=={'rare_long'}


def test_no_imports_from_previous_clustering_package():
    root=Path(__file__).resolve().parents[1]/'post_analysis/zero_anchored_ward'
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert 'session_clustering' not in (node.module or '')
            if isinstance(node,ast.Import):
                assert all('session_clustering' not in alias.name for alias in node.names)


def rt_four_groups():
    records=[]
    durations=[.35,.6,.8,1.,1.5,1.7,1.9,2.15]
    for cluster in (1,2,3,4):
        for block in ('N','S','L'):
            for duration in durations:
                for repeat in range(6):
                    category=int(duration>1.25)
                    records.append(dict(subject=f'M{cluster}',session_id=f's{cluster}',cluster_label=cluster,
                        true_block=block,category_long=category,duration_s=duration,
                        latency_s=.2+.04*cluster+.01*repeat+.002*duration,
                        latency_eligible=True,outcome_strict='Reward' if repeat%2 else 'Punish',
                        is_rare=block in ('S','L') and category!=(1 if block=='L' else 0)))
    return pd.DataFrame(records)


def test_k4_holm_known_values_and_missing_tests():
    from post_analysis.zero_anchored_ward.k4_figures import holm_adjust
    raw=pd.DataFrame({'p_raw':[.04,np.nan,.01,.03]})
    adjusted=holm_adjust(raw)
    np.testing.assert_allclose(adjusted.p_holm,[.06,np.nan,.03,.06],equal_nan=True)
    assert raw.columns.tolist()==['p_raw']


def test_k4_statistics_cover_all_six_pairs_and_96_condition_comparisons():
    from post_analysis.zero_anchored_ward.k4_figures import rt_statistics,PAIRS
    data=rt_four_groups()
    summary,tests,bins,bin_tests=rt_statistics(data)
    assert len(summary)==64
    assert len(tests)==96 and tests.p_holm.notna().all()
    assert set(map(tuple,tests[['cluster_a','cluster_b']].to_numpy()))==set(PAIRS)
    assert tests.groupby('condition').size().eq(6).all()
    assert (tests.p_holm>=tests.p_raw).all()
    pair=tests[tests.condition.eq('All trials')&tests.cluster_a.eq(1)&tests.cluster_b.eq(4)].iloc[0]
    assert pair.delta_s==pytest.approx(-.12)
    assert pair.n_a==pair.n_b==144
    assert pair.p_holm<.001
    assert bin_tests.groupby(['condition','category_long','isi_bin']).size().eq(6).all()
    assert (bin_tests.p_holm>=bin_tests.p_raw).all()


def test_k4_duration_bins_preserve_trial_counts_and_match_shared_physical_support():
    from post_analysis.zero_anchored_ward.k4_figures import rt_statistics
    from post_analysis.zero_anchored_ward.behavior import duration_bins
    data=rt_four_groups()
    # Unequal representation must not create separate bin boundaries per cluster.
    data=pd.concat([data,data[data.cluster_label.eq(4)&data.duration_s.eq(.35)]],ignore_index=True)
    assigned=duration_bins(data,shared_edges=True)
    assert assigned.groupby('duration_s').duration_bin.nunique().eq(1).all()
    summary,tests,bins,bin_tests=rt_statistics(data)
    original=summary[summary.condition.eq('All trials')].set_index('cluster_label').n_trials
    counted=bins[bins.condition.eq('All trials')].groupby('cluster_label').n_trials.sum()
    pd.testing.assert_series_equal(original.sort_index(),counted.sort_index(),check_names=False)
    for _,g in bins.groupby(['condition','category_long','isi_bin']):
        assert g.duration_min.nunique()==1 and g.duration_max.nunique()==1
