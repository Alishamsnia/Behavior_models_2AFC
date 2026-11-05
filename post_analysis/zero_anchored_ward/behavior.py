"""Observed behavior summaries on intact held-out timelines; no legacy imports."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit
from .core import KEYS


def attach(trials, labels):
    if labels.duplicated(KEYS).any():
        raise ValueError('Duplicate session labels')
    result = trials.merge(labels[KEYS+['cluster_label']], on=KEYS, how='left', validate='many_to_one')
    if result.cluster_label.isna().any():
        raise ValueError('Missing session labels')
    return result[result.cluster_label.gt(0)].copy()


def duration_bins(t, shared_edges=False):
    t = t.copy()
    t['duration_bin'] = -1
    for cat, g in t.groupby('category_long'):
        if shared_edges:
            edges = np.unique(np.quantile(g.duration_s, np.linspace(0, 1, 5)))
            codes = pd.cut(g.duration_s, edges, include_lowest=True, labels=False) if len(edges) > 1 else np.zeros(len(g))
        else:
            codes = pd.qcut(g.duration_s.rank(method='first'), min(4, len(g)), labels=False)
        t.loc[g.index, 'duration_bin'] = np.asarray(codes, int)+4*int(cat)
    return t


def curve(x, boundary, slope, low, high):
    return low+(1-low-high)*expit(slope*(x-boundary))


def psychometric(g):
    if len(g) < 8 or g.duration_s.nunique() < 4 or g.choice_long.nunique() < 2:
        return None
    try:
        return curve_fit(curve, g.duration_s, g.choice_long, p0=[1.25, 10., .05, .05],
                         bounds=([g.duration_s.min(), 0, 0, 0], [g.duration_s.max(), 100, .5, .5]), maxfev=10000)[0]
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def binned(g, value, session_weighted=False):
    if g.empty:
        return pd.DataFrame(columns=['duration_s', 'mean', 'sem', 'n_trials', 'n_sessions'])
    t = duration_bins(g)
    base = t.groupby('duration_bin').agg(duration_s=('duration_s','mean'), n_trials=(value,'size'))
    if session_weighted:
        vals = t.groupby(KEYS+['duration_bin'])[value].mean().reset_index()
    else:
        vals = t
    stats = vals.groupby('duration_bin')[value].agg(['mean','sem'])
    ns = t.groupby('duration_bin')[KEYS].apply(lambda x: len(x.drop_duplicates())).rename('n_sessions')
    return base.join(stats).join(ns).reset_index().sort_values('duration_s')


def rt_masks(t):
    b = t.true_block.isin(['S','L'])
    r, p = t.outcome_strict.eq('Reward'), t.outcome_strict.eq('Punish')
    c, rare = b & ~t.is_rare, b & t.is_rare
    return {'All trials':np.ones(len(t),bool), 'Neutral block':t.true_block.eq('N'),
            'Short block':t.true_block.eq('S'), 'Long block':t.true_block.eq('L'),
            'Common trials':c, 'Rare trials':rare, 'Reward - biased':b&r, 'Punish - biased':b&p,
            'Short - reward':t.true_block.eq('S')&r, 'Short - punish':t.true_block.eq('S')&p,
            'Long - reward':t.true_block.eq('L')&r, 'Long - punish':t.true_block.eq('L')&p,
            'Common - reward':c&r, 'Common - punish':c&p, 'Rare - reward':rare&r, 'Rare - punish':rare&p}


def adjusted_start(rare, block, formal):
    start = formal
    if formal > 0 and rare[formal-1]:
        start = formal-1
        while start > 0 and block[start-1] == block[formal-1] and rare[start-1]:
            start -= 1
    return start


def event_tables(tagged):
    """Transitions -20..19; rare events -2..2, original unfiltered timeline.

    Early rare stimuli (first four formal block trials) excluded only in transition
    plots. Bridges copy the last prior eligible matching-category event to -1,
    explicitly flagged; they never change the original event table.
    """
    transitions, rares = [], []
    for key, full in tagged.groupby(KEYS, sort=False):
        g = full.sort_values('source_trial_index')
        g = g[g.true_block.isin(['S','L'])].reset_index(drop=True)
        if g.empty:
            continue
        if len(g) > 1 and not np.all(np.diff(g.source_trial_index) == 1):
            raise ValueError(f'Internal gap in biased trial timeline: {key}')
        block = g.true_block.to_numpy()
        rare = g.is_rare.to_numpy(bool)
        runs = g.true_block.ne(g.true_block.shift()).cumsum()
        age = g.groupby(runs).cumcount().to_numpy()+1
        switches = np.flatnonzero(block[1:] != block[:-1])+1
        def record(i, event, rel, kind):
            r = g.iloc[i]
            return dict(subject=key[0], session_id=key[1], cluster_label=int(r.cluster_label), event_id=event,
                        kind=kind, rel=rel, category_long=int(r.category_long), source_trial_index=int(r.source_trial_index),
                        choice_valid=bool(r.choice_eligible), correct=r.correct if r.choice_eligible else np.nan,
                        rt_valid=bool(r.latency_eligible), latency_s=r.latency_s if r.latency_eligible else np.nan,
                        outcome=r.outcome_strict, bridge=False)
        for formal in switches:
            anchor = adjusted_start(rare, block, formal)
            lo = formal-1
            while lo > 0 and block[lo-1] == block[formal-1]:
                lo -= 1
            hi = formal
            while hi+1 < len(g) and block[hi+1] == block[formal]:
                hi += 1
            event = f'{key[0]}|{key[1]}|T{formal}'
            kind = 'S2L' if block[formal] == 'L' else 'L2S'
            for rel in range(-20,20):
                i = anchor+rel
                if lo <= i <= hi and not (rare[i] and age[i] <= 4):
                    transitions.append(record(i,event,rel,kind))
        for anchor in np.flatnonzero(rare):
            event = f'{key[0]}|{key[1]}|R{anchor}'
            for rel in range(-2,3):
                i = anchor+rel
                if 0 <= i < len(g) and runs.iloc[i] == runs.iloc[anchor]:
                    rares.append(record(i,event,rel,'rare_long' if g.category_long.iloc[anchor] == 1 else 'rare_short'))
    columns = [*KEYS,'cluster_label','event_id','kind','rel','category_long','source_trial_index',
               'choice_valid','correct','rt_valid','latency_s','outcome','bridge']
    return pd.DataFrame(transitions,columns=columns),pd.DataFrame(rares,columns=columns)


def bridge_events(g, value):
    additions=[]
    for _, e in g.dropna(subset=[value]).groupby('event_id',sort=False):
        if not e.rel.eq(-1).any():
            prior=e[e.rel.lt(-1)].sort_values('rel')
            if not prior.empty:
                r=prior.iloc[-1].copy(); r['rel']=-1; r['bridge']=True; additions.append(r)
    return pd.concat([g,pd.DataFrame(additions)],ignore_index=True) if additions else g


def session_metrics(trials):
    rows=[]
    for key,g in trials.groupby(KEYS,sort=False):
        v=g[g.choice_eligible]; b=v[v.true_block.isin(['S','L'])]
        rows.append(dict(subject=key[0],session_id=key[1],n_trials=len(v),accuracy=v.correct.mean(),
                         choose_long=v.choice_long.mean(),common_accuracy=b.loc[~b.is_rare,'correct'].mean(),
                         rare_accuracy=b.loc[b.is_rare,'correct'].mean(),
                         mean_rt=g.loc[g.latency_eligible,'latency_s'].mean(),
                         short_block_long=b.loc[b.true_block.eq('S'),'choice_long'].mean(),
                         long_block_long=b.loc[b.true_block.eq('L'),'choice_long'].mean()))
    return pd.DataFrame(rows)


def interpretation_metrics(trials):
    """Stimulus-matched effects freshly reconstructed from raw held-out choices."""
    basic=session_metrics(trials).set_index(KEYS)
    t=trials.copy();t['stim_bin']=-1
    for cat in (0,1):
        eligible=t[t.choice_eligible & t.category_long.eq(cat)]
        edges=np.unique(np.quantile(eligible.duration_s,np.linspace(0,1,5)))
        if len(edges)<2:raise ValueError('Insufficient variation for stimulus matching')
        edges[0],edges[-1]=-np.inf,np.inf
        mask=t.category_long.eq(cat)
        t.loc[mask,'stim_bin']=pd.cut(t.loc[mask,'duration_s'],edges,include_lowest=True,labels=False).to_numpy()+4*cat
    v=t[t.choice_eligible]
    basic['balanced_accuracy']=v.groupby(KEYS+['stim_bin']).correct.mean().groupby(KEYS).mean()
    block=v[v.true_block.isin(['S','L'])].groupby(KEYS+['stim_bin','true_block']).choice_long.agg(['mean','size']).reset_index()
    block=block[block['size'].ge(2)].pivot(index=KEYS+['stim_bin'],columns='true_block',values='mean').reindex(columns=['S','L']).dropna()
    basic['block_context_shift']=(block.L-block.S).groupby(KEYS).mean()
    basic['rare_accuracy_cost']=basic.common_accuracy-basic.rare_accuracy
    rt=t[t.latency_eligible]
    basic['rt_stimulus_balanced']=rt.groupby(KEYS+['stim_bin']).latency_s.mean().groupby(KEYS).mean()
    transitions,_=event_tables(t.assign(cluster_label=1))
    for kind,cat in [('S2L',1),('L2S',0)]:
        e=transitions[transitions.kind.eq(kind)&transitions.category_long.eq(cat)&transitions.choice_valid&transitions.rel.isin([0,1,3,4])].copy()
        e['phase']=np.where(e.rel.lt(2),'early','late')
        phase=e.groupby(KEYS+['event_id','phase']).correct.mean().unstack('phase').reindex(columns=['early','late']).dropna()
        basic[f'adjusted_rapid_{kind}_recovery']=(phase.late-phase.early).groupby(KEYS).mean()
    return basic.reset_index()
