"""Cluster overlays of lick latency by ISI, following Behavior/New_version.

Reference: licking_analysis.ipynb cells 2/4/6 and the six-subject nonnegative
latency runners. This module never executes the notebook or edits raw inputs.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import test_sessions

RAW_ROOT = Path('/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Behavior/New_version')
KEYS = ['subject', 'session_id']
COLORS = ['#4477AA', '#EE6677', '#228833']
METHOD = dict(
    reference=str(RAW_ROOT/'licking_analysis.ipynb'),
    definition='Saved lick_reaction_time: first lick at/after choice-window onset, seconds.',
    bins='Four rank/qcut ISI bins separately within short and long trial type, per cluster/condition.',
    weighting='Mean of within-session latency means; SEM across sessions. Session identity includes mouse.',
    exclusions=['warm_up != 0', 'any negative value in either all-lick reaction/start-from-choice array',
                'nonfinite or negative scalar latency'],
    trial_population='Only trials in clustered sliding-window held-out test sessions; no new fits.',
    source_join='Unique session date within mouse, then source_trial_index; stimulus, block and warm-up verified.',
)


def numeric_list(value):
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    elif isinstance(value, str) and value.strip():
        try:
            values = json.loads(value)
        except (ValueError, TypeError):
            try:
                values = ast.literal_eval(value)
            except (ValueError, SyntaxError, TypeError):
                values = []
    else:
        values = []
    result = []
    if isinstance(values, (list, tuple)):
        for v in values:
            try:
                n = float(v)
            except (ValueError, TypeError):
                continue
            if np.isfinite(n):
                result.append(n)
    return result


def filter_trials(data):
    """Keep filter flags for an auditable, non-destructive exclusion."""
    data = data.copy()
    data['negative_lick'] = False
    for col in ['all_lick_reaction_times', 'all_lick_start_times_from_choice']:
        if col not in data:
            raise ValueError(f'Missing negative-lick screening column: {col}')
        data['negative_lick'] |= data[col].apply(lambda v: any(x < 0 for x in numeric_list(v)))
    data['latency_s'] = pd.to_numeric(data.lick_reaction_time, errors='coerce')
    data['latency_eligible'] = (data.warm_up.eq(0) & ~data.negative_lick &
                                np.isfinite(data.latency_s) & data.latency_s.ge(0))
    data['outcome_strict'] = data.outcome.replace({'RewardNaive':'Reward', 'PunishNaive':'Punish'})
    data['is_common'] = np.nan
    biased = data.true_block.isin(['S', 'L'])
    data.loc[biased, 'is_common'] = (data.loc[biased, 'category_long'].eq(1) ==
                                    data.loc[biased, 'true_block'].eq('L')).astype(float)
    return data


def verify_join(expected, raw):
    """A raw row number is safe only after independent stimulus/context checks."""
    if len(expected) != len(raw):
        raise ValueError('Raw and held-out session trial counts differ')
    if not np.allclose(expected.duration_s, raw.trial_isi, rtol=0, atol=1e-10):
        raise ValueError('Raw reaction-time join has different ISIs')
    if not np.array_equal(expected.category_long, raw.trial_type-1):
        raise ValueError('Raw reaction-time join has different stimulus categories')
    if not np.array_equal(expected.true_block, raw.block_type.map({0:'N',1:'S',2:'L'})):
        raise ValueError('Raw reaction-time join has different block labels')
    if not np.array_equal(expected.warm_up, raw.warm_up):
        raise ValueError('Raw reaction-time join has different warm-up flags')


def load_trials(labels, out, raw_root=RAW_ROOT):
    requested = labels[KEYS].drop_duplicates()
    all_frames = []
    audits = []
    raw_columns = ['session_idx','session_date','trial_isi','trial_type','block_type','warm_up','outcome',
                   'lick_reaction_time','all_lick_reaction_times','all_lick_start_times_from_choice','Contingency_idx']
    for subject in requested.subject.unique():
        path = raw_root/f'{subject}_final_data.csv'
        raw = pd.read_csv(path, usecols=raw_columns)
        dates = raw[['session_idx','session_date']].drop_duplicates()
        if dates.session_date.duplicated().any():
            raise ValueError(f'{subject}: ambiguous raw session dates')
        raw['_raw_trial_index'] = raw.groupby('session_idx', sort=False).cumcount()
        wanted = set(requested[requested.subject.eq(subject)].session_id)
        for _, pred in test_sessions(subject, extra_columns=['source_trial_index']):
            sid = pred.session_id.iloc[0]
            if sid not in wanted:
                continue
            date_match = re.search(r'_(\d{8})_\d{6}$', sid)
            if date_match is None:
                raise ValueError(f'Cannot recover raw session date: {sid}')
            date = int(date_match[1])
            source = raw[raw.session_date.eq(date)].set_index('_raw_trial_index', drop=False)
            if source.empty:
                raise ValueError(f'No raw reaction-time session: {sid}')
            ix = pred.source_trial_index.to_numpy()
            if len(source) != len(pred) or not set(ix) == set(source.index):
                raise ValueError(f'{sid}: raw and canonical source trial indices differ')
            source = source.loc[ix].reset_index(drop=True)
            pred = pred.reset_index(drop=True)
            verify_join(pred, source)
            # These clustered mice are the reverse-contingency cohort in the reference.
            if not source.Contingency_idx.eq(2).all():
                raise ValueError(f'{sid}: unexpected contingency for MC06/07/08')
            frame = pred.assign(subject=subject)
            for col in ['outcome','lick_reaction_time','all_lick_reaction_times','all_lick_start_times_from_choice']:
                frame[col] = source[col]
            frame = filter_trials(frame)
            audits.append(dict(subject=subject,session_id=sid,raw_file=str(path),raw_session_idx=int(source.session_idx.iloc[0]),
                               n_trials=len(frame),n_warmup=int(frame.warm_up.ne(0).sum()),
                               n_negative_lick=int(frame.negative_lick.sum()),
                               n_invalid_scalar=int((~np.isfinite(frame.latency_s)|frame.latency_s.lt(0)).sum()),
                               n_retained=int(frame.latency_eligible.sum()),stimulus_context_verified=True))
            all_frames.append(frame)
    data = pd.concat(all_frames, ignore_index=True)
    got = data[KEYS].drop_duplicates()
    check = requested.merge(got,on=KEYS,how='outer',indicator=True,validate='one_to_one')
    if not check._merge.eq('both').all():
        raise ValueError('Reaction-time source sessions do not match cluster assignments')
    if data.duplicated(KEYS+['trial_index']).any():
        raise ValueError('Duplicate reaction-time trials')
    pd.DataFrame(audits).to_csv(out/'reaction_time_input_audit.csv',index=False)
    # Keep reproducible flags and raw trial identity; do not duplicate long lick arrays in output.
    data.drop(columns=['all_lick_reaction_times','all_lick_start_times_from_choice']).to_csv(out/'reaction_time_trials.csv',index=False)
    return data[data.latency_eligible].copy()


def binned_summary(data, n_bins=4):
    """Exactly the reference's rank bins, raw mean ISI, and mean/SEM of session means."""
    pieces = []
    for category in [0, 1]:
        t = data[data.category_long.eq(category)].dropna(subset=['duration_s','latency_s']).copy()
        if t.empty:
            continue
        t['isi_bin'] = pd.qcut(t.duration_s.rank(method='first'), q=min(n_bins,len(t)), labels=False, duplicates='drop').astype(int)
        group = ['category_long','isi_bin']
        x = t.groupby(group,observed=True).agg(isi_mean=('duration_s','mean'),n_trials=('latency_s','size')).reset_index()
        sessions = t.groupby(KEYS+group,observed=True).latency_s.mean().rename('session_latency_mean').reset_index()
        y = sessions.groupby(group,observed=True).agg(latency_mean=('session_latency_mean','mean'),
                          latency_sem=('session_latency_mean','sem'),n_sessions=('session_latency_mean','size')).reset_index()
        summary = x.merge(y,on=group,validate='one_to_one')
        summary['latency_sem'] = summary.latency_sem.fillna(0)
        pieces.append(summary)
    if not pieces:
        return pd.DataFrame(columns=['category_long','isi_bin','isi_mean','n_trials','latency_mean','latency_sem','n_sessions'])
    return pd.concat(pieces,ignore_index=True).sort_values('isi_mean').reset_index(drop=True)


def make_page(tagged, k):
    biased = tagged.true_block.isin(['S','L'])
    reward = tagged.outcome_strict.eq('Reward')
    punish = tagged.outcome_strict.eq('Punish')
    common = tagged.is_common.eq(1)
    rare = tagged.is_common.eq(0)
    conditions = [
        ('All trials', pd.Series(True,index=tagged.index)),
        ('Neutral block',tagged.true_block.eq('N')),
        ('Short block',tagged.true_block.eq('S')),
        ('Long block',tagged.true_block.eq('L')),
        ('Common trials',common),('Rare trials',rare),
        ('Reward · non-neutral',biased&reward),('Punish · non-neutral',biased&punish),
        ('Common · reward',common&reward),('Common · punish',common&punish),
        ('Rare · reward',rare&reward),('Rare · punish',rare&punish),
    ]
    fig,axes = plt.subplots(3,4,figsize=(18,14),gridspec_kw=dict(wspace=.38,hspace=.50))
    records=[]
    for ax,(name,mask) in zip(axes.flat,conditions):
        for c in range(1,k+1):
            data = tagged[mask & tagged.cluster_label.eq(c)]
            summary = binned_summary(data)
            if summary.empty:
                continue
            counts = f'{len(data[KEYS].drop_duplicates())} sessions, {len(data)} trials'
            ax.errorbar(summary.isi_mean,summary.latency_mean,yerr=summary.latency_sem,
                        fmt='-',color=COLORS[c-1],lw=1.6,elinewidth=1,capsize=2,
                        label=f'Cluster {c} ({counts})')
            records.append(summary.assign(K=k,cluster=c,condition=name))
        ax.axvline(1.25,color='.5',ls=':',lw=.8)
        ax.set_title(name,fontsize=11)
        ax.set_xlabel('ISI (s)');ax.set_ylabel('Reaction time (s)')
        ax.set_box_aspect(1)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False,fontsize=6.7,loc='best')
        else:
            ax.text(.5,.5,'No eligible trials',ha='center',transform=ax.transAxes)
    sizes=tagged[KEYS+['cluster_label']].drop_duplicates().groupby('cluster_label').size().to_dict()
    fig.suptitle(f'Pooled reaction time by ISI — K={k}, clusters superimposed',fontsize=16,fontweight='bold',y=.985)
    fig.text(.5,.955,'Four bins per short/long stimulus type · mean ± SEM across session means',ha='center',fontsize=11)
    fig.subplots_adjust(left=.065,right=.98,top=.91,bottom=.09)
    fig.text(.065,.025,'First lick at/after choice-window onset. Warm-up, invalid latencies and any pre-choice lick trials excluded.\n'
             'Only clustered held-out sessions (MC06/MC07/MC08, reverse contingency); no latency trimming. Bins are computed per cluster/condition.',fontsize=9,color='.3')
    summaries=pd.concat(records,ignore_index=True) if records else pd.DataFrame()
    return fig,summaries,sizes
