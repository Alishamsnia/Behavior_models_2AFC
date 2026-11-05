"""Observed, within-session matched follow-ups for frozen seven-feature K3 labels.

Run: python -m post_analysis.session_clustering.hypothesis_followup
No decision models or cluster labels are fitted or modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .behavior import attach_labels
from .features import BASE
from .latency import RAW_ROOT, verify_join

PARENT = BASE / 'pooled/clustering/sliding_window_pattern_exploration_rbf_seven_features'
OUT = PARENT / '07_hypothesis_followup'
CACHE = BASE / 'pooled/clustering/version1/sliding_window_final/reaction_time_trials.csv'
EXPECTED_SIZES = {1: 11, 2: 25, 3: 56}
KEYS = ['subject', 'session_id']
MICE = ('MC06', 'MC07', 'MC08')
MARKERS = dict(zip(MICE, ('o', '^', 's')))
COLORS = {1: '#4477AA', 2: '#CC6677', 3: '#228833'}
HISTORY_COLORS = ('#3D648A', '#CB7845')
DURATIONS = np.array([.2, .35, .5, .65, .8, 1.7, 1.85, 2., 2.15, 2.3])
LABELS = {1: 'Action-history hypothesis', 2: 'Feedback / context hypothesis',
          3: 'Sensory / block-age hypotheses'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    sources = [CACHE, PARENT / 'selected_session_clusters.csv',
               *[RAW_ROOT / f'{mouse}_final_data.csv' for mouse in MICE]]
    source_hashes = {str(p): sha(p) for p in sources}
    labels = pd.read_csv(PARENT / 'selected_session_clusters.csv')
    labels = labels[labels.K.eq(3)].copy()
    if labels.groupby('cluster_label').size().to_dict() != EXPECTED_SIZES:
        raise ValueError('Unexpected K3 cohort: do not silently relabel')
    data = attach_labels(pd.read_csv(CACHE), labels)
    audit = []
    for mouse, requested in data.groupby('subject'):
        path = RAW_ROOT / f'{mouse}_final_data.csv'
        raw = pd.read_csv(path, usecols=['session_idx', 'session_date', 'trial_isi',
            'trial_type', 'block_type', 'warm_up', 'outcome', 'block_length',
            'n_max_same_side', 'Anti_bias'])
        raw['_raw_trial_index'] = raw.groupby('session_idx', sort=False).cumcount()
        if raw[['session_idx', 'session_date']].drop_duplicates().session_date.duplicated().any():
            raise ValueError('Ambiguous raw session dates')
        for sid, expected in requested.groupby('session_id'):
            src = raw[raw.session_date.eq(int(sid.rsplit('_', 2)[-2]))].set_index('_raw_trial_index')
            if len(src) != len(expected) or set(src.index) != set(expected.source_trial_index):
                raise ValueError(f'Incomplete cache {mouse}/{sid}')
            src = src.loc[expected.source_trial_index].reset_index(drop=True)
            verify_join(expected.reset_index(drop=True), src)
            if not np.array_equal(src.outcome, expected.outcome):
                raise ValueError(f'Outcome mismatch {sid}')
            audit.append(dict(subject=mouse, session_id=sid, n_trials=len(src),
                raw_verified=True, programmed_lengths='/'.join(map(str, sorted(src.block_length.unique()))),
                n_max_same_side='/'.join(map(str, sorted(src.n_max_same_side.unique()))),
                anti_bias='/'.join(map(str, sorted(src.Anti_bias.unique())))))
    if any(sha(p) != digest for p, digest in source_hashes.items()):
        raise RuntimeError('Input changed during raw verification')
    return labels, prepare_trials(data), pd.DataFrame(audit), source_hashes


def prepare_trials(data):
    """Keep all rows in place; derive histories before applying trial eligibility."""
    pieces = []
    for _, source in data.groupby(KEYS, sort=False):
        g = source.sort_values('source_trial_index').reset_index(drop=True).copy()
        if not np.array_equal(np.diff(g.source_trial_index), np.ones(len(g)-1)):
            raise ValueError('Raw trial cache has a gap')
        g['formal_block'] = g.true_block.ne(g.true_block.shift()).cumsum()
        g['block_uid'] = g.subject + '|' + g.session_id + '|B' + g.formal_block.astype(str)
        g['block_age'] = g.groupby('formal_block').cumcount() + 1
        g['formal_block_length'] = g.groupby('formal_block').trial_index.transform('size')
        g['block_complete'] = g.formal_block.ne(g.formal_block.iloc[-1])
        g['correct'] = g.outcome.map({'Reward': 1., 'RewardNaive': 1., 'Punish': 0., 'PunishNaive': 0.})
        g['eligible'] = (g.warm_up.eq(0) & g.model_evaluated.eq(True)
                         & g.choice_long.isin([0, 1]) & g.correct.notna()
                         & g.true_block.isin(['S', 'L']))
        if not ((g.choice_long.eq(g.category_long)).astype(float)[g.eligible].to_numpy()
                == g.correct[g.eligible].to_numpy()).all():
            raise ValueError('Choice/category does not agree with raw outcome')
        g['is_rare'] = ((g.true_block.eq('S') & g.category_long.eq(1))
                        | (g.true_block.eq('L') & g.category_long.eq(0)))
        g['duration_key'] = g.duration_s.round(3)
        if not g.duration_key.isin(DURATIONS).all():
            raise ValueError('Unexpected stimulus durations: define bins explicitly')
        # Fixed, outcome-independent difficulty bins within each category.
        g['difficulty'] = (abs(g.duration_s - 1.25) < .60).astype(int)
        g['age_band'] = np.select([g.block_age.le(15), g.block_age.ge(21)], [0, 2], default=1)
        for lag in (1, 2, 3):
            for col in ('choice_long', 'correct', 'is_rare', 'eligible', 'difficulty', 'formal_block'):
                g[f'prev{lag}_{col}'] = g[col].shift(lag)
            g[f'valid{lag}'] = (g.eligible & g[f'prev{lag}_eligible'].eq(True)
                & g.formal_block.eq(g[f'prev{lag}_formal_block']))
        g['valid2'] &= g.valid1
        g['valid3'] &= g.valid2
        # Exact preceding action run, starting only at a known interruption/boundary.
        run, before = 0, np.nan
        lengths = []
        for row in g.itertuples():
            if not row.eligible:
                run, before = 0, np.nan
            else:
                run = run + 1 if row.valid1 and row.choice_long == before else 1
                before = row.choice_long
            lengths.append(run)
        g['action_run'] = lengths
        g['previous_run'] = g.action_run.shift().clip(upper=3)
        g['repeat'] = g.choice_long.eq(g.prev1_choice_long).astype(float)
        g['against_block'] = np.where(g.true_block.eq('L'), 1-g.choice_long, g.choice_long)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def shared_support(group, conditions, strata, value='response'):
    """Pair conditions within a session, then standardize to identical cell weights.

    Cell weight is the smallest condition count (overlap population). No
    extrapolation. With two conditions, return a fixed-support block-linearized
    uncertainty estimate conditional on selected strata and overlap weights.
    """
    if group.empty:
        return None, group.iloc[:0].copy()
    g = group.dropna(subset=[*strata, 'condition', value]).copy()
    if g.empty:
        return None, g
    g['_cell'] = list(map(tuple, g[strata].to_numpy()))
    counts = g.groupby(['_cell', 'condition'], observed=True).size().unstack('condition', fill_value=0)
    counts = counts.reindex(columns=conditions, fill_value=0)
    counts = counts[counts.gt(0).all(axis=1)]
    if counts.empty:
        return None, g.iloc[:0]
    weights = counts.min(axis=1).astype(float)
    weights /= weights.sum()
    retained = g[g._cell.isin(counts.index)].copy()
    means = retained.groupby(['_cell', 'condition'], observed=True)[value].mean().unstack('condition')
    values = (means.reindex(index=weights.index, columns=conditions).mul(weights, axis=0)).sum()
    result = {f'mean_{c}': float(values[c]) for c in conditions}
    for c in conditions:
        result[f'n_{c}'] = int(retained.condition.eq(c).sum())
    result.update(n_cells=len(weights), n_retained=len(retained), n_eligible=len(group),
                  n_blocks=retained.block_uid.nunique())
    retained['cell_weight'] = retained._cell.map(weights)
    if len(conditions) == 2:
        a, b = conditions
        result['contrast'] = result[f'mean_{b}'] - result[f'mean_{a}']
        # Influence scores respect within-formal-block dependence; no trial SEM.
        cell_mu = retained.set_index(['_cell', 'condition']).index.map(means.stack().to_dict())
        cell_n = retained.set_index(['_cell', 'condition']).index.map(counts.stack().to_dict())
        signs = np.where(retained.condition.eq(b), 1., -1.)
        score = signs * retained.cell_weight.to_numpy() * (retained[value].to_numpy()-np.asarray(cell_mu)) / np.asarray(cell_n)
        block_scores = pd.Series(score).groupby(retained.block_uid.to_numpy()).sum()
        m = len(block_scores)
        result['se_block'] = float(np.sqrt(m/(m-1) * np.sum(block_scores**2))) if m > 1 else np.nan
        # Sparse cells can have zero residuals: require repeat observations in every condition/cell.
        enough = (m >= 6 and min(result[f'n_{a}'], result[f'n_{b}']) >= 8
                  and counts.ge(2).all().all() and result['se_block'] > 0)
        result['interval_supported'] = bool(enough)
        half = student_t.ppf(.975, m-1) * result['se_block'] if enough else np.nan
        result['ci_low'] = max(-1., result['contrast']-half) if enough else np.nan
        result['ci_high'] = min(1., result['contrast']+half) if enough else np.nan
    return result, retained


def summarize_events(events, conditions, strata, facet=None):
    rows, retained = [], []
    grouping = [*KEYS, 'cluster_label'] + ([facet] if facet else [])
    for key, group in events.groupby(grouping, observed=True, sort=False):
        result, keep = shared_support(group, conditions, strata)
        if result is None:
            continue
        row = dict(zip(grouping, key)); row.update(result)
        rows.append(row)
        retained.append(keep.drop(columns='_cell'))
    schema = [*grouping, *[f'mean_{c}' for c in conditions], *[f'n_{c}' for c in conditions],
              'n_cells', 'n_retained', 'n_eligible', 'n_blocks']
    if len(conditions) == 2:
        schema += ['contrast', 'se_block', 'interval_supported', 'ci_low', 'ci_high']
    return pd.DataFrame(rows, columns=schema), (pd.concat(retained, ignore_index=True) if retained else events.iloc[:0])


def build_analyses(data):
    result = {}
    # Current stimulus remains an exact physical-duration facet, not a fitted curve.
    for outcome, name in ((1, 'after_reward'), (0, 'after_error')):
        e = data[data.valid1 & data.prev1_correct.eq(outcome)].copy()
        e['condition'] = e.prev1_choice_long.astype(int)
        e['response'] = e.choice_long
        result[name] = (e, [0, 1], ['true_block', 'prev1_difficulty'], 'duration_key')
    e = data[data.valid1].copy()
    e['condition'] = e.previous_run.astype(int)
    e['response'] = e.repeat
    result['action_run'] = (e, [1, 2, 3], ['duration_key', 'true_block', 'prev1_correct', 'prev1_choice_long'], None)
    # Exactly C-R-target versus C-R-R-target (exclusive histories), same block.
    one = data.valid2 & data.prev1_is_rare.eq(True) & data.prev2_is_rare.eq(False)
    two = (data.valid3 & data.prev1_is_rare.eq(True) & data.prev2_is_rare.eq(True)
           & data.prev3_is_rare.eq(False))
    e = data[one | two].copy()
    e['condition'] = np.where(two.loc[e.index], 2, 1)
    e['response'] = e.against_block
    result['contradictory_sequence'] = (e, [1, 2],
        ['duration_key', 'true_block', 'prev1_correct', 'age_band'], None)
    # Age belongs to the RARE ANCHOR (previous trial), not target or eventual fraction.
    isolated = (data.valid3 & data.prev1_is_rare.eq(True) & data.prev2_is_rare.eq(False)
                & data.prev3_is_rare.eq(False) & data.is_rare.eq(False))
    anchor_age = data.block_age - 1
    stage = pd.Series(np.select([anchor_age.between(6, 15), anchor_age.between(21, 27)], [0, 1], default=-1), index=data.index)
    e = data[isolated & stage.ge(0)].copy()
    e['condition'] = stage.loc[e.index].astype(int)
    e['anchor_age'] = e.block_age - 1
    e['response'] = e.against_block
    # Coarse but FIXED current difficulty to keep session overlap visible; retain exact-duration sensitivity.
    result['block_age'] = (e, [0, 1], ['difficulty', 'true_block', 'prev1_correct', 'prev1_difficulty'], None)
    result['block_age_exact_stimulus'] = (e, [0, 1], ['duration_key', 'true_block', 'prev1_correct', 'prev1_difficulty'], None)
    # Paired-block sensitivity: same sequence and age bins, no cross-block composition.
    result['block_age_paired_block'] = (e, [0, 1], ['block_uid'], None)
    return result


def mouse_means(table, conditions, cluster, facet=None):
    g = table[table.cluster_label.eq(cluster)]
    cols = [f'mean_{c}' for c in conditions]
    return g.groupby(['subject']+([facet] if facet else []), observed=True)[cols].mean().reset_index()


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E8EBEF', lw=.7)
    ax.tick_params(length=3, color='#73808D', labelsize=10)
    ax.set_ylim(-.025, 1.025)
    ax.set_yticks([0, .25, .5, .75, 1], ['0', '25', '50', '75', '100'])


def shell(title, question, hypothesis, n_axes=1, width=9.8):
    fig, axes = plt.subplots(1, n_axes, figsize=(width, 7.3), squeeze=False)
    fig.subplots_adjust(left=.105, right=.965, bottom=.37, top=.75, wspace=.32)
    fig.text(.075, .95, title, fontsize=17, weight='bold', color='#233143')
    fig.text(.075, .885, question, fontsize=12, color='#233143')
    fig.text(.075, .83, hypothesis, fontsize=10.5, color='#647182')
    return fig, axes.ravel()


def footer(fig, text, support):
    fig.text(.105, .235, support, fontsize=9.2, color='#465365', va='top')
    fig.text(.105, .15, text, fontsize=9, color='#4D5B6C', va='top', linespacing=1.35)
    fig.text(.965, .018, 'Observed behavior | frozen K=3 labels | exploratory', fontsize=8,
             color='#7B8794', ha='right')


def support_text(table, cluster, total):
    g = table[table.cluster_label.eq(cluster)]
    counts = g.groupby('subject').session_id.nunique()
    return (f'Shared support: {g.session_id.nunique()}/{total} sessions  |  '
            + '   '.join(f'{m}: {counts.get(m, 0)}' for m in MICE))


def draw_psych(ax, table, cluster, subtitle):
    g = table[table.cluster_label.eq(cluster)]
    means = mouse_means(table, [0, 1], cluster, 'duration_key')
    for cond, color, label in ((0, HISTORY_COLORS[0], 'Previous short choice'),
                               (1, HISTORY_COLORS[1], 'Previous long choice')):
        for mouse in MICE:
            m = means[means.subject.eq(mouse)].set_index('duration_key')[f'mean_{cond}'].reindex(DURATIONS)
            ax.scatter(DURATIONS, m, s=15, marker=MARKERS[mouse], color=color, alpha=.40, linewidths=.5)
        v = means.groupby('duration_key')[f'mean_{cond}'].mean().reindex(DURATIONS)
        # Category stimulus ranges are separated; never imply observations in their gap.
        for sl in (slice(0, 5), slice(5, 10)):
            ax.plot(DURATIONS[sl], v.iloc[sl], color=color, marker='o', ms=4.8, lw=1.7,
                    label=label if sl.start == 0 else None)
    ax.axvspan(.8, 1.7, facecolor='#F2F4F7', zorder=-1)
    ax.set_xticks([.2, .5, .8, 1.7, 2., 2.3])
    ax.set_xlim(.12, 2.38)
    ax.set_xlabel('Current stimulus duration (s)', fontsize=11)
    ax.set_ylabel('Choose long (%)', fontsize=11)
    ax.set_title(subtitle, fontsize=12, pad=12)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    if not g.empty:
        count = g.groupby('duration_key').session_id.nunique()
        ax.text(.98, .04, f'{int(count.min())}-{int(count.max())} sessions / duration',
                transform=ax.transAxes, ha='right', fontsize=8, color='#657182')


def psych_figure(cluster, tables, total, error_only=False):
    fig, axes = shell(f'Cluster {cluster} | '+LABELS[cluster],
        'Does the previous action bias the next sensory decision?',
        'Prediction: action persistence follows the previous choice; feedback correction follows the opposite choice after errors.',
        1 if error_only else 2, 10.8 if not error_only else 9.8)
    names = [('after_error', 'After an error')] if error_only else [('after_reward', 'After reward'), ('after_error', 'After an error')]
    for ax, (name, title) in zip(axes, names):
        draw_psych(ax, tables[name], cluster, title)
    foot = ('Within-session matching: current exact duration, block direction and previous stimulus difficulty.\n'
            'Small shapes: mouse means (circle MC06, triangle MC07, square MC08); large points: equal-mouse means.\n'
            'Lines join observed means within each stimulus category; gray region contains no stimuli. No fitted curves or inferential bands.')
    footer(fig, foot, '  |  '.join(f'{title}: '+support_text(tables[name], cluster, total).replace('Shared support: ', '') for name, title in names))
    return fig


def point_figure(cluster, name, table, conditions, total):
    if name == 'action_run':
        question = 'Does a longer action run increase the probability of repetition?'
        prediction = 'Prediction: repetition rises from one preceding choice to longer action runs.'
        xticklabels = ['1 choice', '2 choices', '3+ choices']
        ylabel = 'Repeat previous choice (%)'
        xlabel = 'Consecutive preceding choices toward the same category'
        method = 'Within-session matching: exact current duration, block direction, last choice and last outcome.\nRun length counts consecutive valid actions; omissions, warmup and formal block switches break runs.\nOnly sessions / strata containing all three run groups contribute; earlier outcomes are not fully matched.'
    elif name == 'contradictory_sequence':
        question = 'Do additional rare categories shift the next choice?'
        prediction = 'Prediction: two consecutive rare categories shift choices further against the current block than one.'
        xticklabels = ['1 rare category', '2 rare categories']
        ylabel = 'Choose against current block (%)'
        xlabel = 'Consecutive preceding categories opposing the block'
        method = 'Exclusive sequences: common-rare-target versus common-rare-rare-target, within one formal block.\nWithin-session matching: exact current duration, block direction, last outcome and coarse block age.\nConditional on the last response / feedback; earlier actions, outcomes and stimulus difficulty may differ.'
    else:
        question = 'After an isolated rare event, do early- and late-block choices differ?'
        prediction = 'Prediction: following the rare event, choices oppose the current block more often at later ages.'
        xticklabels = ['Early\n6-15 trials', 'Late\n21-27 trials']
        ylabel = 'Choose against current block (%)'
        xlabel = 'Absolute formal-block age of the rare event'
        method = 'Sequence: common-common-rare-common(target); valid consecutive trials in one formal block.\nWithin-session matching: current difficulty, block direction, rare outcome and rare difficulty.\nAge-associated difference only; block survival and earlier histories remain possible explanations.'
    fig, (ax,) = shell(f'Cluster {cluster} | '+LABELS[cluster], question, prediction)
    means = mouse_means(table, conditions, cluster)
    xx = np.arange(len(conditions))
    for mouse in MICE:
        m = means[means.subject.eq(mouse)]
        if m.empty:
            continue
        yy = m[[f'mean_{c}' for c in conditions]].iloc[0].to_numpy(float)
        ax.plot(xx, yy, marker=MARKERS[mouse], ms=6, lw=1, alpha=.6, color=COLORS[cluster], label=mouse)
    if not means.empty:
        yy = means[[f'mean_{c}' for c in conditions]].mean().to_numpy()
        ax.plot(xx, yy, marker='o', ms=9, lw=2.5, color=COLORS[cluster], label='Equal-mouse mean')
    else:
        ax.text(.5, .5, 'Insufficient shared support', transform=ax.transAxes, ha='center')
    ax.set_xticks(xx, xticklabels)
    ax.set_xlim(-.35, len(conditions)-.65)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    style(ax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=2)
    g = table[table.cluster_label.eq(cluster)]
    counts = '   '.join(f'{label.split(chr(10))[0]}: {int(g[f"n_{c}"].sum())} trials' for c, label in zip(conditions, xticklabels))
    footer(fig, method+'\nSmall shapes: mouse means; large points: equal-mouse mean. Mouse spread is descriptive, not a confidence interval.',
           support_text(table, cluster, total)+'\n'+counts)
    return fig


def comparison_figure(tables):
    fig, axes = shell('Previous-choice associations | compare the three clusters',
        'After an error, does the next choice follow the action or the feedback?',
        'Orange above blue: action-consistent association. Orange below blue: feedback/category-consistent association.',
        3, 14.4)
    for cluster, ax in enumerate(axes, 1):
        draw_psych(ax, tables['after_error'], cluster, f'Cluster {cluster}')
    footer(fig, 'Same matching and axes. Small circle / triangle / square: MC06 / MC07 / MC08; large points: equal-mouse means.\n'
           'Current exact duration, block direction and previous stimulus difficulty matched within each session.\n'
           'Previous category and feedback are not separated by this comparison. Same-session checks are descriptive.',
           'Frozen cluster sizes: C1 = 11 sessions   |   C2 = 25 sessions   |   C3 = 56 sessions')
    return fig


def session_figure(labels, age):
    fig, axes = plt.subplots(3, 1, figsize=(13.2, 10), sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.16, top=.80, hspace=.56)
    fig.text(.075, .95, 'Which sessions show an age-associated choice shift?', fontsize=18, weight='bold', color='#233143')
    fig.text(.075, .905, 'Late minus early probability of choosing against the block, after an isolated rare event', fontsize=12)
    fig.text(.075, .865, f'{len(age)}/92 descriptive estimates; {int(age.interval_supported.sum())} meet the interval-support rule. No temporal-awareness classification.', fontsize=10, color='#647182')
    for ax, mouse in zip(axes, MICE):
        cohort = labels[labels.subject.eq(mouse)].sort_values('test_rank').copy()
        summary = cohort.merge(age, on=[*KEYS, 'cluster_label'], how='left', validate='one_to_one')
        for position, row in enumerate(summary.itertuples()):
            color = COLORS[row.cluster_label]
            if pd.isna(row.contrast):
                ax.scatter(position, -116, marker='x', color='#A6AEB7', s=22, lw=1)
                continue
            supported = row.interval_supported is True or row.interval_supported == True
            if supported:
                ax.plot([position]*2, [100*row.ci_low, 100*row.ci_high], color=color, lw=1, alpha=.7)
            ax.scatter(position, row.contrast*100, marker='o', s=36,
                       facecolors=color if supported else 'white', edgecolors=color, lw=1.3, zorder=4)
        ax.axhline(0, color='#627083', lw=.9)
        ax.axhline(-108, color='#E0E4EA', lw=.7)
        ax.set_ylim(-123, 108)
        ax.set_yticks([-100, -50, 0, 50, 100])
        ax.set_ylabel('Late - early\n(percentage points)', fontsize=10)
        ax.set_title(mouse, loc='left', fontsize=12, weight='bold', pad=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E8EBEF', lw=.7)
        xx = np.arange(0, len(cohort), 4)
        dates = [sid.rsplit('_', 2)[-2] for sid in cohort.session_id]
        ax.set_xticks(xx, [f'{dates[i][4:6]}/{dates[i][6:]}' for i in xx], fontsize=9)
        ax.set_xlabel('Session date (month/day, 2026); equally spaced sessions', fontsize=9)
        ax.set_xlim(-1, len(cohort))
        ax.text(1, 1.08, f'{cohort.session_id.nunique()} sessions, chronological order', transform=ax.transAxes,
                ha='right', fontsize=9, color='#647182')
    handles = [Line2D([], [], marker='o', color=COLORS[c], lw=0, label=f'Cluster {c}') for c in (1, 2, 3)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.975, .85), ncol=3, frameon=False, fontsize=10)
    fig.text(.095, .095, 'Open circles: limited support; no interval. Filled circles: approximate 95% block-cluster intervals. Gray x below plot: no shared support.\n'
        'Within-session strata: current difficulty, block direction, rare outcome and rare difficulty. Early = 6-15; late = 21-27 (rare age).\n'
        'Intervals require >=6 contributing blocks, >=8 retained events per age, and >=2 events in each condition / stratum.\n'
        'Fixed-support linearization; bounds clipped to the possible [-100, 100] range. Sparse session estimates are exploratory.',
        fontsize=9, color='#647182', va='top', linespacing=1.5)
    return fig


def render_figures(out, labels, tables):
    """Render this analysis's figures without changing inputs or estimates."""
    sizes = labels.groupby('cluster_label').size().to_dict()
    figures = [('cross_cluster/01_after_error_comparison', comparison_figure(tables))]
    figures += [('cluster_1/01_action_after_reward_and_error', psych_figure(1, tables, sizes[1])),
                ('cluster_1/02_action_run_length', point_figure(1, 'action_run', tables['action_run'], [1, 2, 3], sizes[1])),
                ('cluster_2/01_feedback_after_error', psych_figure(2, tables, sizes[2], True)),
                ('cluster_2/02_contradictory_sequence', point_figure(2, 'contradictory_sequence', tables['contradictory_sequence'], [1, 2], sizes[2])),
                ('cluster_3/01_sensory_and_history', psych_figure(3, tables, sizes[3])),
                ('cluster_3/02_block_age', point_figure(3, 'block_age', tables['block_age'], [0, 1], sizes[3])),
                ('cross_cluster/02_session_block_age', session_figure(labels, tables['block_age']))]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        with PdfPages(out/'hypothesis_followup_all_clusters.pdf') as pdf:
            for name, fig in figures:
                fig.savefig(out/f'{name}.png', dpi=190, facecolor='white')
                fig.savefig(out/f'{name}.svg', facecolor='white')
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)
    return len(figures)


def run(out=OUT):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing follow-up: {out}')
    labels, data, audit, source_hashes = load_inputs()
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('cluster_1', 'cluster_2', 'cluster_3', 'cross_cluster', 'tables'):
        (out/folder).mkdir(exist_ok=True)
    protected = {str(p): sha(p) for p in PARENT.rglob('*') if p.is_file() and out not in p.parents}
    audit.to_csv(out/'tables/raw_session_audit.csv', index=False)
    labels.to_csv(out/'tables/session_clusters_used.csv', index=False)
    data.to_csv(out/'tables/annotated_trials.csv', index=False)
    definitions = build_analyses(data)
    tables, supports = {}, []
    for name, (events, conditions, strata, facet) in definitions.items():
        table, retained = summarize_events(events, conditions, strata, facet)
        tables[name] = table
        table.to_csv(out/f'tables/{name}_session_estimates.csv', index=False)
        retained.to_csv(out/f'tables/{name}_retained_events.csv', index=False)
        events.to_csv(out/f'tables/{name}_eligible_events.csv', index=False)
        for cluster in (1, 2, 3):
            e, r = events[events.cluster_label.eq(cluster)], retained[retained.cluster_label.eq(cluster)]
            supports.append(dict(analysis=name, cluster=cluster, n_eligible=len(e), n_retained=len(r),
                sessions_eligible=e.session_id.nunique(), sessions_retained=r.session_id.nunique(),
                mice_retained=r.subject.nunique(), strata=' + '.join(strata), facet=facet))
    pd.DataFrame(supports).to_csv(out/'tables/support_summary.csv', index=False)
    sizes = labels.groupby('cluster_label').size().to_dict()
    n_figures = render_figures(out, labels, tables)
    results = {}
    for name, table in tables.items():
        cond = definitions[name][1]
        if definitions[name][3] or table.empty:
            continue
        results[name] = {}
        for cluster in (1, 2, 3):
            mm = mouse_means(table, cond, cluster)
            results[name][cluster] = {c: float(mm[f'mean_{c}'].mean()) for c in cond}
    manifest = dict(seed=None, cluster_sizes=sizes, n_figures=n_figures, results=results,
        source_hashes=source_hashes,
        sources_unchanged=all(sha(p) == digest for p, digest in source_hashes.items()),
        n_age_sessions=int(tables['block_age'].session_id.nunique()),
        n_age_intervals=int(tables['block_age'].interval_supported.sum()),
        protected_files_unchanged=all(sha(p) == h for p, h in protected.items()))
    if not manifest['protected_files_unchanged'] or not manifest['sources_unchanged']:
        raise RuntimeError('An existing result changed')
    (out/'analysis_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write_readme(out, tables, manifest)
    print(json.dumps(manifest, indent=2))
    print(pd.DataFrame(supports).to_string(index=False))


def write_readme(out, tables, manifest):
    text = '''# Hypothesis-driven follow-up of seven-feature K3 clusters

Eight simple observed-behavior figures, grouped by frozen cluster. Start with
`hypothesis_followup_all_clusters.pdf`; individual PNG and editable vector SVG
files are in `cluster_1/`, `cluster_2/`, `cluster_3/`, and `cross_cluster/`.

## Questions and figures

- Cluster 1 / 01: Does the previous action bias choices even after an error?
- Cluster 1 / 02: Does repetition increase with the preceding action run (1, 2, 3+)?
- Cluster 2 / 01: Does the next choice shift toward feedback after an error?
- Cluster 2 / 02: Do two contradictory categories shift choices more than one?
- Cluster 3 / 01: Do sensory responses remain similar after different previous choices?
- Cluster 3 / 02: Does the same isolated rare sequence have a larger late-block association?
- Cross-cluster / 01: Identical after-error psychometric axes for all three clusters.
- Cross-cluster / 02: Session-level late-minus-early contrasts, with limited support visible.

## Population, histories and matching

All 92 frozen K3 test sessions: C1=11, C2=25, C3=56; MC06/MC07/MC08.
Read the archived complete canonical trial cache at
`version1/sliding_window_final/reaction_time_trials.csv`, not RT-eligible trials.
Verify every session against the original complete raw CSV, including trial index,
stimulus, category, block, warmup and outcome. Raw source identities/hashes are in
the audit and manifest. Do not use model-winner subsets.

Formal block IDs and absolute age are constructed on ALL rows before exclusions.
Only non-warmup, evaluated, completed-choice Reward/Punish trials in biased blocks
contribute. Histories require consecutive eligible rows in the same formal block;
omissions, invalid feedback, warmup and boundaries break them. No adjusted switch
alignment or retrospective fractional age. No decision model refitting.

Psychometrics use each of the 10 exact observed stimulus durations, with shared
block direction and preceding difficulty within each session and duration.
Previous difficulty is fixed: hard when abs(duration - 1.25) < 0.60 seconds; otherwise
easy. These are two coarse difficulty strata, not exact prior-stimulus matching.
The stimulus-free 0.8-1.7 second gap is shaded; lines do not bridge it.

Run comparisons use exactly 1, exactly 2, or at least 3 preceding same-category
ACTIONS. Match exact current duration, block direction, last choice and last correctness.
Earlier outcomes and previous stimulus difficulty are not fully matched.
Sequence comparisons use exclusive C-R-target vs C-R-R-target, where C is a
block-common CATEGORY and R a rare category. Match exact current duration,
block direction, last correctness and target-age band (<=15,16-20,>=21).
This is conditional on last response/feedback, not the total effect of the sequence.

Age comparisons use C-C-R-C(target), with all four completed trials inside one
formal block. Early and late refer to R's absolute age: 6-15 and 21-27 trials.
Actual anchor support ends at 27 because a following target is required.
No target choice/outcome is used to select the sequence (beyond completed choice).
Primary age matching uses current difficulty, block direction, rare correctness
and rare difficulty. The two fixed current difficulty bins allow within-session
support but leave residual differences in exact duration and earlier histories.
`block_age_exact_stimulus_*` tables use exact target duration as a sensitivity;
`block_age_paired_block_*` tables pair ages within the same formal block without
stimulus/outcome matching. Neither sensitivity silently replaces the main result.
Final partial blocks contribute when a complete eligible sequence exists; age
analyses remain subject to survival and block-composition differences.

For every session / plotted facet, retain only covariate cells represented in ALL
compared conditions. Weight cells in proportion to the smallest condition count
(overlap population); use the SAME weights for every condition. Average session
estimates equally within mouse, then mice equally. No extrapolation into missing
cells. Tables retain per-cell weights, condition counts and support losses.
Psychometric mouse/session support can vary with duration; see support tables.
Conditioning on previous correctness makes previous choice identify previous
category, so feedback-driven updating is not isolated from preceding-category
attraction by the after-error plot. These are associations consistent with the
proposed hypotheses, not unique mechanistic signatures.

## Display and uncertainty

Small circle/triangle/square markers identify MC06/MC07/MC08. Large points are
means across available mice; thin mouse traces show descriptive heterogeneity,
not inferential confidence bands. Figures contain no trial-level SEM or p-values.

Session age contrasts use fixed-common-support block-cluster linearized intervals:
weighted cell residual influence scores are summed within formal blocks; the
variance uses m/(m-1), and the critical value is t(m-1). Intervals condition on
selected cells and their overlap weights. Only show intervals with >=6 retained
formal blocks, >=8 retained events per condition and >=2 events per condition in
EVERY cell. Approximate interval bounds are clipped to the feasible [-1,1] range.
Open points have limited support and no interval. Gray crosses below the plotting
range indicate no shared-support estimate. These are NOT temporal-awareness labels.
Block independence and linearization are approximations; fixed support selection
and sparse events limit inference. Adjacent fitted windows and same-session
cluster discovery prevent independent-validation claims. This is a descriptive
follow-up, not proof of distinct neural mechanisms, habit, or block-length learning.

## Reproduction

`python -m post_analysis.session_clustering.hypothesis_followup`

The runner refuses to overwrite a nonempty follow-up folder. To make a new version,
call `run(new_output_path)` from Python. Original clustering outputs and raw inputs
are hash checked and never edited. CSVs store eligible events, retained events,
within-session estimates, the annotated complete timeline, and all source audits.
'''
    text += f'\nCurrent run: {manifest["n_age_sessions"]}/92 sessions have an age contrast; '
    text += f'{manifest["n_age_intervals"]} meet the interval support rule.\n'
    (out/'README.md').write_text(text)
    for c in (1, 2, 3):
        (out/f'cluster_{c}/README.md').write_text(f'# Cluster {c}: {LABELS[c]}\n\n'
            'See the parent README for questions, matching, support and interpretation.\n'
            'Each PNG has an editable vector SVG companion. These are descriptive follow-ups.\n')


if __name__ == '__main__':
    run()
