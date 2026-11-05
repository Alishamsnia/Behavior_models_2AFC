"""K4 pages 5-8 in the reference report layout, using fixed Ward labels.

No previous clustering modules are imported. Only observed choices and RTs
enter these descriptive figures; no clustering or normalization is performed.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from . import behavior as bh, plots
from .core import KEYS

K = 4
PAIRS = tuple(combinations(range(1, K+1), 2))
# Display grouping follows the reference's duration page. Mean-RT conditions
# follow its mean/SEM page (expectation x outcome precedes block x outcome).
RT_ROWS = (
    ('Block', ('All trials', 'Neutral block', 'Short block', 'Long block')),
    ('Stimulus expectation and outcome', ('Common trials', 'Rare trials', 'Reward - biased', 'Punish - biased')),
    ('Block x outcome', ('Short - reward', 'Short - punish', 'Long - reward', 'Long - punish')),
    ('Expectation x outcome', ('Common - reward', 'Common - punish', 'Rare - reward', 'Rare - punish')),
)
MEAN_CONDITIONS = (*RT_ROWS[0][1], *RT_ROWS[1][1], *RT_ROWS[3][1], *RT_ROWS[2][1])
TITLES = {'Common trials': 'Common stimulus', 'Rare trials': 'Rare stimulus',
          'Reward - biased': 'Reward (biased blocks)', 'Punish - biased': 'Punish (biased blocks)'}


def holm_adjust(frame):
    """One explicit family of finite two-sided p values, preserving row order."""
    frame = frame.copy()
    frame['p_holm'] = np.nan
    valid = frame.loc[np.isfinite(frame.p_raw), 'p_raw'].sort_values(kind='stable')
    m = len(valid)
    if m:
        adjusted = np.minimum(1., np.maximum.accumulate(valid.to_numpy() * (m-np.arange(m))))
        frame.loc[valid.index, 'p_holm'] = adjusted
    return frame


def welch_pairs(frame, **metadata):
    rows = []
    for a, b in PAIRS:
        x = frame.loc[frame.cluster_label.eq(a), 'latency_s'].dropna().to_numpy(float)
        y = frame.loc[frame.cluster_label.eq(b), 'latency_s'].dropna().to_numpy(float)
        result = ttest_ind(x, y, equal_var=False) if min(len(x), len(y)) >= 2 else None
        rows.append(dict(**metadata, cluster_a=a, cluster_b=b, n_a=len(x), n_b=len(y),
                         delta_s=float(x.mean()-y.mean()) if len(x) and len(y) else np.nan,
                         t=float(result.statistic) if result is not None else np.nan,
                         df=float(getattr(result, 'df', np.nan)) if result is not None else np.nan,
                         p_raw=float(result.pvalue) if result is not None else np.nan))
    return rows


def rt_statistics(tagged):
    """Reference trial-pooled means, shared bins, Welch tests and two Holm families."""
    eligible = tagged[tagged.latency_eligible].copy()
    summary, tests, duration, bin_tests = [], [], [], []
    masks = bh.rt_masks(eligible)
    for condition in MEAN_CONDITIONS:
        frame = eligible[masks[condition]]
        for c in range(1, K+1):
            x = frame[frame.cluster_label.eq(c)]
            summary.append(dict(condition=condition, cluster_label=c, mean=x.latency_s.mean(),
                                sd=x.latency_s.std(ddof=1), sem=x.latency_s.sem(ddof=1),
                                n_trials=len(x), n_sessions=len(x[KEYS].drop_duplicates()), n_mice=x.subject.nunique()))
        tests.extend(welch_pairs(frame, condition=condition))
        # Shared boundaries are computed from all four groups before slicing.
        # Identical physical durations always remain together.
        binned = bh.duration_bins(frame, shared_edges=True)
        for bin_number, group in binned.groupby('duration_bin', sort=True):
            category = int(bin_number)//4
            isi_bin = int(bin_number)%4
            metadata = dict(condition=condition, category_long=category, isi_bin=isi_bin,
                            duration_min=group.duration_s.min(), duration_max=group.duration_s.max())
            bin_tests.extend(welch_pairs(group, **metadata))
            for c in range(1, K+1):
                x = group[group.cluster_label.eq(c)]
                if x.empty:
                    continue
                duration.append(dict(**metadata, cluster_label=c, isi_mean=x.duration_s.mean(),
                                     latency_mean=x.latency_s.mean(), latency_sem=x.latency_s.sem(ddof=1),
                                     n_trials=len(x), n_sessions=len(x[KEYS].drop_duplicates())))
    return pd.DataFrame(summary), holm_adjust(pd.DataFrame(tests)), pd.DataFrame(duration), holm_adjust(pd.DataFrame(bin_tests))


def significance(value):
    if not np.isfinite(value):
        return 'NA'
    return '***' if value < .001 else '**' if value < .01 else '*' if value < .05 else 'ns'


def axis_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(False)


def psych_panel(ax, frame, title, records, cluster=None):
    colors = plots.COLORS
    if cluster is None:
        traces = [(c, frame[frame.cluster_label.eq(c)], '-', colors[c-1],
                   f'Cluster {c} ({len(frame[frame.cluster_label.eq(c)][KEYS].drop_duplicates())} sessions, '
                   f'{int(frame.cluster_label.eq(c).sum())} trials)', colors[c-1]) for c in range(1, K+1)]
    else:
        traces = [(cluster, frame[frame.true_block.eq(block)], line, colors[cluster-1],
                   f'{label} ({int(frame.true_block.eq(block).sum())} trials)', face)
                  for block, label, line, face in [('S', 'Short block', '--', 'white'),
                                                 ('L', 'Long block', '-', colors[cluster-1])]]
    for c, trials, line, color, label, face in traces:
        if trials.empty:
            continue
        stats = bh.binned(trials, 'choice_long')
        ax.errorbar(stats.duration_s, stats['mean'], yerr=stats['sem'], fmt='o', color=color, mfc=face,
                    ms=4.5, capsize=2, lw=1.1, label=label)
        params = bh.psychometric(trials)
        if params is not None:
            x = np.linspace(.15, 2.35, 200)
            ax.plot(x, bh.curve(x, *params), color=color, lw=1.8, ls=line)
        records.append(stats.assign(K=K, cluster_label=c, panel=title, trace=label))
    ax.axhline(.5, color='.6', ls='--', lw=.8)
    ax.axvline(1.25, color='.6', ls=':', lw=.8)
    ax.set_ylim(-.03, 1.03)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Stimulus duration (s)')
    ax.set_ylabel('P(choose long)')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    axis_style(ax)


def event_panel(ax, events, title, value='correct', transition=True, show_legend=True):
    summaries = []
    xx = np.arange(-20, 20) if transition else np.arange(-2, 3)
    for c in range(1, K+1):
        original = events[events.cluster_label.eq(c)].dropna(subset=[value])
        g = bh.bridge_events(original, value) if transition else original
        if g.empty:
            continue
        stats = g.groupby('rel')[value].agg(['mean', 'sem', 'size']).reindex(xx)
        label = f'Cluster {c} ({len(original[KEYS].drop_duplicates())} sessions, {original.event_id.nunique()} events)'
        color = plots.COLORS[c-1]
        ax.plot(xx, stats['mean'], '-o', color=color, lw=1.5, ms=2.5, label=label)
        ax.fill_between(xx, stats['mean']-stats['sem'], stats['mean']+stats['sem'], color=color, alpha=.15, lw=0)
        if transition and g.bridge.any() and np.isfinite(stats.loc[-1, 'mean']):
            ax.scatter(-1, stats.loc[-1, 'mean'], s=30, facecolors='white', edgecolors=color, linewidths=1.3, zorder=5)
        summaries.append(stats.reset_index(names='rel').assign(K=K, cluster_label=c, panel=title))
    ax.axvline(0, color='black', ls='--', lw=.9)
    if transition:
        ax.axvspan(0, 19.5, color='gray', alpha=.08, lw=0)
        ax.set_xlim(-20.5, 19.5)
    else:
        ax.set_xticks([-2, -1, 0, 1, 2], ['-2', '-1', 'rare', '+1', '+2'])
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('P(correct)' if value == 'correct' else 'Reaction time (s)')
    ax.set_xlabel('Trial relative to adjusted block transition' if transition else 'Trial relative to rare trial')
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=7.2, loc='lower left')
    axis_style(ax)
    return summaries


def choice_page(tagged, transitions, rare_events):
    """Reference's 13-panel page: 3 pooled + 4 within-cluster + 4 switch + 2 rare."""
    height = 23.45
    fig = plt.figure(figsize=(20, height))
    panel_height, top, gap = 3.2, 1.85, 1.1
    def axis(row, left, width):
        bottom = height-top-panel_height-row*(panel_height+gap)
        return fig.add_axes([left, bottom/height, width, panel_height/height])
    valid = tagged[tagged.choice_eligible]
    records = []
    for left, (title, frame) in zip([.12, .44, .76], [('All blocks', valid),
            ('Short block', valid[valid.true_block.eq('S')]), ('Long block', valid[valid.true_block.eq('L')])]):
        psych_panel(axis(0, left, .16), frame, title, records)
    for c, left in zip(range(1, K+1), [.13, .335, .54, .745]):
        psych_panel(axis(1, left, .16), valid[valid.cluster_label.eq(c)],
                    f'Cluster {c}: short vs long block', records, cluster=c)
    for row, kind in [(2, 'S2L'), (3, 'L2S')]:
        for cat, left in [(0, .148), (1, .63)]:
            title = ('Short to Long' if kind == 'S2L' else 'Long to Short') + (' - short trials' if cat == 0 else ' - long trials')
            selected = transitions[transitions.kind.eq(kind) & transitions.category_long.eq(cat)]
            records.extend(event_panel(axis(row, left, .264), selected, title))
    for kind, left, title in [('rare_long', .283, 'Rare long trial (in short block)'),
                              ('rare_short', .604, 'Rare short trial (in long block)')]:
        records.extend(event_panel(axis(4, left, .16), rare_events[rare_events.kind.eq(kind)], title, transition=False))
    labels = tagged[KEYS+['cluster_label']].drop_duplicates()
    counts = labels.cluster_label.value_counts()
    fig.suptitle('Choice Behavior | Zero-anchored Ward | K=4', fontsize=16, weight='bold', y=.985)
    fig.text(.5, .962, 'Psychometrics - block transitions - rare trials     |     ' + '   '.join(
        f'Cluster {c}: {counts.get(c, 0)} sessions' for c in range(1, K+1)), ha='center', fontsize=11)
    mice = set(labels.subject)
    partial = [f'C{c}: '+', '.join(sorted(set(group.subject))) for c, group in labels.groupby('cluster_label') if set(group.subject) != mice]
    if partial:
        fig.text(.5, .945, 'Mouse support: ' + '; '.join(partial) + '. Group means reflect the available mice.', ha='center', fontsize=10, color='#a32626')
    fig.text(.06, .025, 'Observed held-out test-session choices, pooled across MC06/MC07/MC08. Error bars/bands: trial/event SEM. '
             'Open transition markers copy the previous eligible matching-category trial. Exploratory partitions; cluster numbers are local to K.', fontsize=9, color='.3')
    return fig, pd.concat(records, ignore_index=True)


def duration_page(summary, tests):
    """Page 6: same 4x4 reference grid and binwise significance tables, six pairs."""
    fig, axes = plt.subplots(4, 4, figsize=(22, 26))
    fig.subplots_adjust(left=.065, right=.98, top=.875, bottom=.135, hspace=1.65, wspace=.32)
    for row, (heading, conditions) in enumerate(RT_ROWS):
        fig.text(.065, axes[row, 0].get_position().y1+.022, heading, fontsize=14, weight='bold')
        for ax, condition in zip(axes[row], conditions):
            g = summary[summary.condition.eq(condition)]
            for c in range(1, K+1):
                h = g[g.cluster_label.eq(c)].sort_values(['category_long', 'isi_bin'])
                ax.errorbar(h.isi_mean, h.latency_mean, yerr=h.latency_sem, fmt='-o', ms=4,
                            color=plots.COLORS[c-1], lw=1.5, capsize=2)
            ax.axvline(1.25, color='.6', ls=':', lw=1)
            ax.set(title=TITLES.get(condition, condition.replace(' - ', ' / ')), ylabel='Reaction time (s)', xlabel='Stimulus duration (s)', xlim=(.1, 2.4))
            axis_style(ax)
            bins = g[['category_long', 'isi_bin']].drop_duplicates().sort_values(['category_long', 'isi_bin'])
            headers = [('S' if r.category_long == 0 else 'L')+str(r.isi_bin+1) for r in bins.itertuples()]
            cells = []
            for a, b in PAIRS:
                group_tests = tests[tests.condition.eq(condition) & tests.cluster_a.eq(a) & tests.cluster_b.eq(b)]
                values = []
                for r in bins.itertuples():
                    match = group_tests[group_tests.category_long.eq(r.category_long) & group_tests.isi_bin.eq(r.isi_bin)]
                    values.append(significance(float(match.p_holm.iloc[0])) if len(match) else 'NA')
                cells.append(values)
            if headers:
                table = ax.table(cellText=cells, rowLabels=[f'C{a}-C{b}' for a, b in PAIRS], colLabels=headers,
                                 cellLoc='center', bbox=[0, -.94, 1, .65])
                table.auto_set_font_size(False)
                table.set_fontsize(7.2)
                for cell in table.get_celld().values():
                    cell.set_linewidth(.3)
                    cell.set_edgecolor('.8')
    fig.suptitle('K=4: reaction time by stimulus duration - pooled trials', x=.065, ha='left', y=.975, fontsize=22, weight='bold')
    fig.text(.065, .94, 'Mean +/- SEM across individual trials; shared duration bins across clusters within each condition. No mouse/session adjustment.', fontsize=12)
    fig.legend(handles=[plt.Line2D([], [], color=plots.COLORS[c-1], marker='o', label=f'Cluster {c}') for c in range(1, K+1)],
               loc='upper right', bbox_to_anchor=(.98, .962), ncol=K, frameon=False)
    n = int(tests.p_holm.notna().sum())
    fig.text(.065, .025, f'Tables: two-sided Welch tests for all six cluster pairs in each duration bin; Holm correction across {n} estimable binwise tests on this page.\n'
             'S1-S4 / L1-L4 = short/long stimulus quartile bins; identical durations stay together. *** p<.001; ** p<.01; * p<.05; ns p>=.05 (adjusted).\n'
             'Trials treated as independent observations; repeated measures are not modeled. Original RT exclusions retained; no latency trimming.', fontsize=11, color='.3')
    return fig


def mean_page(summary, tests, error='sem'):
    """Page 7: reference 16-condition means, sample counts and six adjusted p values."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.subplots_adjust(left=.06, right=.98, top=.88, bottom=.11, hspace=1.05, wspace=.35)
    for ax, condition in zip(axes.flat, MEAN_CONDITIONS):
        g = summary[summary.condition.eq(condition)].set_index('cluster_label').reindex(range(1, K+1))
        for c in range(1, K+1):
            if pd.notna(g.loc[c, 'mean']):
                ax.errorbar(c, g.loc[c, 'mean'], yerr=g.loc[c, error], fmt='o', color=plots.COLORS[c-1], capsize=6, lw=2)
                ax.annotate(f'n={int(g.loc[c, "n_trials"])}', (c, g.loc[c, 'mean']), xytext=(6, 0), textcoords='offset points', fontsize=7)
        ax.set_xticks(range(1, K+1), [f'C{c}' for c in range(1, K+1)])
        ax.set_xlim(.5, 4.8)
        ax.set_ylabel('Reaction time (s)')
        ax.set_title(condition.replace(' - ', ' / '), fontsize=12)
        ax.margins(y=.20)
        t = tests[tests.condition.eq(condition)].set_index(['cluster_a', 'cluster_b'])
        lines = []
        for pairs in (PAIRS[:3], PAIRS[3:]):
            lines.append('   '.join(f'{a}-{b}: {t.loc[(a,b), "p_holm"]:.2g}' for a, b in pairs))
        ax.text(.5, -.27, 'Holm p  '+lines[0]+'\n'+lines[1], transform=ax.transAxes, ha='center', va='top', fontsize=7.5, linespacing=1.5)
        axis_style(ax)
    fig.suptitle(f'K=4: mean reaction time +/- {error.upper()} - pooled trials', x=.06, ha='left', y=.97, fontsize=21, weight='bold')
    fig.text(.06, .925, f'Points = pooled trial mean; whiskers = +/-1 {error.upper()} across individual trials; n = trials. Each condition pools its stimulus durations.', fontsize=12)
    n = int(tests.p_holm.notna().sum())
    fig.text(.06, .035, f'Pairwise two-sided Welch t-tests allow unequal sample sizes and variances. Holm correction across all {n} estimable tests on this page.\n'
             'All eligible trials are pooled within each condition and cluster; no mouse stratification or session averaging.\n'
             'Trials are treated as independent; repeated measures are not modeled. SEM = sample SD / square root of trial count.', fontsize=10, color='.3')
    return fig


def transition_page(events):
    """Page 8: reference outcome rows x transition/category columns, four clusters."""
    fig, axes = plt.subplots(3, 4, figsize=(24, 16), sharex=True)
    fig.subplots_adjust(left=.055, right=.985, top=.88, bottom=.10, hspace=.42, wspace=.18)
    records = []
    for row, outcome in enumerate(['All outcomes', 'Reward', 'Punish']):
        for col, (kind, category) in enumerate([('S2L', 0), ('S2L', 1), ('L2S', 0), ('L2S', 1)]):
            selected = events[events.kind.eq(kind) & events.category_long.eq(category)]
            if outcome != 'All outcomes':
                selected = selected[selected.outcome.eq(outcome)]
            title = ('Short to Long' if kind == 'S2L' else 'Long to Short') + (' - short trials' if category == 0 else ' - long trials') + '\n'+outcome
            stats = event_panel(axes[row, col], selected, title, value='latency_s', show_legend=False)
            records.extend([s.assign(kind=kind, category_long=category, outcome=outcome) for s in stats])
            counts = [selected[selected.cluster_label.eq(c) & selected.rt_valid].event_id.nunique() for c in range(1, K+1)]
            axes[row, col].text(.02, .97, 'Events C1 / C2 / C3 / C4: '+' / '.join(map(str, counts)), transform=axes[row, col].transAxes, va='top', fontsize=8)
            axes[row, col].margins(y=.12)
    fig.legend(handles=[plt.Line2D([], [], color=plots.COLORS[c-1], label=f'Cluster {c}') for c in range(1, K+1)],
               loc='upper right', bbox_to_anchor=(.985, .955), ncol=K, frameon=False)
    fig.suptitle('K=4: reaction time aligned to block transitions', x=.055, ha='left', fontsize=22, weight='bold', y=.975)
    fig.text(.055, .925, 'Same adjusted transitions and stimulus-category panels as page 5. Rows split by current-trial outcome; all four clusters are overlaid.', fontsize=12)
    fig.text(.055, .035, 'Mean +/- trial/event SEM; automatic y limits separately for each panel. Repeated/overlapping events are descriptive.\n'
             'Warm-up, negative-lick and invalid latency trials excluded; no trimming. Open markers at -1 copy the previous eligible matching trial when -1 is absent.', fontsize=11, color='.3')
    return fig, pd.concat(records, ignore_index=True)


def render(tagged, transitions, rare_events, dest, tables, pdf):
    """Write exactly pages 5,6,7,8 plus SD and machine-readable statistics."""
    summary, tests, duration, bin_tests = rt_statistics(tagged)
    for name, frame in [('condition_summary', summary), ('pairwise_tests', tests), ('duration_binned_extended', duration),
                        ('binwise_pairwise_tests', bin_tests), ('transition_events', transitions)]:
        frame.to_csv(tables/f'K4_rt_{name}.csv', index=False)
    fig, choice = choice_page(tagged, transitions, rare_events)
    choice.to_csv(tables/'K4_choice_behavior_plotted.csv', index=False)
    plots.save(fig, dest/'K4_5_choice_behavior', pdf)
    plots.save(duration_page(duration, bin_tests), dest/'K4_6_reaction_time', pdf)
    plots.save(mean_page(summary, tests), dest/'K4_7_reaction_time_mean_sem', pdf)
    plots.save(mean_page(summary, tests, error='sd'), dest/'K4_7_reaction_time_mean_sd')
    fig, aligned = transition_page(transitions)
    aligned.to_csv(tables/'K4_rt_transition_summary.csv', index=False)
    plots.save(fig, dest/'K4_8_reaction_time_transitions', pdf)
    from .pipeline import json_write
    json_write(tables/'K4_rt_extension_methods.json', dict(
        K=K, cluster_pairs=[list(pair) for pair in PAIRS],
        page_6='Pooled-trial RT mean and SEM at shared duration bins, with binwise Holm significance tables',
        page_7='Pooled-trial RT mean and SEM by condition; all six cluster pairs with Holm p values',
        page_8='Transition-aligned RT mean and event SEM, split by current outcome',
        tests='Two-sided Welch t-tests across individual eligible trial RTs; no subject/session adjustment',
        correction=dict(condition_level_tests=int(tests.p_holm.notna().sum()),
                        binwise_tests=int(bin_tests.p_holm.notna().sum()), families='Separate condition-level and binwise Holm families'),
        bins='Shared pooled-duration quartile edges within condition and stimulus category; identical durations stay together',
        assumptions='Trials treated as independent; repeated measures are not modeled; exploratory descriptive comparisons',
        clustering='Existing zero-anchored Ward K4 labels; no refitting or reclustering'))
    return ['choice_behavior', 'reaction_time_shared_bins_with_statistics', 'reaction_time_mean_sem_with_statistics', 'reaction_time_transitions']
