"""Extend the frozen seven-feature K3 report without refitting or reclustering.

Run: python -m post_analysis.session_clustering.extend_k3_reaction_time
"""
from itertools import combinations
import json
import shutil

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

from . import pattern_rbf_seven_features as base, behavior

def welch_pairs(data, **metadata):
    rows = []
    for a, b in combinations([1, 2, 3], 2):
        x = data.loc[data.cluster_label.eq(a), 'latency_s'].dropna().to_numpy()
        y = data.loc[data.cluster_label.eq(b), 'latency_s'].dropna().to_numpy()
        result = ttest_ind(x, y, equal_var=False) if min(len(x), len(y)) >= 2 else None
        rows.append(dict(**metadata, cluster_a=a, cluster_b=b, n_a=len(x), n_b=len(y),
                         delta_s=x.mean()-y.mean() if len(x) and len(y) else np.nan,
                         t=result.statistic if result else np.nan, df=result.df if result else np.nan,
                         p_raw=result.pvalue if result else np.nan))
    return rows


def correct_tests(rows):
    tests = pd.DataFrame(rows)
    ok = tests.p_raw.notna()
    tests.loc[ok, 'p_holm'] = multipletests(tests.loc[ok, 'p_raw'], method='holm')[1]
    return tests


def trial_comparisons(tagged):
    summaries, tests = [], []
    for condition, mask in base.rt_masks(tagged).items():
        data = tagged[mask]
        for c, g in data.groupby('cluster_label'):
            summaries.append(dict(condition=condition, cluster=int(c), mean=g.latency_s.mean(),
                                  sem=g.latency_s.sem(), sd=g.latency_s.std(), n_trials=len(g)))
        tests.extend(welch_pairs(data, condition=condition))
    return pd.DataFrame(summaries), correct_tests(tests)


def duration_comparisons(tagged):
    # Shared stimulus-duration boundaries avoid comparing different duration ranges across clusters.
    pieces, records = [], []
    for condition, mask in base.rt_masks(tagged).items():
        data = tagged[mask]
        for cat, g in data.groupby('category_long'):
            g = g.copy()
            edges = np.unique(np.quantile(g.duration_s, np.linspace(0, 1, 5)))
            if len(edges) < 2:
                g['isi_bin'] = 0
            else:
                g['isi_bin'] = pd.cut(g.duration_s, edges, labels=False, include_lowest=True).astype(int)
            for bin_id, h in g.groupby('isi_bin'):
                records.extend(welch_pairs(h, condition=condition, category_long=int(cat), isi_bin=int(bin_id),
                                          duration_min=h.duration_s.min(), duration_max=h.duration_s.max()))
            pieces.append(g.assign(condition=condition))
    binned = pd.concat(pieces, ignore_index=True)
    summary = binned.groupby(['condition', 'category_long', 'isi_bin', 'cluster_label']).agg(
        isi_mean=('duration_s','mean'), latency_mean=('latency_s','mean'),
        latency_sem=('latency_s','sem'), n_trials=('latency_s','size')).reset_index()
    return summary, correct_tests(records)


def significance(p):
    if not np.isfinite(p): return 'NA'
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'


def duration_page(summary, tests):
    fig, axes = plt.subplots(4, 4, figsize=(22, 23))
    fig.subplots_adjust(left=.065, right=.98, top=.89, bottom=.16, hspace=1.15, wspace=.32)
    for row, (row_title, conditions) in enumerate(base.RT_ROWS):
        fig.text(.065, axes[row, 0].get_position().y1+.025, row_title, fontsize=14, weight='bold')
        for ax, (condition, title) in zip(axes[row], conditions):
            g = summary[summary.condition.eq(condition)]
            for c in [1, 2, 3]:
                h = g[g.cluster_label.eq(c)].sort_values(['category_long','isi_bin'])
                ax.errorbar(h.isi_mean, h.latency_mean, yerr=h.latency_sem, fmt='-o', ms=4,
                            color=base.PALETTE[c-1], lw=1.5, capsize=2)
            ax.axvline(1.25, color='.6', ls=':', lw=1)
            ax.set_title(title, fontsize=12); ax.set_ylabel('Reaction time (s)')
            ax.set_xlabel('Stimulus duration (s)'); ax.set_xlim(.1, 2.4); behavior.style(ax)
            keys = g[['category_long','isi_bin']].drop_duplicates().sort_values(['category_long','isi_bin'])
            text, headers = [], []
            for r in keys.itertuples(): headers.append(f"{'S' if r.category_long == 0 else 'L'}{r.isi_bin+1}")
            for a, b in combinations([1, 2, 3], 2):
                tt = tests[tests.condition.eq(condition) & tests.cluster_a.eq(a) & tests.cluster_b.eq(b)]
                text.append([significance(tt[tt.category_long.eq(r.category_long) & tt.isi_bin.eq(r.isi_bin)].p_holm.iloc[0])
                             for r in keys.itertuples()])
            table = ax.table(cellText=text, rowLabels=['C1–C2','C1–C3','C2–C3'], colLabels=headers,
                             cellLoc='center', bbox=[0, -.59, 1, .34])
            table.auto_set_font_size(False); table.set_fontsize(8)
            for cell in table.get_celld().values(): cell.set_linewidth(.3); cell.set_edgecolor('.8')
    fig.suptitle('K=3: reaction time by stimulus duration — pooled trials', x=.065, ha='left', y=.975, fontsize=22, weight='bold')
    fig.text(.065, .94, 'Mean ± SEM across individual trials; shared duration bins across clusters within each condition. No mouse/session adjustment.', fontsize=12)
    handles = [plt.Line2D([], [], color=base.PALETTE[c-1], label=f'Cluster {c}') for c in [1,2,3]]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98,.963), ncol=3, frameon=False)
    fig.text(.065, .025, 'Tables: two-sided Welch tests for each cluster pair in each duration bin; Holm correction across all binwise comparisons on this page.\n'
             'S1–S4 / L1–L4 = short/long stimulus quartile bins; identical durations stay together. *** p<.001; ** p<.01; * p<.05; ns p≥.05 (adjusted).\n'
             'Trials treated as independent observations; repeated measures are not modeled. Original RT exclusions retained; no latency trimming.', fontsize=11, color='.3')
    return fig


def mean_sem_page(summary, tests):
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    fig.subplots_adjust(left=.06, right=.98, top=.88, bottom=.11, hspace=.85, wspace=.35)
    for ax, condition in zip(axes.flat, summary.condition.drop_duplicates()):
        g = summary[summary.condition.eq(condition)].set_index('cluster').reindex([1, 2, 3])
        for c in [1, 2, 3]:
            ax.errorbar(c, g.loc[c, 'mean'], yerr=g.loc[c, 'sem'], fmt='o', color=base.PALETTE[c-1], capsize=7, lw=2)
            ax.annotate(f"n={int(g.loc[c, 'n_trials'])}", (c, g.loc[c, 'mean']), xytext=(9, 0),
                        textcoords='offset points', fontsize=8)
        ax.set_xticks([1, 2, 3], ['C1', 'C2', 'C3']); ax.set_xlim(.5, 3.65)
        ax.set_ylabel('Reaction time (s)'); ax.set_title(condition, fontsize=12)
        ax.margins(y=.20)
        tt = tests[tests.condition.eq(condition)]
        label = '   '.join(f"{int(r.cluster_a)}–{int(r.cluster_b)}: {r.p_holm:.2g}" for r in tt.itertuples())
        ax.text(.5, -.25, 'Holm p  '+label, transform=ax.transAxes, ha='center', fontsize=8)
        behavior.style(ax)
    fig.suptitle('K=3: mean reaction time ± SEM — pooled trials', x=.06, ha='left', y=.97, fontsize=21, weight='bold')
    fig.text(.06, .925, 'Points = pooled trial mean; whiskers = ±1 SEM across individual trials; n = trials. Each condition pools its stimulus durations.', fontsize=12)
    fig.text(.06, .045, 'Pairwise two-sided Welch t-tests allow unequal sample sizes and unequal variances. Holm correction across all 48 tests on this page.\n'
             'All eligible trials are pooled within each condition and cluster; no mouse stratification or session averaging.\n'
             'Trials are treated as independent observations; repeated measures are not modeled. SEM = sample SD / square root of trial count.', fontsize=10, color='.3')
    return fig


def transition_events(tagged):
    """Mirror behavior.collect_events, retaining trial identity and RT/outcome on the original timeline."""
    rows, event = [], 0
    for (subject, sid), t in tagged.groupby(base.KEYS, sort=False):
        sl = t.sort_values('trial_index'); sl = sl[sl.true_block.isin(['S', 'L'])].reset_index(drop=True)
        blk = sl.true_block.to_numpy()
        rare = ((sl.true_block.eq('S') & sl.category_long.eq(1)) | (sl.true_block.eq('L') & sl.category_long.eq(0))).to_numpy()
        run = sl.true_block.ne(sl.true_block.shift()).cumsum()
        early = rare & ((sl.groupby(run).cumcount()+1).to_numpy() <= behavior.FIRST_BLOCK_TRIALS_FORCED_COMMON)
        for formal in np.flatnonzero((sl.true_block.ne(sl.true_block.shift()) & sl.true_block.shift().notna()).to_numpy()):
            align = behavior.adjusted_start(rare, blk, formal)
            pb, ce = formal-1, formal
            while pb > 0 and blk[pb-1] == blk[formal-1]: pb -= 1
            while ce+1 < len(sl) and blk[ce+1] == blk[formal]: ce += 1
            event += 1
            for rel in behavior.TRIAL_RANGE:
                i = align+rel
                if i < pb or i > ce or early[i]: continue
                r = sl.iloc[i]
                rows.append(dict(subject=subject, session_id=sid, cluster=int(r.cluster_label), event_id=event,
                    kind='S2L' if blk[formal] == 'L' else 'L2S', rel=rel, category=int(r.category_long),
                    valid=bool(r.latency_eligible), mouse=r.latency_s if r.latency_eligible else np.nan,
                    outcome=r.outcome_strict, trial_index=r.trial_index, bridge=False))
    return pd.DataFrame(rows)


def transition_page(events):
    fig, axes = plt.subplots(3, 4, figsize=(24, 16), sharex=True, sharey=False)
    fig.subplots_adjust(left=.055, right=.985, top=.88, bottom=.10, hspace=.42, wspace=.18)
    records = []
    for row, outcome in enumerate(['All outcomes', 'Reward', 'Punish']):
        for col, (kind, cat) in enumerate([('S2L', 0), ('S2L', 1), ('L2S', 0), ('L2S', 1)]):
            ax = axes[row, col]
            g = events[events.kind.eq(kind) & events.category.eq(cat)]
            if outcome != 'All outcomes': g = g[g.outcome.eq(outcome)]
            title = f"{'Short → Long' if kind == 'S2L' else 'Long → Short'} · {'short' if cat == 0 else 'long'} trials\n{outcome}"
            behavior.event_panel(ax, g, 3, title, behavior.TRIAL_RANGE, transition=True)
            ax.set_ylabel('Reaction time (s)')
            legend = ax.get_legend()
            if legend is not None: legend.remove()
            counts = [g[g.cluster.eq(c) & g.valid].event_id.nunique() for c in [1, 2, 3]]
            ax.text(.02, .97, 'Events C1 / C2 / C3: ' + ' / '.join(map(str, counts)),
                    transform=ax.transAxes, va='top', fontsize=8)
            for c in [1, 2, 3]:
                cg = behavior.add_bridge(g[g.cluster.eq(c) & g.valid])
                agg = cg.groupby('rel').mouse.agg(['mean', 'sem', 'size']).reset_index()
                records.append(agg.assign(cluster=c, kind=kind, category=cat, outcome=outcome))
    for ax in axes.flat:
        ax.autoscale(axis='y')
        ax.margins(y=.12)
    handles = [plt.Line2D([], [], color=base.PALETTE[c-1], label=f'Cluster {c}') for c in [1, 2, 3]]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.985, .955), ncol=3, frameon=False)
    fig.suptitle('K=3: reaction time aligned to block transitions', x=.055, ha='left', fontsize=22, weight='bold', y=.975)
    fig.text(.055, .925, 'Same adjusted transitions and stimulus-category panels as page 5. Rows split by current-trial outcome; all three clusters are overlaid.', fontsize=12)
    fig.text(.055, .035, 'Mean ± trial/event SEM, matching page 5 weighting; automatic y limits separately for each panel. Repeated/overlapping events are descriptive.\n'
             'Warm-up, negative-lick and invalid latency trials excluded; no trimming. Open markers at −1 copy the previous eligible matching trial when −1 is absent, as on page 5.', fontsize=11, color='.3')
    return fig, pd.concat(records, ignore_index=True)


def main():
    out = base.OUT; dest = out/'04_clusters_K3'; tables = out/'05_cluster_tables'
    report = dest/'K3_report.pdf'; backup = dest/'K3_report_before_rt_extension.pdf'
    if not backup.exists(): shutil.copy2(report, backup)
    labels = pd.read_csv(out/'selected_session_clusters.csv'); labels = labels[labels.K.eq(3)]
    trials, eligible = base.behavior_trials()
    tagged = behavior.attach_labels(eligible, labels)
    summary, tests = trial_comparisons(tagged)
    bins, bin_tests = duration_comparisons(tagged)
    events = transition_events(behavior.attach_labels(trials, labels))
    figs = []
    figs.append(('6_reaction_time', duration_page(bins, bin_tests)))
    figs.append(('7_reaction_time_mean_sem', mean_sem_page(summary, tests)))
    fig, aligned = transition_page(events); figs.append(('8_reaction_time_transitions', fig))
    appendix = dest/'K3_rt_extension_pages.pdf'
    with plt.rc_context(base.EDITABLE_PDF), PdfPages(appendix) as pdf:
        for name, fig in figs:
            pdf.savefig(fig); fig.savefig(dest/f'K3_{name}.png', dpi=100); plt.close(fig)
    writer = PdfWriter()
    original = PdfReader(backup)
    for page in original.pages[:5]: writer.add_page(page)
    writer.append(appendix)
    temporary = dest/'K3_report.tmp.pdf'
    with temporary.open('wb') as f: writer.write(f)
    temporary.replace(report)
    for name, frame in [('condition_summary', summary), ('pairwise_tests', tests), ('binwise_pairwise_tests', bin_tests),
                        ('transition_events', events), ('transition_summary', aligned), ('duration_binned_extended', bins)]:
        frame.to_csv(tables/f'K3_rt_{name}.csv', index=False)
    pages = pd.read_csv(tables/'report_pages.csv')
    pages = pages[~(pages.K.eq(3) & pages.page.ge(7))]
    pages = pd.concat([pages, pd.DataFrame([dict(K=3, page=7, content='reaction_time_mean_sem'),
                                          dict(K=3, page=8, content='reaction_time_transitions')])], ignore_index=True)
    pages.sort_values(['K', 'page']).to_csv(tables/'report_pages.csv', index=False)
    (tables/'K3_rt_extension_methods.json').write_text(json.dumps(dict(
        tests='Two-sided unpaired Welch t-tests across individual trial RTs; no subject/session adjustment',
        correction='Holm: 48 condition-level tests (page 7); separate family of all binwise tests (page 6)',
        whiskers='SEM across individual trial RTs', bins='Shared pooled-duration quartile edges per condition and stimulus category; ties stay together',
        assumption='Trials treated as independent; repeated measures not modeled',
        source=str(base.OLD/'reaction_time_trials.csv'), original_report=str(backup)), indent=2))
    print(report); print(summary.to_string(index=False)); print(tests.to_string(index=False))


if __name__ == '__main__':
    main()
