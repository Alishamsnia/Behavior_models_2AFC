"""Reproduce the supplied E/F sequence plots in fixed all-model RBF K=3 sessions.

Run: python -m post_analysis.session_clustering.sequence_performance
Raw analysis functions are imported without executing its notebook. No fits or
cluster assignments are changed. E deliberately enforces the three commons in
the supplied example; the raw Python collector only requires a common pair.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .behavior import attach_labels
from .features import LEGACY_OUT
from .latency import RAW_ROOT, verify_join
from .pattern_exploration import OUT as CLUSTER_OUT

OUT = CLUSTER_OUT / 'sequence_performance'
SUBJECTS = ('MC06', 'MC07', 'MC08')
CONTEXTS = ('All biased blocks', 'Short block', 'Long block')
E_SPECS = (
    ('rewarded_common', 'After rewarded common', '#009E73'),
    ('punished_common', 'After punished common', '#D55E00'),
)
F_SPECS = (
    ('pre', 'Common before rare (−1)', '#0072B2'),
    ('rare', 'Rare (0)', '#D55E00'),
    ('post', 'Common after rare (+1)', '#009E73'),
)


def source_module():
    path = RAW_ROOT / 'block_sequence_epoch_analysis.py'
    spec = importlib.util.spec_from_file_location('raw_block_sequence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adjusted_block_start(sdf, formal_switch_idx):
    """Raw notebook's alignment: include the previous block's terminal rare run."""
    last = formal_switch_idx - 1
    if last < 0 or not bool(sdf.loc[last, 'is_rare']):
        return formal_switch_idx
    previous_block = sdf.loc[last, 'block_type']
    start = last
    while start > 0:
        row = sdf.loc[start - 1]
        if row.block_type != previous_block or not bool(row.is_rare):
            break
        start -= 1
    return start


def load_trials(labels):
    """Use the complete canonical test-trial cache, verified again against raw CSVs.

    This is NOT the RT-filtered population: negative/missing licks and no-choice
    trials remain in place. Non-choice outcomes have missing performance and
    cannot silently turn separated trials into adjacent sequences.
    """
    data = pd.read_csv(LEGACY_OUT / 'reaction_time_trials.csv')
    data = attach_labels(data, labels)
    audits = []
    for subject, requested in data.groupby('subject', sort=False):
        raw = pd.read_csv(RAW_ROOT / f'{subject}_final_data.csv', usecols=[
            'session_idx', 'session_date', 'trial_isi', 'trial_type',
            'block_type', 'warm_up', 'outcome'])
        raw['_raw_trial_index'] = raw.groupby('session_idx', sort=False).cumcount()
        dates = raw[['session_idx', 'session_date']].drop_duplicates()
        if dates.session_date.duplicated().any():
            raise ValueError(f'Ambiguous raw session date for {subject}')
        for sid, expected in requested.groupby('session_id', sort=False):
            date = int(sid.rsplit('_', 2)[-2])
            source = raw[raw.session_date.eq(date)].set_index('_raw_trial_index')
            if len(source) != len(expected) or set(source.index) != set(expected.source_trial_index):
                raise ValueError(f'Incomplete test session: {subject}/{sid}')
            source = source.loc[expected.source_trial_index].reset_index(drop=True)
            verify_join(expected.reset_index(drop=True), source)
            if not np.array_equal(expected.outcome.to_numpy(), source.outcome.to_numpy()):
                raise ValueError(f'Cached and raw outcomes differ: {subject}/{sid}')
            audits.append(dict(subject=subject, session_id=sid, n_trials=len(expected),
                               cluster_label=int(expected.cluster_label.iloc[0]), raw_verified=True))
    data['block_type'] = data.true_block.map({'N': 0, 'S': 1, 'L': 2})
    data['trial_type'] = data.category_long + 1
    data['_source_order'] = data.source_trial_index
    data['change_session'] = data.subject + '|' + data.session_id
    data['correct'] = data.outcome.map({'Reward': 1., 'RewardNaive': 1.,
                                      'Punish': 0., 'PunishNaive': 0.})
    data['is_rare'] = data.block_type.isin([1, 2]) & data.trial_type.ne(data.block_type)
    data = data[data.warm_up.eq(0)].sort_values(['change_session', '_source_order']).copy()
    # The original collector removes neutral trials before splitting blocks. In
    # this cohort neutral trials must therefore occur only before biased trials.
    # Refuse internal removals rather than joining two unrelated block fragments.
    for sid, session in data.groupby('change_session', sort=False):
        biased = session[session.block_type.isin([1, 2])]
        if not np.array_equal(np.diff(biased._source_order), np.ones(max(0, len(biased)-1))):
            raise ValueError(f'Internal gap among biased trials in {sid}')
    return data, pd.DataFrame(audits)


def collect_strict_common_events(data, source):
    """C → C(reward/punish) → C; measure the FINAL common's correctness."""
    rows = []
    for session, sdf, number, start, stop in source._adjusted_blocks(data, adjusted_block_start):
        for target in range(start + 2, stop):
            train = sdf.loc[target-2:target]
            if train.is_rare.any():
                continue
            previous, final = sdf.loc[target-1, 'correct'], sdf.loc[target, 'correct']
            if pd.isna(previous) or pd.isna(final):
                continue
            trace = 'rewarded_common' if previous == 1 else 'punished_common'
            rows.append(dict(
                subject=sdf.loc[target, 'subject'], session=session,
                block_id=f'{session}_B{number}', sequence_id=f'{session}_B{number}_E{target-start}',
                block_context='Short block' if sdf.loc[target, 'block_type'] == 1 else 'Long block',
                trace=trace, performance=float(final), anchor_position=target-start,
                train_start_position=target-start-2, train_end_position=target-start,
                block_length=stop-start))
    return pd.DataFrame(rows)


def tag_events(events, data):
    lookup = data[['change_session', 'cluster_label']].drop_duplicates()
    return events.merge(lookup.rename(columns={'change_session': 'session'}),
                        on='session', validate='many_to_one')


def summarize(events, panel):
    """Original pooled-event means and event SEM, without treating windows as independent."""
    rows = []
    for cluster in (1, 2, 3):
        for subject in (*SUBJECTS, 'All subjects pooled'):
            selected = events[events.cluster_label.eq(cluster)]
            if subject in SUBJECTS:
                selected = selected[selected.subject.eq(subject)]
            for context in CONTEXTS:
                subset = selected if context == CONTEXTS[0] else selected[selected.block_context.eq(context)]
                for center, group in subset.groupby('window_center'):
                    for trace, _, _ in E_SPECS if panel == 'E' else F_SPECS:
                        values = (group.loc[group.trace.eq(trace), 'performance'] if panel == 'E'
                                  else group[trace]).dropna()
                        if values.empty:
                            continue
                        units = group[group.trace.eq(trace)] if panel == 'E' else group
                        rows.append(dict(cluster_label=cluster, subject=subject, block_context=context,
                                         panel=panel, trace=trace, window_center=center,
                                         mean=float(values.mean()),
                                         sem=float(values.sem()) if len(values) > 1 else 0.,
                                         n_sequences=len(values), n_sessions=units.session.nunique(),
                                         n_subjects=units.subject.nunique()))
    return pd.DataFrame(rows)


def plot_axis(ax, summary, cluster, subject, context, panel, overview=False):
    selected = summary[summary.cluster_label.eq(cluster) & summary.subject.eq(subject)
                       & summary.block_context.eq(context) & summary.panel.eq(panel)]
    for name, _, color in E_SPECS if panel == 'E' else F_SPECS:
        trace = selected[selected.trace.eq(name)].set_index('window_center').reindex(range(4, 26, 3))
        ax.errorbar(trace.index, trace['mean'], yerr=trace['sem'], color=color,
                    marker='o', markersize=2.5 if overview else 3.4, linewidth=1.1,
                    capsize=1.8, elinewidth=.65)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(3, 26)
    ax.set_xticks([4, 10, 16, 22, 25])
    ax.tick_params(labelsize=7 if overview else 8)
    ax.margins(y=.15)
    if selected.empty:
        ax.text(.5, .5, 'No qualifying sequences', transform=ax.transAxes, ha='center', fontsize=8)
    else:
        low, high = int(selected.n_sequences.min()), int(selected.n_sequences.max())
        ax.text(.02, .02, f'n/window: {low}–{high}', transform=ax.transAxes, fontsize=6.5, color='.4')


def legends(fig, y, fontsize):
    for x, specs, title in ((.28, E_SPECS, 'E: C → C(reward/punish) → C*'),
                             (.76, F_SPECS, 'F: C → C* → R* → C*')):
        handles = [Line2D([0], [0], color=color, marker='o', markersize=3, label=label)
                   for _, label, color in specs]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(x, y),
                   frameon=False, ncol=1, fontsize=fontsize, title=title,
                   title_fontsize=fontsize+1)


def make_figures(summary, labels, out):
    rows = (*SUBJECTS, 'All subjects pooled')
    footer = ('7-trial windows; step 3; adjusted block alignment; final session block omitted. '
              'Mean ± event SEM; pooled row pools sequences across mice.\n'
              'C = common; R = rare; * = scored trial. Overlapping windows reuse sequences; '
              'curves are descriptive. Y axes autoscale.')
    with PdfPages(out/'rbf_sequence_overview.pdf') as pdf, PdfPages(out/'rbf_sequence_by_cluster.pdf') as detail:
        for cluster in (1, 2, 3):
            counts = labels[labels.cluster_label.eq(cluster)].subject.value_counts()
            total = int(counts.sum())
            fig, axes = plt.subplots(4, 6, figsize=(21, 12))
            fig.subplots_adjust(left=.075, right=.985, bottom=.085, top=.79, hspace=.48, wspace=.40)
            fig.suptitle(f'RBF cluster {cluster} · {total} test sessions', fontsize=16, y=.98)
            legends(fig, .945, 8)
            for col, context in enumerate(CONTEXTS):
                midpoint = (axes[0, col*2].get_position().x0 + axes[0, col*2+1].get_position().x1)/2
                fig.text(midpoint, .825, context, ha='center', fontsize=12, weight='bold')
                for row, subject in enumerate(rows):
                    for offset, panel in enumerate(('E', 'F')):
                        ax = axes[row, 2*col+offset]
                        plot_axis(ax, summary, cluster, subject, context, panel, overview=True)
                        if row == 0:
                            ax.set_title(f'{panel}  ·  '+('Outcome history' if panel == 'E' else 'Rare sequence'), fontsize=8)
                        if row == 3:
                            ax.set_xlabel('Window center within adjusted block', fontsize=7)
            for row, subject in enumerate(rows):
                name = subject if subject in SUBJECTS else 'All mice\npooled'
                label = f'{name}\n{int(counts.get(subject, total))} sessions'
                axes[row, 0].set_ylabel('Performance', fontsize=8)
                fig.text(.043, (axes[row, 0].get_position().y0+axes[row, 0].get_position().y1)/2,
                         label, fontsize=8, va='center', ha='right')
            fig.text(.075, .018, footer, fontsize=8)
            pdf.savefig(fig)
            fig.savefig(out/f'cluster_{cluster}_overview.png', dpi=110)
            plt.close(fig)
            # Three pages PER cluster: the same E/F comparison for each context.
            for context in CONTEXTS:
                fig, axes = plt.subplots(4, 2, figsize=(12, 13))
                fig.subplots_adjust(left=.12, right=.98, bottom=.09, top=.81, hspace=.48, wspace=.24)
                fig.suptitle(f'RBF cluster {cluster} · {context} · {total} test sessions', fontsize=15, y=.98)
                legends(fig, .947, 9)
                for row, subject in enumerate(rows):
                    for col, panel in enumerate(('E', 'F')):
                        ax = axes[row, col]
                        plot_axis(ax, summary, cluster, subject, context, panel)
                        if col == 0:
                            ax.set_ylabel(f'{subject} ({int(counts.get(subject, total))} sessions)\nPerformance', fontsize=9)
                        if row == 3:
                            ax.set_xlabel('Seven-trial window center within adjusted block', fontsize=9)
                fig.text(.12, .025, footer.replace('alignment;', 'alignment;\n'), fontsize=8)
                detail.savefig(fig)
                plt.close(fig)
    # These are the identical detailed pages, separated for convenient review.
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(out/'rbf_sequence_by_cluster.pdf')
    for cluster in (1, 2, 3):
        writer = PdfWriter()
        for page in reader.pages[(cluster-1)*3:cluster*3]:
            writer.add_page(page)
        with (out/f'rbf_cluster_{cluster}_sequence_performance.pdf').open('wb') as handle:
            writer.write(handle)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    label_path = CLUSTER_OUT/'selected_session_clusters.csv'
    original_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
    all_labels = pd.read_csv(label_path)
    labels = all_labels[all_labels.variant.eq('rbf_reference') & all_labels.K.eq(3)].copy()
    if len(labels) != 92 or labels.groupby('cluster_label').size().to_dict() != {1: 24, 2: 34, 3: 34}:
        raise ValueError('Unexpected preferred RBF cluster population')
    data, audit = load_trials(labels)
    source = source_module()
    e = collect_strict_common_events(data, source)
    f = source.collect_sequence_events(data, adjusted_block_start)
    e_windows = tag_events(source.collect_complete_train_sliding_events(e), data)
    f_windows = tag_events(source.collect_complete_train_sliding_events(f), data)
    summary = pd.concat([summarize(e_windows, 'E'), summarize(f_windows, 'F')], ignore_index=True)
    audit.to_csv(OUT/'session_input_audit.csv', index=False)
    e_windows.to_csv(OUT/'common_outcome_window_events.csv', index=False)
    f_windows.to_csv(OUT/'rare_sequence_window_events.csv', index=False)
    summary.to_csv(OUT/'window_summary.csv', index=False)
    labels.to_csv(OUT/'session_clusters_used.csv', index=False)
    make_figures(summary, labels, OUT)
    if original_hash != hashlib.sha256(label_path.read_bytes()).hexdigest():
        raise AssertionError('Cluster labels changed')
    method = dict(
        variant='rbf_reference', K=3, session_counts={1: 24, 2: 34, 3: 34},
        selected_label_file_sha256=original_hash,
        raw_source=str(RAW_ROOT/'block_sequence_epoch_analysis.py'),
        source_sha256=hashlib.sha256((RAW_ROOT/'block_sequence_epoch_analysis.py').read_bytes()).hexdigest(),
        window_size=source.WINDOW_SIZE, window_step=source.WINDOW_STEP,
        window_centers=list(range(4, 26, 3)), max_window_center=source.MAX_WINDOW_CENTER,
        panel_E='Common/common/common; middle common rewarded vs punished; score final common. '
                'First common is context only; its outcome is unconstrained. This matches supplied E '
                'sequence; raw collect_previous_outcome_common_events only requires TWO commons.',
        panel_F='Common/common/rare/common; score second common, rare, following common; '
                'all three scored outcomes must be valid, as in source collector.',
        performance='Reward/RewardNaive=1; Punish/PunishNaive=0; other outcomes missing, kept in trial order.',
        exclusions='Warm-up removed; neutral trials excluded; final adjusted block of each session omitted '
                   'as in source. Complete train AND complete window must lie within the same adjusted block.',
        alignment='Move formal boundary backward to first trial in previous terminal rare run; '
                  'preserve original trial common/rare labels as in source. Context is anchor trial formal block.',
        pooling='Within every plotted point, pool qualifying events; SEM across events (ddof=1). '
                'Final row pools all mice events, NOT equal-mouse means. For n=1 SEM=0, as in source. '
                'Overlapping windows reuse events; no independent-window tests or inferential claims.',
        population='Complete held-out test sessions corresponding to fixed RBF labels; all trials '
                   'before RT screening; cached canonical trial identity reverified against raw CSVs.',
        axes='Autoscaled y axes, independently by panel, following user preference. Compare numeric values.',
        outputs='Overview 3 pages (one per cluster, all/short/long paired E/F). '
                'Detailed 9 pages (three per cluster, all then short then long).')
    (OUT/'methods.json').write_text(json.dumps(method, indent=2)+'\n')
    (OUT/'README.md').write_text(
        '# Sequence performance in fixed RBF clusters\n\n'
        'Run `python -m post_analysis.session_clustering.sequence_performance`.\n\n'
        'The overview has one page per cluster; the detailed PDF has three pages per cluster '
        '(all biased, short, long). Every page includes MC06, MC07, MC08, then pooled subjects.\n\n'
        'E matches the supplied **three-common** sequence, conditioning the middle common outcome '
        'and scoring the final common. The current raw Python function uses only a pair of commons; '
        'that looser definition was not substituted silently. F uses the source four-trial sequence.\n\n'
        'Seven-trial windows step by three; centers are 4, 7, 10, 13, 16, 19, 22, 25. '
        'Complete windows and sequences stay inside one adjusted block. The original analysis omits '
        'each session\'s final adjusted block. This is retained.\n\n'
        'All 92 target test sessions are included with unchanged labels. No RT filters are applied. '
        'Non-choice trials stay in position and have missing performance. Final-row means and error bars '
        'are pooled-event means and event SEM, as in the raw pooled analysis; they are not mouse-balanced '
        'estimates or population-level confidence intervals. Sample sizes per point are exported in '
        '`window_summary.csv`; small-n points are descriptive. Overlapping windows share sequences. '
        'Y axes autoscale independently. See methods.json and session_input_audit.csv for details.\n')
    print(json.dumps(dict(output=str(OUT), sessions=len(audit),
                          E_sequences=e.sequence_id.nunique(), F_sequences=f.sequence_id.nunique(),
                          overview_pages=3, detailed_pages=9), indent=2))


if __name__ == '__main__':
    main()
