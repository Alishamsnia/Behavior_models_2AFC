"""Mouse-only pooled behaviour and reaction time: cluster overlays for K=2 and K=3.

Uses observed choices only, with the existing pooled quartile bins, logistic
psychometric summaries, event alignment, bridge markers and trial/event SEM.
No model predictions, decompositions or difference panels are loaded or drawn.
Reaction-time pages follow the source notebook's session-weighted ISI summaries.
"""
from __future__ import annotations

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from post_analysis.expanded_model_comparison.action_history_disentangled import (
    assign_bins, bin_stats, fit_data, logist, adjusted_start, add_bridge,
    FIRST_BLOCK_TRIALS_FORCED_COMMON, TRIAL_RANGE, RARE_RANGE,
)
from .features import OUT, test_sessions

COLORS = ['#4477AA', '#EE6677', '#228833']
VISUALIZATION = 'Mouse-only cluster overlays: K=2/K=3 choice behaviour and K=2/K=3 reaction time by ISI (four pages).'
WEIGHTING = 'Choice behaviour: pooled trial/event SEM. Reaction time: mean +/- SEM across within-session means.'

KEYS = ['subject', 'session_id']


def attach_labels(trials, labels):
    """Join by mouse AND session; missing, duplicated or extra session labels are errors."""
    if labels.duplicated(KEYS).any():
        raise ValueError('Duplicate cluster labels for a mouse/session')
    ids = trials[KEYS].drop_duplicates()
    check = ids.merge(labels[KEYS], on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not check._merge.eq('both').all():
        raise ValueError('Pooled trials and cluster labels do not match by mouse/session')
    return trials.merge(labels[KEYS + ['cluster_label']], on=KEYS, how='left', validate='many_to_one')


def cluster_frames(trials, labels, k):
    if set(labels.cluster_label.unique()) != set(range(1, k+1)):
        raise ValueError(f'Expected cluster labels 1..{k}')
    tagged = attach_labels(trials, labels)
    for c in range(1, k+1):
        yield c, tagged[tagged.cluster_label.eq(c)].copy()


def collect_events(trials):
    trans, rare = [], []
    ev_id = 0
    for (subj, sid), t in trials.groupby(KEYS, sort=False):
        t = t.sort_values("trial_index")
        cluster = int(t.cluster_label.iloc[0])
        sl = t[t.true_block.isin(["S", "L"])].reset_index(drop=True)
        if sl.empty:
            continue
        block_run = sl.true_block.ne(sl.true_block.shift()).cumsum()
        in_block = (sl.groupby(block_run).cumcount() + 1).to_numpy()
        is_rare = ((sl.true_block.eq("S") & sl.category_long.eq(1)) |
                   (sl.true_block.eq("L") & sl.category_long.eq(0))).to_numpy()
        early_rare = is_rare & (in_block <= FIRST_BLOCK_TRIALS_FORCED_COMMON)
        blk = sl.true_block.to_numpy()
        valid = (sl.model_evaluated.astype(bool) & sl.choice_long.isin([0, 1])).to_numpy()
        correct = (sl.choice_long == sl.category_long).to_numpy(float)
        cat = sl.category_long.to_numpy(int)

        def row(i, kind, rel):
            return dict(subject=subj, session_id=sid, cluster=cluster, event_id=ev_id, kind=kind, rel=rel,
                        category=int(cat[i]), valid=bool(valid[i]), mouse=correct[i] if valid[i] else np.nan,
                        rare=float(is_rare[i]), bridge=False)
        change = (sl.true_block.ne(sl.true_block.shift()) & sl.true_block.shift().notna()).to_numpy()
        for formal in np.flatnonzero(change):
            prev, cur = blk[formal - 1], blk[formal]
            kind = "S2L" if (prev, cur) == ("S", "L") else "L2S"
            align = adjusted_start(is_rare, blk, formal)
            pb = formal - 1
            while pb - 1 >= 0 and blk[pb - 1] == prev:
                pb -= 1
            ce = formal
            while ce + 1 < len(sl) and blk[ce + 1] == cur:
                ce += 1
            ev_id += 1
            for rel in TRIAL_RANGE:
                i = align + rel
                if i < pb or i > ce or early_rare[i]:
                    continue
                trans.append(row(i, kind, rel))
        run = block_run.to_numpy()
        for a in np.flatnonzero(is_rare):
            ev_id += 1
            for rel in RARE_RANGE:
                i = a + rel
                if 0 <= i < len(sl) and run[i] == run[a]:
                    rare.append(row(i, "rare_long" if cat[a] == 1 else "rare_short", rel))
    columns = ['subject', 'session_id', 'cluster', 'event_id', 'kind', 'rel', 'category', 'valid', 'mouse', 'rare', 'bridge']
    return pd.DataFrame(trans, columns=columns), pd.DataFrame(rare, columns=columns)



def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(False)


def psychometric_panel(ax, data, k, title):
    for c in range(1, k+1):
        g = data[data.cluster_label.eq(c)].copy()
        if g.empty:
            continue
        g['_bin'] = assign_bins(g)
        b = bin_stats(g.assign(v=g.choice_long.astype(float)), 'v')
        col = COLORS[c-1]
        n_sessions = len(g[KEYS].drop_duplicates())
        ax.errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], fmt='o', color=col, ms=4,
                    capsize=2, lw=1.1, label=f'Cluster {c} ({n_sessions} sessions, {len(g)} trials)')
        popt = fit_data(g.duration_s.to_numpy(float), g.choice_long.to_numpy(float))
        if np.isfinite(popt).all():
            xs = np.linspace(.15, 2.35, 200)
            ax.plot(xs, logist(xs, *popt), color=col, lw=1.7)
    ax.axvline(1.25, color='.6', lw=.8, ls=':')
    ax.set_ylim(-.03, 1.03)
    ax.set_box_aspect(1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Stimulus duration (s)')
    ax.set_ylabel('P(choose long)')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    style(ax)


def event_panel(ax, rows, k, title, offsets, transition=False):
    xx = np.array(list(offsets))
    for c in range(1, k+1):
        g = rows[rows.cluster.eq(c) & rows.valid].copy()
        if g.empty:
            continue
        n_events = g.event_id.nunique()
        n_sessions = len(g[KEYS].drop_duplicates())
        if transition:
            g = add_bridge(g)
        grp = g.groupby('rel').mouse
        mu, se = grp.mean().reindex(xx), grp.sem().reindex(xx)
        col = COLORS[c-1]
        # Preserve missing positions rather than joining across an unobserved interval.
        ax.plot(xx, mu, color=col, lw=1.5, marker='o', ms=2.5,
                label=f'Cluster {c} ({n_sessions} sessions, {n_events} events)')
        ax.fill_between(xx, mu-se, mu+se, color=col, alpha=.15, lw=0)
        if transition and g.bridge.any() and np.isfinite(mu.get(-1, np.nan)):
            ax.scatter(-1, mu.loc[-1], s=30, facecolors='white', edgecolors=col, linewidths=1.3, zorder=5)
    ax.axvline(0, color='black', ls='--', lw=.9)
    if transition:
        ax.axvspan(0, xx[-1]+.5, color='gray', alpha=.08, lw=0)
        ax.set_xlim(xx[0]-.5, xx[-1]+.5)
    else:
        ax.set_xticks([-2, -1, 0, 1, 2], ['−2', '−1', 'rare', '+1', '+2'])
    if not transition:
        ax.set_box_aspect(1)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('P(correct)')
    ax.set_xlabel('Trial relative to adjusted block transition' if transition else 'Trial relative to rare trial')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc='lower left')
    style(ax)


def make_page(tagged, k):
    trans, rare = collect_events(tagged)
    valid = tagged[tagged.model_evaluated.astype(bool) & tagged.choice_long.isin([0, 1]) & tagged.warm_up.eq(0)]
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 12, height_ratios=[1.55, .78, .78, 1.5], left=.06, right=.98,
                          top=.90, bottom=.075, hspace=.52, wspace=1.15)
    for j, (title, g) in enumerate([('All blocks', valid), ('Short block', valid[valid.true_block.eq('S')]),
                                   ('Long block', valid[valid.true_block.eq('L')])]):
        psychometric_panel(fig.add_subplot(gs[0, j*4:(j+1)*4]), g, k, title)
    for j, (kind, cat) in enumerate([('S2L', 0), ('S2L', 1), ('L2S', 0), ('L2S', 1)]):
        title = f'{"Short→Long" if kind == "S2L" else "Long→Short"} · {"short" if cat == 0 else "long"} trials'
        ax = fig.add_subplot(gs[1+j//2, (j%2)*6:(j%2+1)*6])
        event_panel(ax, trans[trans.kind.eq(kind) & trans.category.eq(cat)],
                    k, title, TRIAL_RANGE, transition=True)
        # Reduce width alone to exactly 60% of the previous slot, keeping it centred.
        pos = ax.get_position()
        ax.set_position([pos.x0 + .20*pos.width, pos.y0, .60*pos.width, pos.height])
    for j, kind in enumerate(['rare_long', 'rare_short']):
        title = 'Rare long trial (in short block)' if kind == 'rare_long' else 'Rare short trial (in long block)'
        event_panel(fig.add_subplot(gs[3, 2+j*4:6+j*4]), rare[rare.kind.eq(kind)], k, title, RARE_RANGE)
    counts = tagged[KEYS+['cluster_label']].drop_duplicates().groupby('cluster_label').size().to_dict()
    fig.suptitle(f'Pooled mouse behaviour — K={k}, clusters superimposed', fontsize=16, fontweight='bold', y=.98)
    fig.text(.5, .955, 'Psychometrics · block transitions · rare trials     |     ' +
             '   '.join(f'Cluster {c}: {counts[c]} sessions' for c in range(1,k+1)), ha='center', fontsize=11)
    fig.text(.06, .025,
             'Observed held-out test-session choices, pooled across MC06/MC07/MC08. Error bars/bands: trial/event SEM. '
             'Open transition markers: previous matching trial. Exploratory partitions; cluster numbers are local to K.',
             fontsize=9, color='.3')
    return fig, trans, rare


def annotate_mouse_support(fig, labels):
    """Make subject-confined groups visible on behavioral figures too."""
    counts = pd.crosstab(labels.subject, labels.cluster_label)
    missing = counts.columns[(counts == 0).any(axis=0)].tolist()
    if missing:
        fig.text(.5, .938, 'CAUTION: cluster(s) ' + ', '.join(map(str, missing)) +
                 ' lack representation from every mouse; do not interpret as a shared strategy.',
                 ha='center', color='#a32626', fontsize=10)


def figures(all_labels, out, allow_subset=False):
    # Load only observed trial fields from the same canonical sliding-window files.
    frames = [t.assign(subject=s) for s in all_labels.subject.unique() for _, t in test_sessions(s)]
    data = pd.concat(frames, ignore_index=True)
    if allow_subset:
        # Explicit candidate cohort only; attach_labels still rejects missing/unknown labels.
        requested = all_labels[KEYS].drop_duplicates()
        data = data.merge(requested, on=KEYS, how='inner', validate='many_to_one')
    if data.duplicated(KEYS+['trial_index']).any():
        raise ValueError('Duplicated pooled test-session trials')
    from . import latency
    reaction = latency.load_trials(all_labels, out)
    pages, latency_summaries = [], []
    with PdfPages(out/'cluster_behavior_pooled.pdf') as pdf:
        for k in (2,3):
            labels = all_labels[all_labels.K.eq(k)]
            # Validate full cluster membership before rendering.
            list(cluster_frames(data, labels, k))
            tagged = attach_labels(data, labels)
            fig, trans, rare = make_page(tagged, k)
            annotate_mouse_support(fig, labels)
            pdf.savefig(fig)
            if k == 2:
                fig.savefig(out/'cluster_behavior_pooled.png', dpi=130)
            plt.close(fig)
            counts = labels.groupby('cluster_label').size().to_dict()
            pages.append(dict(page=len(pages)+1, analysis='choice_behaviour', K=k, n_sessions=len(labels), n_trials=len(data),
                              cluster_sizes={int(c):int(n) for c,n in counts.items()},
                              n_transitions=int(trans.event_id.nunique()), n_rare_events=int(rare.event_id.nunique())))
            print(f'Mouse-only behaviour page {len(pages)}: K={k}, cluster sizes={counts}', flush=True)
        for k in (2, 3):
            labels = all_labels[all_labels.K.eq(k)]
            # A session may have zero eligible latencies after filtering; retain its audit, not a fake zero RT.
            tagged = reaction.merge(labels[KEYS+['cluster_label']],on=KEYS,how='left',validate='many_to_one')
            if tagged.cluster_label.isna().any():
                raise ValueError('Reaction-time trials lack cluster labels')
            fig, summary, sizes = latency.make_page(tagged, k)
            annotate_mouse_support(fig, labels)
            pdf.savefig(fig)
            plt.close(fig)
            latency_summaries.append(summary)
            pages.append(dict(page=len(pages)+1,analysis='reaction_time',K=k,
                              n_sessions=len(tagged[KEYS].drop_duplicates()),n_trials=len(tagged),
                              cluster_sizes={int(c):int(n) for c,n in sizes.items()}))
            print(f'Reaction-time page {len(pages)}: K={k}, eligible trials={len(tagged)}',flush=True)
    pd.concat(latency_summaries,ignore_index=True).to_csv(out/'reaction_time_binned.csv',index=False)
    pd.DataFrame(pages).to_csv(out/'behavior_pages.csv', index=False)
    return pages


def refresh(out=OUT):
    labels = pd.read_csv(out/'session_clusters.csv')
    pages = figures(labels, out)
    summary = json.loads((out/'summary.json').read_text())
    from .latency import METHOD
    summary.update(behavior_pages=pages, behavior_visualization=VISUALIZATION, behavior_weighting=WEIGHTING,
                   reaction_time=METHOD)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    refresh()
